#!/usr/bin/env python3
"""
全景图 (equirectangular) → 多张透视图 (perspective)

针对 8K 全景无人机(如影翎360)输出设计。
将每张 equirectangular 全景拆分为多张针孔相机透视图，
供 COLMAP + 3DGS 流水线使用。

用法:
    python pano_to_perspective.py \
        --input /path/to/panoramas \
        --output /path/to/scene/input \
        --fov 90 --size 2048 --preset dense
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np


# ── 预设的视角采样方案 ──────────────────────────────────────────────

PRESETS = {
    # 6 面 cubemap，最快但覆盖角度离散
    "cubemap": [
        (0, 0), (90, 0), (180, 0), (270, 0),   # 前右后左
        (0, 90), (0, -90),                       # 上下
    ],
    # 14 个方向: cubemap + 8 个 45° 对角，推荐日常使用
    "standard": [
        (0, 0), (90, 0), (180, 0), (270, 0),
        (0, 90), (0, -90),
        (45, 30), (135, 30), (225, 30), (315, 30),
        (45, -30), (135, -30), (225, -30), (315, -30),
    ],
    # 26 个方向: 更密采样，大场景或需要高重叠率时使用
    "dense": [
        # 水平一圈 (pitch=0), 每 45°
        (0, 0), (45, 0), (90, 0), (135, 0),
        (180, 0), (225, 0), (270, 0), (315, 0),
        # 上仰 30°
        (0, 30), (60, 30), (120, 30), (180, 30), (240, 30), (300, 30),
        # 下俯 30°
        (0, -30), (60, -30), (120, -30), (180, -30), (240, -30), (300, -30),
        # 上仰 60°
        (0, 60), (120, 60), (240, 60),
        # 下俯 60°
        (0, -60), (120, -60), (240, -60),
    ],
}


def rotation_matrix(yaw_deg, pitch_deg):
    """构造从相机坐标系到世界坐标系的旋转矩阵 (yaw-pitch 顺序)"""
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)

    # 绕 Y 轴旋转 (yaw)
    Ry = np.array([
        [np.cos(yaw),  0, np.sin(yaw)],
        [0,            1, 0           ],
        [-np.sin(yaw), 0, np.cos(yaw)],
    ])

    # 绕 X 轴旋转 (pitch)
    Rx = np.array([
        [1, 0,             0            ],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch) ],
    ])

    return Ry @ Rx


def equirect_to_perspective(equirect, fov_deg, yaw_deg, pitch_deg, out_size):
    """
    从 equirectangular 全景图中提取一张透视图。

    参数:
        equirect:   输入全景图 (H, W, 3), equirectangular 投影
        fov_deg:    输出透视图的水平视场角 (度)
        yaw_deg:    水平朝向角 (度), 0=前, 90=右, 180=后, 270=左
        pitch_deg:  俯仰角 (度), 正=上, 负=下
        out_size:   输出尺寸 (width, height)

    返回:
        透视图 (out_h, out_w, 3)
    """
    h_eq, w_eq = equirect.shape[:2]
    out_w, out_h = out_size

    # 针孔相机焦距 (像素)
    f = (out_w / 2) / np.tan(np.radians(fov_deg) / 2)

    # 输出图像每个像素对应的 3D 射线方向 (相机坐标系, z 轴朝前)
    u = np.arange(out_w, dtype=np.float64) - out_w / 2 + 0.5
    v = np.arange(out_h, dtype=np.float64) - out_h / 2 + 0.5
    uu, vv = np.meshgrid(u, v)

    # 相机坐标系下的方向: (x=右, y=下, z=前)
    dirs = np.stack([uu, vv, np.full_like(uu, f)], axis=-1)
    dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)

    # 旋转到世界坐标系
    R = rotation_matrix(yaw_deg, pitch_deg)
    dirs_world = dirs @ R.T  # (out_h, out_w, 3)

    x, y, z = dirs_world[..., 0], dirs_world[..., 1], dirs_world[..., 2]

    # 世界坐标 → equirectangular 坐标
    # longitude: [-pi, pi], latitude: [-pi/2, pi/2]
    lon = np.arctan2(x, z)       # 水平角
    lat = -np.arcsin(np.clip(y, -1, 1))  # 垂直角 (y向下为正, lat向上为正)

    # 映射到像素坐标
    map_x = ((lon / np.pi + 1) / 2 * w_eq).astype(np.float32)
    map_y = ((1 - (lat / (np.pi / 2) + 1) / 2) * h_eq).astype(np.float32)

    # 用 remap 采样
    result = cv2.remap(
        equirect, map_x, map_y,
        interpolation=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_WRAP,
    )
    return result


def get_camera_intrinsics(fov_deg, out_size):
    """返回 COLMAP 格式的相机内参 (PINHOLE 模型)"""
    out_w, out_h = out_size
    f = (out_w / 2) / np.tan(np.radians(fov_deg) / 2)
    cx = out_w / 2.0
    cy = out_h / 2.0
    return f, f, cx, cy


def _write_single_view(args):
    """Worker: render one (pano, yaw, pitch) view and write it to disk.

    Used by the parallel pool; arguments are packed in a tuple so this
    works with joblib / multiprocessing.Pool without lambda pickling.
    """
    equirect, pano_name, fov_deg, yaw, pitch, out_size, output_dir, quality = args
    persp = equirect_to_perspective(equirect, fov_deg, yaw, pitch, out_size)
    out_name = f"{pano_name}_y{yaw:+04d}_p{pitch:+03d}.jpg"
    out_path = os.path.join(output_dir, out_name)
    cv2.imwrite(out_path, persp, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return {
        "filename": out_name,
        "source_pano": pano_name,
        "yaw": yaw,
        "pitch": pitch,
        "fov": fov_deg,
    }


def process_panorama(pano_path, output_dir, fov_deg, out_size, directions, pano_idx,
                     total_panos, quality):
    """处理一张全景图，输出多张透视图"""
    pano_name = Path(pano_path).stem
    equirect = cv2.imread(str(pano_path))
    if equirect is None:
        print(f"  [WARN] 无法读取: {pano_path}, 跳过")
        return []

    h, w = equirect.shape[:2]
    print(f"  [{pano_idx+1}/{total_panos}] {pano_name} ({w}x{h}) → {len(directions)} 张透视图")

    results = []
    for i, (yaw, pitch) in enumerate(directions):
        persp = equirect_to_perspective(equirect, fov_deg, yaw, pitch, out_size)

        out_name = f"{pano_name}_y{yaw:+04d}_p{pitch:+03d}.jpg"
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, persp, [cv2.IMWRITE_JPEG_QUALITY, quality])

        results.append({
            "filename": out_name,
            "source_pano": pano_name,
            "yaw": yaw,
            "pitch": pitch,
            "fov": fov_deg,
        })

    return results


def extract_gps_from_exif(pano_path):
    """尝试从 EXIF 中读取 GPS 坐标, 返回 (lat, lon, alt) 或 None"""
    try:
        import piexif
        exif_dict = piexif.load(str(pano_path))
        gps = exif_dict.get("GPS", {})
        if not gps:
            return None

        def to_deg(val):
            d, m, s = val
            return d[0]/d[1] + m[0]/m[1]/60 + s[0]/s[1]/3600

        lat = to_deg(gps[piexif.GPSIFD.GPSLatitude])
        if gps.get(piexif.GPSIFD.GPSLatitudeRef, b'N') == b'S':
            lat = -lat

        lon = to_deg(gps[piexif.GPSIFD.GPSLongitude])
        if gps.get(piexif.GPSIFD.GPSLongitudeRef, b'E') == b'W':
            lon = -lon

        alt = 0.0
        if piexif.GPSIFD.GPSAltitude in gps:
            alt_val = gps[piexif.GPSIFD.GPSAltitude]
            alt = alt_val[0] / alt_val[1]
            if gps.get(piexif.GPSIFD.GPSAltitudeRef, 0) == 1:
                alt = -alt

        return (lat, lon, alt)
    except Exception:
        return None


def write_geo_registration(metadata_list, output_dir, pano_dir):
    """
    生成 COLMAP geo-registration 文件。
    同一张全景拆出的所有透视图共享同一个 GPS 坐标。
    """
    geo_lines = []
    seen_panos = {}

    for item in metadata_list:
        pano_name = item["source_pano"]
        if pano_name not in seen_panos:
            # 查找原始全景图并提取 GPS
            for ext in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG", ".tif", ".tiff"]:
                pano_path = Path(pano_dir) / (pano_name + ext)
                if pano_path.exists():
                    gps = extract_gps_from_exif(pano_path)
                    seen_panos[pano_name] = gps
                    break
            else:
                seen_panos[pano_name] = None

        gps = seen_panos[pano_name]
        if gps is not None:
            lat, lon, alt = gps
            # COLMAP geo.txt 格式: image_name X Y Z
            # 这里简单用经纬度作为局部坐标(小范围场景误差可忽略)
            # 大场景应转 UTM
            geo_lines.append(f"{item['filename']} {lon:.10f} {lat:.10f} {alt:.4f}")

    if geo_lines:
        geo_path = os.path.join(os.path.dirname(output_dir), "geo.txt")
        with open(geo_path, "w") as f:
            f.write("# image_name X Y Z\n")
            for line in geo_lines:
                f.write(line + "\n")
        print(f"\n已生成 GPS 注册文件: {geo_path} ({len(geo_lines)} 条)")
        return geo_path
    else:
        print("\n未检测到 GPS 信息，跳过 geo-registration")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="全景图 (equirectangular) → 多张透视图 (perspective)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法
  python pano_to_perspective.py -i ./panoramas -o ./scene/input

  # 指定参数
  python pano_to_perspective.py -i ./panoramas -o ./scene/input \\
      --fov 90 --size 2048 --preset dense --quality 95

  # 自定义视角
  python pano_to_perspective.py -i ./panoramas -o ./scene/input \\
      --directions "0,0 90,0 180,0 270,0 0,45 0,-45"
        """,
    )
    parser.add_argument("-i", "--input", required=True, help="全景图目录")
    parser.add_argument("-o", "--output", required=True, help="透视图输出目录 (建议为 scene/input)")
    parser.add_argument("--fov", type=float, default=90, help="透视图视场角 (度, 默认 90)")
    parser.add_argument("--size", type=int, default=2048, help="输出图像边长 (默认 2048)")
    parser.add_argument("--preset", default="standard", choices=PRESETS.keys(),
                        help="视角采样预设 (默认 standard)")
    parser.add_argument("--directions", type=str, default=None,
                        help="自定义视角, 格式: 'yaw1,pitch1 yaw2,pitch2 ...'")
    parser.add_argument("--quality", type=int, default=95, help="JPEG 输出质量 (默认 95)")
    parser.add_argument("--ext", type=str, default="jpg,jpeg,png,tif,tiff",
                        help="输入文件扩展名过滤 (逗号分隔)")
    parser.add_argument("--workers", type=int, default=0,
                        help="并行 worker 数 (0=auto, CPU 核数-1)")
    args = parser.parse_args()

    # 解析视角方向
    if args.directions:
        directions = []
        for pair in args.directions.split():
            yaw, pitch = map(float, pair.split(","))
            directions.append((yaw, pitch))
    else:
        directions = PRESETS[args.preset]

    # 收集全景图文件
    input_dir = Path(args.input)
    exts = set(args.ext.lower().split(","))
    pano_files = sorted([
        f for f in input_dir.iterdir()
        if f.suffix.lstrip(".").lower() in exts and f.is_file()
    ])

    if not pano_files:
        print(f"错误: 在 {input_dir} 中未找到全景图文件")
        sys.exit(1)

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    out_size = (args.size, args.size)
    fx, fy, cx, cy = get_camera_intrinsics(args.fov, out_size)

    print(f"全景图目录: {input_dir}")
    print(f"找到 {len(pano_files)} 张全景图")
    print(f"每张拆分为 {len(directions)} 张透视图")
    print(f"输出大小: {args.size}x{args.size}, FOV: {args.fov}°")
    print(f"焦距: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
    print(f"预计总输出: {len(pano_files) * len(directions)} 张")
    print()

    # 处理每张全景 — 并行: 一张全景一次只解码一次, 然后并行渲染 14 个 view
    import time
    from multiprocessing import Pool, cpu_count

    n_workers = args.workers if args.workers > 0 else max(1, cpu_count() - 1)
    print(f"使用 {n_workers} 个并行 worker")

    all_metadata = []
    t0 = time.time()
    with Pool(n_workers) as pool:
        for idx, pano_file in enumerate(pano_files):
            pano_name = Path(pano_file).stem
            equirect = cv2.imread(str(pano_file))
            if equirect is None:
                print(f"  [WARN] 无法读取: {pano_file}, 跳过")
                continue
            h, w = equirect.shape[:2]
            print(f"  [{idx+1}/{len(pano_files)}] {pano_name} ({w}x{h}) → {len(directions)} 透视图",
                  flush=True)
            tasks = [
                (equirect, pano_name, args.fov, yaw, pitch, out_size, args.output, args.quality)
                for yaw, pitch in directions
            ]
            # imap_unordered 让 worker 一边完成一边返回, 顺序无所谓 (我们最后按文件名取顺序)
            for meta in pool.imap_unordered(_write_single_view, tasks, chunksize=2):
                all_metadata.append(meta)

    elapsed = time.time() - t0
    print(f"\n并行处理完成: {len(all_metadata)} 张, 用时 {elapsed:.1f}s")

    # 保存元数据
    meta_path = os.path.join(os.path.dirname(args.output), "pano_metadata.json")
    with open(meta_path, "w") as f:
        json.dump({
            "fov_deg": args.fov,
            "image_size": list(out_size),
            "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
            "directions": directions,
            "images": all_metadata,
        }, f, indent=2)
    print(f"\n元数据已保存: {meta_path}")

    # 生成 GPS 注册文件
    write_geo_registration(all_metadata, args.output, str(input_dir))

    print(f"\n完成! 共输出 {len(all_metadata)} 张透视图到 {args.output}")
    print(f"\n下一步: 运行 COLMAP")
    print(f"  python convert.py -s {Path(args.output).parent}")


if __name__ == "__main__":
    main()
