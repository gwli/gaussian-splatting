#!/usr/bin/env python3
"""
测试全景图拆分功能。
生成一张带网格线的合成全景图，拆分后检查透视图是否正确。

用法:
    python test_pano_split.py
"""

import os
import sys
import numpy as np
import cv2

# 添加上级目录以导入 pano_to_perspective
sys.path.insert(0, os.path.dirname(__file__))
from pano_to_perspective import equirect_to_perspective, PRESETS


def make_test_panorama(width=4096, height=2048):
    """生成一张带经纬网格和方向标记的测试全景图"""
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # 背景渐变 (蓝天到绿地)
    for y in range(height):
        t = y / height
        if t < 0.5:
            img[y, :] = [int(200 * (1-2*t)), int(150 + 50*(1-2*t)), int(50 + 200*(1-2*t))]
        else:
            img[y, :] = [int(50 * (2*t-1)), int(100 + 100*(2-2*t)), int(50 * (2-2*t))]

    # 标准 equirectangular 约定: 图像中心 = lon=0° (正前方)
    # x=0 → lon=-180°, x=w/2 → lon=0°, x=w → lon=+180°
    def lon_to_x(lon_deg):
        """经度(-180~180) → 像素x坐标"""
        return int((lon_deg + 180) / 360 * width)

    # 经线 (每 30°)
    for lon_deg in range(-180, 180, 30):
        x = lon_to_x(lon_deg)
        cv2.line(img, (x, 0), (x, height), (200, 200, 200), 2)
        cv2.putText(img, f"{lon_deg}deg", (x+5, height//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # 纬线 (每 30°)
    for lat_deg in range(-60, 90, 30):
        y = int((90 - lat_deg) / 180 * height)
        cv2.line(img, (0, y), (width, y), (200, 200, 200), 2)
        cv2.putText(img, f"lat{lat_deg:+d}", (10, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # 方向标记 (yaw角: 0=前, 90=右, 180/-180=后, -90/270=左)
    markers = [
        (0, "FRONT", (0, 0, 255)),
        (90, "RIGHT", (0, 255, 0)),
        (180, "BACK", (255, 0, 0)),
        (-90, "LEFT", (255, 255, 0)),
    ]
    for lon_deg, label, color in markers:
        x = lon_to_x(lon_deg)
        y = height // 2
        cv2.putText(img, label, (x - 40, y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

    # 棋盘格 (每 10°)
    for lon in range(-180, 180, 10):
        for lat in range(-90, 90, 10):
            if ((lon + 180) // 10 + lat // 10) % 2 == 0:
                continue
            x1 = lon_to_x(lon)
            x2 = lon_to_x(lon + 10)
            y1 = int((90 - lat - 10) / 180 * height)
            y2 = int((90 - lat) / 180 * height)
            overlay = img[y1:y2, x1:x2].copy()
            cv2.rectangle(img, (x1, y1), (x2, y2), (80, 80, 80), -1)
            img[y1:y2, x1:x2] = cv2.addWeighted(overlay, 0.5, img[y1:y2, x1:x2], 0.5, 0)

    return img


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "test_output")
    os.makedirs(out_dir, exist_ok=True)

    # 生成测试全景图
    print("生成测试全景图...")
    pano = make_test_panorama(4096, 2048)
    pano_path = os.path.join(out_dir, "test_panorama.jpg")
    cv2.imwrite(pano_path, pano)
    print(f"  保存到: {pano_path}")

    # 拆分测试
    fov = 90
    size = (1024, 1024)
    directions = PRESETS["standard"]

    print(f"\n拆分为 {len(directions)} 张透视图 (FOV={fov}°, size={size[0]}x{size[1]})...")
    for i, (yaw, pitch) in enumerate(directions):
        persp = equirect_to_perspective(pano, fov, yaw, pitch, size)
        name = f"view_{i:02d}_y{yaw:+04d}_p{pitch:+03d}.jpg"
        path = os.path.join(out_dir, name)
        cv2.imwrite(path, persp)
        print(f"  [{i+1:2d}/{len(directions)}] yaw={yaw:+4d}° pitch={pitch:+3d}° → {name}")

    print(f"\n测试完成! 输出目录: {out_dir}")
    print("请检查:")
    print("  - test_panorama.jpg: 应有网格线和 FRONT/RIGHT/BACK/LEFT 标记")
    print("  - view_00_*: 应看到 FRONT 标记和直线网格")
    print("  - view_01_*: 应看到 RIGHT 标记")
    print("  - 所有透视图中直线应保持为直线 (无弯曲)")


if __name__ == "__main__":
    main()
