#!/usr/bin/env python3
"""P1.3c: direct equirectangular 3DGS training.

Trains directly on panoramas (one LONLAT camera per pano) using the ported
diff_gaussian_rasterization_pano. Reuses scene.gaussian_model.GaussianModel
for the parameter container + densification (projection-agnostic).

Usage: train_pano.py <pano_cams.json> <out_dir> [iters=7000] [width=1024]
"""
import sys, os, json, math, random, numpy as np, torch
from types import SimpleNamespace
from PIL import Image

sys.path.insert(0, "/workspace/gaussian-splatting")
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import BasicPointCloud
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from plyfile import PlyData
from diff_gaussian_rasterization_pano import (
    GaussianRasterizationSettings, GaussianRasterizer, CAMERA_LONLAT)

cams_json, out_dir = sys.argv[1], sys.argv[2]
ITERS = int(sys.argv[3]) if len(sys.argv) > 3 else 7000
W = int(sys.argv[4]) if len(sys.argv) > 4 else 1024
H = W // 2
os.makedirs(out_dir, exist_ok=True)
dev = "cuda"

meta = json.load(open(cams_json))

# --- init point cloud from VGGT points3D.ply ---
ply = PlyData.read(meta["point_cloud"]); v = ply["vertex"]
pts = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
if "red" in v.data.dtype.names:
    cols = np.stack([v["red"], v["green"], v["blue"]], 1).astype(np.float32) / 255.0
else:
    cols = np.full_like(pts, 0.5)
pcd = BasicPointCloud(points=pts, colors=cols, normals=np.zeros_like(pts))

# --- cameras (equirect): precompute world_view_transform + center + GT image ---
class PanoCam:
    pass
def load_cam(c):
    R = np.array(c["R_wp"], dtype=np.float32)      # world->view rotation
    T = np.array(c["T"], dtype=np.float32)
    W2V = np.eye(4, dtype=np.float32); W2V[:3, :3] = R; W2V[:3, 3] = T
    cam = PanoCam()
    cam.world_view_transform = torch.tensor(W2V.T, device=dev)  # glm column-major
    cam.camera_center = torch.tensor(np.array(c["C"], np.float32), device=dev)
    im = Image.open(c["image"]).convert("RGB").resize((W, H), Image.LANCZOS)
    cam.image = torch.tensor(np.asarray(im), dtype=torch.float32, device=dev).permute(2, 0, 1) / 255.0
    cam.image_name = f"pano_{c['idx']:04d}"
    return cam

all_cams = [load_cam(c) for c in meta["cameras"]]
# hold out every 8th as test
test_cams = all_cams[::8]
train_cams = [c for i, c in enumerate(all_cams) if i % 8 != 0]
print(f"{len(all_cams)} pano cams -> {len(train_cams)} train / {len(test_cams)} test | img {W}x{H}")

# --- gaussian model ---
gaussians = GaussianModel(3)
cam_infos = [SimpleNamespace(image_name=c.image_name) for c in all_cams]
gaussians.create_from_pcd(pcd, cam_infos, meta["cameras_extent"])
opt = SimpleNamespace(
    iterations=ITERS, position_lr_init=0.00016, position_lr_final=0.0000016,
    position_lr_delay_mult=0.01, position_lr_max_steps=ITERS, feature_lr=0.0025,
    opacity_lr=0.025, scaling_lr=0.005, rotation_lr=0.001, percent_dense=0.01,
    exposure_lr_init=0.0, exposure_lr_final=0.0, exposure_lr_delay_steps=0,
    exposure_lr_delay_mult=0.0, optimizer_type="default")
gaussians.training_setup(opt)
bg = torch.zeros(3, device=dev)

def render(cam):
    sp = torch.zeros_like(gaussians.get_xyz, requires_grad=True) + 0
    try: sp.retain_grad()
    except: pass
    st = GaussianRasterizationSettings(
        image_height=H, image_width=W, tanfovx=1.0, tanfovy=1.0, bg=bg,
        scale_modifier=1.0, viewmatrix=cam.world_view_transform,
        projmatrix=cam.world_view_transform, sh_degree=gaussians.active_sh_degree,
        campos=cam.camera_center, prefiltered=False,
        camera_type=CAMERA_LONLAT, render_depth=False)
    r = GaussianRasterizer(raster_settings=st)
    color, radii = r(means3D=gaussians.get_xyz, means2D=sp,
                     opacities=gaussians.get_opacity, shs=gaussians.get_features,
                     scales=gaussians.get_scaling, rotations=gaussians.get_rotation)
    return color, radii, sp

DENSIFY_UNTIL = int(ITERS * 0.5); DENSIFY_FROM = 500; DENSIFY_INT = 100
OPACITY_RESET = 3000
stack = []
import time as _time; _t0 = _time.time()
for it in range(1, ITERS + 1):
    gaussians.update_learning_rate(it)
    if it % 1000 == 0: gaussians.oneupSHdegree()
    if not stack: stack = train_cams.copy(); random.shuffle(stack)
    cam = stack.pop()
    color, radii, sp = render(cam)
    gt = cam.image
    Ll1 = l1_loss(color, gt)
    loss = (1.0 - 0.2) * Ll1 + 0.2 * (1.0 - ssim(color, gt))
    loss.backward()
    with torch.no_grad():
        if it < DENSIFY_UNTIL:
            vis = radii > 0
            gaussians.max_radii2D[vis] = torch.max(gaussians.max_radii2D[vis], radii[vis])
            gaussians.add_densification_stats(sp, vis)
            if it > DENSIFY_FROM and it % DENSIFY_INT == 0:
                size_thr = 20 if it > OPACITY_RESET else None
                gaussians.densify_and_prune(0.0002, 0.005, meta["cameras_extent"], size_thr, radii)
            if it % OPACITY_RESET == 0:
                gaussians.reset_opacity()
        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)
    if it % 500 == 0:
        print(f"  iter {it}: loss {loss.item():.4f}  N={gaussians.get_xyz.shape[0]}", flush=True)

torch.cuda.synchronize(); _dt = _time.time() - _t0
print(f"[TIME] LONLAT trained {ITERS} it in {_dt:.1f}s = {ITERS/_dt:.1f} it/s", flush=True)

# --- save ply + eval ---
pc_dir = os.path.join(out_dir, f"point_cloud/iteration_{ITERS}")
os.makedirs(pc_dir, exist_ok=True)
gaussians.save_ply(os.path.join(pc_dir, "point_cloud.ply"))

try:
    from lpipsPyTorch import lpips as _lpips
    HAVE_LPIPS = True
except Exception:
    HAVE_LPIPS = False

with torch.no_grad():
    ps, ss, lp = [], [], []
    for cam in test_cams:
        color, _, _ = render(cam)
        color = color.clamp(0, 1)
        ps.append(psnr(color, cam.image).mean().item())
        ss.append(ssim(color, cam.image).item())
        if HAVE_LPIPS:
            lp.append(_lpips(color[None], cam.image[None], net_type='vgg').item())
    res = {
        "scene": os.path.basename(os.path.dirname(out_dir)) or out_dir,
        "method": "direct-pano",
        "iterations": ITERS, "train_res": [W, H],
        "n_gaussians": int(gaussians.get_xyz.shape[0]),
        "n_train": len(train_cams), "n_test": len(test_cams),
        "PSNR": float(np.mean(ps)),
        "SSIM": float(np.mean(ss)),
        "LPIPS": (float(np.mean(lp)) if lp else None),
    }
    import json as _json
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        _json.dump(res, f, indent=1)
    print(f"[EVAL] held-out test  PSNR={res['PSNR']:.3f}  SSIM={res['SSIM']:.4f}  "
          f"LPIPS={res['LPIPS'] if res['LPIPS'] is None else round(res['LPIPS'],4)}  "
          f"over {len(test_cams)} panos")
print(f"[DONE] N={gaussians.get_xyz.shape[0]} gaussians -> {pc_dir}/point_cloud.ply")
