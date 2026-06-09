#!/usr/bin/env python3
"""Sanity-render an equirectangular view from a trained .ply using the
OmniGS-derived LONLAT rasterizer. Validates the ported forward path.

Usage: python pano_render.py <ply> <out.png> [width=1024]
"""
import sys, math, torch, numpy as np
from plyfile import PlyData

sys.path.insert(0, "/workspace/gaussian-splatting")
from diff_gaussian_rasterization_pano import (
    GaussianRasterizationSettings, GaussianRasterizer, CAMERA_LONLAT)

ply_path, out_path = sys.argv[1], sys.argv[2]
W = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
H = W // 2

# --- load gaussians from ply ---
ply = PlyData.read(ply_path)
v = ply["vertex"]
xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
opacity = v["opacity"].astype(np.float32)[:, None]
scales = np.stack([v[f"scale_{i}"] for i in range(3)], 1).astype(np.float32)
rots = np.stack([v[f"rot_{i}"] for i in range(4)], 1).astype(np.float32)
# SH: f_dc (3) + f_rest (45) → degree 3
fdc = np.stack([v[f"f_dc_{i}"] for i in range(3)], 1).astype(np.float32)
rest_keys = [k for k in v.data.dtype.names if k.startswith("f_rest_")]
frest = np.stack([v[k] for k in sorted(rest_keys, key=lambda s: int(s.split("_")[-1]))], 1).astype(np.float32)
N = xyz.shape[0]
sh_deg = 3
features = np.concatenate([fdc[:, None, :], frest.reshape(N, -1, 3)], axis=1)  # (N, 16, 3)

dev = "cuda"
means3D = torch.tensor(xyz, device=dev)
opacity_t = torch.sigmoid(torch.tensor(opacity, device=dev))
scales_t = torch.exp(torch.tensor(scales, device=dev))
rots_t = torch.nn.functional.normalize(torch.tensor(rots, device=dev))
shs_t = torch.tensor(features, device=dev).contiguous()
means2D = torch.zeros_like(means3D, requires_grad=True)

# --- camera: place at median of points, identity orientation ---
c = torch.tensor(np.median(xyz, axis=0), device=dev)
Rt = torch.eye(4, device=dev)
Rt[:3, 3] = -c                       # world2view translation (R = I)
viewmatrix = Rt.transpose(0, 1).contiguous()   # repo glm convention (transposed)
projmatrix = viewmatrix               # unused for LONLAT
campos = c

settings = GaussianRasterizationSettings(
    image_height=H, image_width=W,
    tanfovx=1.0, tanfovy=1.0,         # unused for LONLAT
    bg=torch.zeros(3, device=dev),
    scale_modifier=1.0,
    viewmatrix=viewmatrix,
    projmatrix=projmatrix,
    sh_degree=sh_deg,
    campos=campos,
    prefiltered=False,
    camera_type=CAMERA_LONLAT,
    render_depth=False,
)
rasterizer = GaussianRasterizer(raster_settings=settings)
color, radii = rasterizer(
    means3D=means3D, means2D=means2D, opacities=opacity_t,
    shs=shs_t, scales=scales_t, rotations=rots_t)

img = (color.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
from PIL import Image
Image.fromarray(img).save(out_path)
print(f"OK rendered {W}x{H} equirect | {N} gaussians | radii>0: {(radii>0).sum().item()} "
      f"| nonzero px: {(img.sum(2)>0).mean()*100:.1f}%")
