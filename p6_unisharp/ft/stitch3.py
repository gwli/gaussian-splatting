#!/usr/bin/env python3
"""Rig-calibrated ERP stitcher, exact COLMAP OPENCV_FISHEYE convention.
ERP frame := down-cam frame via R_e2c=[[1,0,0],[0,0,1],[0,-1,0]] (matches
rig_solve pose json). Up rays: d_up = R_rig @ d_down.  usage: stitch3.py <outdir> [N]
"""
import sys, os, math
import numpy as np, torch, torch.nn.functional as Fn
from PIL import Image

ROOT = "/w"; dev = "cuda" if torch.cuda.is_available() else "cpu"
z = np.load(f"{ROOT}/p3_pano/rig023.npz")
R_rig = torch.tensor(z["R_rig"], dtype=torch.float32, device=dev)
cal_d, cal_u = z["cal_d"], z["cal_u"]
DOWN = f"{ROOT}/data/8kpano/scenes/fish023/images"
UP = f"{ROOT}/data/8kpano/scenes/fish023/images_up"
TH_MAX = math.radians(100.0)
H, W = 2048, 4096
outdir = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 240
os.makedirs(outdir, exist_ok=True)

vv, uu = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")
lon = (uu+0.5)/W*2*math.pi - math.pi
lat = (vv+0.5)/H*math.pi - math.pi/2   # kernel y-down: row0 -> lat=-pi/2 (y=-1)
d_erp = torch.stack([torch.cos(lat)*torch.sin(lon), torch.sin(lat), torch.cos(lat)*torch.cos(lon)], -1)
# R_k2c: kernel pano frame -> down-cam. cols: x=(1,0,0), y=(0,0,1), z=(0,-1,0); det=+1
R_e2c = torch.tensor([[1,0,0],[0,0,-1],[0,1,0]], dtype=torch.float32, device=dev)
d_dn = torch.einsum("ij,hwj->hwi", R_e2c, d_erp)
d_up = torch.einsum("ij,hwj->hwi", R_rig, d_dn)

def proj(d, cal):
    fx, fy, cx, cy, k1, k2, k3, k4 = cal
    x, y, zc = d[..., 0], d[..., 1], d[..., 2]
    hyp = torch.sqrt(x*x + y*y).clamp(min=1e-9)
    th = torch.atan2(hyp, zc)
    t2 = th*th
    r = th*(1 + k1*t2 + k2*t2**2 + k3*t2**3 + k4*t2**4)
    u = fx*r*x/hyp + cx; v = fy*r*y/hyp + cy
    gx = u/(1920-1)*2 - 1; gy = v/(1920-1)*2 - 1
    return torch.stack([gx, gy], -1), th

gd, thd = proj(d_dn, cal_d)
gu, thu = proj(d_up, cal_u)
wd = ((TH_MAX-thd)/0.18).clamp(0, 1) * (thd < TH_MAX)
wu = ((TH_MAX-thu)/0.18).clamp(0, 1) * (thu < TH_MAX)
s = (wd+wu).clamp(min=1e-6)

def load(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), np.float32)).permute(2,0,1)[None].to(dev)/255

for k in range(1, N+1):
    a = Fn.grid_sample(load(f"{DOWN}/f_{k:04d}.jpg"), gd[None], mode="bilinear", padding_mode="zeros", align_corners=True)[0]
    b = Fn.grid_sample(load(f"{UP}/f_{k:04d}.jpg"), gu[None], mode="bilinear", padding_mode="zeros", align_corners=True)[0]
    out = (a*wd + b*wu)/s
    Image.fromarray((out.permute(1,2,0).clamp(0,1)*255).byte().cpu().numpy()).save(f"{outdir}/pano_{k:04d}.jpg", quality=92)
    if k % 80 == 0: print(k, flush=True)
print(f"[stitch3] {N} rig-calibrated ERPs -> {outdir}")
