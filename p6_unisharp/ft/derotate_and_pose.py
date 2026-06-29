#!/usr/bin/env python3
"""A-tier step 1: de-rotate each ERP pano into a common (world) orientation so the
residual between frames is translation-only — the contract UniSHARP's panorama
training (SimPanorama, fixed orientation) expects — and emit a pose CSV.

Input : pano_cams_scene_<S>.json  (per-frame R_wp [world->pano], C [world center])
        panoramas/pano_*.jpg
Output: <out_rgb>/<frame:05d>.jpg   canonical ERP (world-aligned axes)
        <out_pose>                  CSV: frame,x,y,z  (= C * pos_scale, metres)

ERP convention (matches UniK3D Spherical: lon=atan2(x,z), lat=asin(y)):
  pixel(u,v) -> lon=(u+.5)/W*2pi-pi ,  lat=pi/2-(v+.5)/H*pi
  world ray  d_w=[cos lat sin lon, sin lat, cos lat cos lon]
  pano  ray  d_p = R_wp @ d_w   ->  sample source pano at (lon_p,lat_p)
A uniform convention offset is harmless (it only redefines the canonical frame);
what matters is R_wp is applied consistently to every frame.
"""
import argparse, csv, json, re
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--cams", required=True)
ap.add_argument("--panodir", required=True)
ap.add_argument("--out-rgb", required=True)
ap.add_argument("--out-pose", required=True)
ap.add_argument("--pos-scale", type=float, default=1.0,
                help="VGGT->metre scale for C. See task_ft.md §3.2; tune so adjacent "
                     "frame steps match real flight (≈0.1-1m).")
ap.add_argument("--H", type=int, default=1024)
ap.add_argument("--W", type=int, default=2048)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
a = ap.parse_args()

dev = torch.device(a.device)
cams = json.load(open(a.cams))["cameras"]
panodir = Path(a.panodir)
out_rgb = Path(a.out_rgb); out_rgb.mkdir(parents=True, exist_ok=True)
Path(a.out_pose).parent.mkdir(parents=True, exist_ok=True)

H, W = a.H, a.W
# canonical output ray grid (world frame), shape [H,W,3]
vv, uu = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")
lon = (uu + 0.5) / W * 2 * np.pi - np.pi
lat = np.pi / 2 - (vv + 0.5) / H * np.pi
clat = torch.cos(lat)
dw = torch.stack([clat * torch.sin(lon), torch.sin(lat), clat * torch.cos(lon)], -1)  # [H,W,3]

def frame_idx_from_name(name):
    m = re.search(r"(\d+)", Path(name).stem)
    return int(m.group(1)) if m else None

def find_pano(idx, image_field):
    # prefer the json's image field, else pano_<idx>.jpg variants
    cand = []
    if image_field:
        cand.append(panodir / Path(image_field).name)
    cand += [panodir / f"pano_{idx:04d}.jpg", panodir / f"pano_{idx:05d}.jpg"]
    for c in cand:
        if c.exists():
            return c
    return None

rows = []
n_ok = 0
for c in cams:
    idx = int(c["idx"])
    src = find_pano(idx, c.get("image"))
    if src is None:
        print(f"[skip] idx={idx}: no pano file"); continue
    Rwp = torch.tensor(np.array(c["R_wp"], dtype=np.float32), device=dev)  # world->pano
    C = np.array(c["C"], dtype=np.float64) * a.pos_scale

    dp = torch.einsum("ij,hwj->hwi", Rwp, dw)            # pano-frame rays [H,W,3]
    lon_p = torch.atan2(dp[..., 0], dp[..., 2])
    lat_p = torch.asin(dp[..., 1].clamp(-1 + 1e-6, 1 - 1e-6))
    # source pixel (wrap longitude); normalized grid for grid_sample in [-1,1]
    u_p = (lon_p + np.pi) / (2 * np.pi) * W            # [0,W)
    u_p = torch.remainder(u_p, W)
    v_p = (np.pi / 2 - lat_p) / np.pi * H              # [0,H)
    gx = u_p / (W - 1) * 2 - 1
    gy = v_p / (H - 1) * 2 - 1
    grid = torch.stack([gx, gy], -1).unsqueeze(0)       # [1,H,W,2]

    img = Image.open(src).convert("RGB").resize((W, H), Image.BILINEAR)
    t = torch.from_numpy(np.asarray(img, np.float32)).permute(2, 0, 1).unsqueeze(0).to(dev) / 255.0
    out = F.grid_sample(t, grid, mode="bilinear", padding_mode="border", align_corners=True)
    arr = (out[0].permute(1, 2, 0).clamp(0, 1) * 255).round().byte().cpu().numpy()
    Image.fromarray(arr).save(out_rgb / f"{idx:05d}.jpg", quality=95)
    rows.append((idx, float(C[0]), float(C[1]), float(C[2])))
    n_ok += 1

rows.sort()
with open(a.out_pose, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["frame", "x", "y", "z"]); w.writerows(rows)
print(f"[derotate] {n_ok} frames -> {out_rgb}  |  pose -> {a.out_pose}  (pos_scale={a.pos_scale})")
