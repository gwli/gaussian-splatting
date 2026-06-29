#!/usr/bin/env python3
"""A-tier step 2: pseudo metric depth for the canonical ERPs via UniK3D (vitl) —
the same backbone UniSHARP/re10k use for pseudo-depth GT. Sky & far pixels are
set to 0 (= invalid, the convention SimPanorama / the losses expect).

Input : <rgb_dir>/<frame>.jpg   (world-aligned ERP from derotate_and_pose.py)
Output: <rgb_dir>/depth/<frame>.npy   float32 [H,W] radial distance, 0 = invalid
        <rgb_dir>/sky/<frame>.png     uint8 mask, 255 = sky/invalid (for B-tier
                                       photometric down-weighting)

UniK3D.infer wants uint8 0-255; for a full ERP the camera is
Spherical([fx,fy,cx,cy,W,H,HFoV/2=pi,VFoV/2=pi/2]).
"""
import argparse, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--rgb-dir", required=True, help="dir of canonical ERP jpgs")
ap.add_argument("--backbone", default="vitl")
ap.add_argument("--far-invalid-m", type=float, default=300.0,
                help="distance above this -> 0 (sky/unreliable far). Aerial OOD rails "
                     "high (UniK3D med ~100m here), so ~300 keeps more valid; see §5.1.")
ap.add_argument("--sky-blue-thresh", type=float, default=1.10,
                help="top-band B/R ratio above this AND bright -> sky. <=0 disables.")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
a = ap.parse_args()

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "UniSHARP" / "UniK3D"))
from unik3d.models import UniK3D
from unik3d.utils.camera import Spherical

dev = torch.device(a.device)
print(f">> loading UniK3D-{a.backbone} ...", flush=True)
model = UniK3D.from_pretrained(f"lpiccinelli/unik3d-{a.backbone}")
if not hasattr(model, "resolution_level"):
    model.resolution_level = 0
model = model.to(dev).eval()

rgb_dir = Path(a.rgb_dir)
dep_dir = rgb_dir / "depth"; dep_dir.mkdir(parents=True, exist_ok=True)
sky_dir = rgb_dir / "sky";   sky_dir.mkdir(parents=True, exist_ok=True)
jpgs = sorted(rgb_dir.glob("*.jpg"))
if not jpgs:
    sys.exit(f"no jpgs in {rgb_dir}")

def sky_from_color(arr_hwc, frac=0.5):
    """Heuristic blue-sky in top band: bright & blue-dominant."""
    H = arr_hwc.shape[0]
    band = int(H * frac)
    r = arr_hwc[..., 0].astype(np.float32); g = arr_hwc[..., 1].astype(np.float32); b = arr_hwc[..., 2].astype(np.float32)
    bright = (r + g + b) / 3.0 > 140.0
    blue = (b + 1.0) / (r + 1.0) > a.sky_blue_thresh
    m = np.zeros(arr_hwc.shape[:2], bool)
    if a.sky_blue_thresh > 0:
        m[:band] = (bright & blue)[:band]
    return m

n = 0
for jp in jpgs:
    arr = np.asarray(Image.open(jp).convert("RGB"), np.uint8)
    H, W = arr.shape[:2]
    rgb = torch.from_numpy(arr).permute(2, 0, 1).contiguous().to(dev)  # uint8 [3,H,W]
    params = torch.tensor([[1.0, 1.0, W / 2.0, H / 2.0, float(W), float(H),
                            float(np.pi), float(np.pi / 2)]], dtype=torch.float32, device=dev)
    cam = Spherical(params=params)
    with torch.no_grad():
        out = model.infer(rgb, camera=cam, normalize=True)
    dist = out["distance"][0, 0].float().cpu().numpy()  # [H,W] radial metres
    if dist.shape != (H, W):
        dist = np.asarray(Image.fromarray(dist).resize((W, H), Image.NEAREST))

    invalid = ~np.isfinite(dist)
    invalid |= dist <= 0.0
    if a.far_invalid_m > 0:
        invalid |= dist > a.far_invalid_m
    invalid |= sky_from_color(arr)
    dist = dist.astype(np.float32)
    dist[invalid] = 0.0

    np.save(dep_dir / f"{jp.stem}.npy", dist)
    Image.fromarray((invalid.astype(np.uint8) * 255)).save(sky_dir / f"{jp.stem}.png")
    n += 1
    if n % 20 == 0:
        valid = float((dist > 0).mean())
        print(f"  [{n}/{len(jpgs)}] {jp.name}  valid={valid:.2f}  "
              f"med={np.median(dist[dist>0]) if (dist>0).any() else 0:.1f}m", flush=True)

print(f"[pseudo_depth] {n} frames -> {dep_dir}  (far_invalid={a.far_invalid_m}m)")
