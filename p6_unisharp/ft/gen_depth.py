#!/usr/bin/env python3
"""UniK3D depth priors for ERP panoramas (the one GGPS component still untested).

GGPS enables depth only in the block/refine stage (`use_depth: false` in the coarse
config, 1.0 -> 0.001 in c4). UniK3D takes an equirect image directly through its
Spherical camera, so no cube-face detour is needed.

The maps are metric but we do not trust their absolute scale on aerial panoramas,
so the trainer aligns them per image (see DEPTH_* in train_pano_gsplat_sph.py).

usage: gen_depth.py <cams.json> <out_dir> [W=1024] [limit]
"""
import sys, os, json
import numpy as np, torch
from PIL import Image

CAMS, OUT = sys.argv[1], sys.argv[2]
W = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
LIMIT = int(sys.argv[4]) if len(sys.argv) > 4 else 0
H = W // 2
os.makedirs(OUT, exist_ok=True)

from unik3d.models import UniK3D
from unik3d.utils.camera import Spherical

dev = "cuda"
model = UniK3D.from_pretrained("lpiccinelli/unik3d-vitl").to(dev).eval()

meta = json.load(open(CAMS))
cams = meta["cameras"][:LIMIT] if LIMIT else meta["cameras"]
print(f"[depth] {len(cams)} panoramas -> {OUT} at {W}x{H}", flush=True)

for i, c in enumerate(cams):
    dst = os.path.join(OUT, f"{c['idx']:05d}.npy")
    if os.path.exists(dst):
        continue
    im = Image.open(c["image"]).convert("RGB").resize((W, H), Image.LANCZOS)
    rgb = torch.from_numpy(np.asarray(im)).permute(2, 0, 1).to(dev)
    # [fx, fy, cx, cy, W, H, HFoV/2, VFoV/2]; fx..cy are dummies for Spherical
    cam = Spherical(params=torch.tensor([0.0, 0.0, W / 2, H / 2, W, H,
                                         np.pi, np.pi / 2], dtype=torch.float32))
    with torch.no_grad():
        out = model.infer(rgb, camera=cam, normalize=True)
    # "distance" is the radial range, which is what an equirect ray carries; "depth"
    # is the z-component and goes negative behind the camera, meaningless for 360.
    d = out["distance"].squeeze().float().cpu().numpy()   # (H,W) metres, radial
    cf = out["confidence"].squeeze().float().cpu().numpy()
    np.save(dst, np.stack([d, cf]).astype(np.float16))    # (2,H,W)
    if i % 25 == 0:
        print(f"  [{i+1}/{len(cams)}] {c['idx']:05d}  "
              f"dist {np.percentile(d,5):.1f}..{np.percentile(d,95):.1f} m", flush=True)
print("[depth] done", flush=True)
