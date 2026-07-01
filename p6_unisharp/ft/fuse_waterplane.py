#!/usr/bin/env python3
"""Fuse a geometrically-exact water/ground-plane depth (on CLIPSeg water pixels)
with VDA structured depth (elsewhere), on the gravity-leveled ERP.

Water is a horizontal plane at Y=-h below the camera (up=+Y after leveling). A
pixel at latitude lat has ray y-component sin(lat); a downward ray (lat<0) hits
the plane at radial depth = h / (-sin(lat)) — exact plane geometry (nadir=h
nearest, grazing->inf). h (camera height, unknown w/o GPS) is fit GLOBALLY so the
plane matches VDA's scale on water pixels (h = median(VDA * (-sin lat)) there),
keeping VDA's temporal consistency. Sky -> 0 (invalid).

  fuse_waterplane.py --vda-dir <scene_vda> --mask-dir <clipseg> --out-dir <scene_wp>
"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--vda-dir", required=True, help="scene dir with depth/<stem>.npy (VDA)")
ap.add_argument("--mask-dir", required=True, help="clipseg dir with water/ and sky/ png")
ap.add_argument("--out-dir", required=True)
ap.add_argument("--far-invalid", type=float, default=300.0)
ap.add_argument("--mask-thresh", type=int, default=127)
a = ap.parse_args()

vda_dir = Path(a.vda_dir) / "depth"
water_dir = Path(a.mask_dir) / "water"
sky_dir = Path(a.mask_dir) / "sky"
out_dir = Path(a.out_dir) / "depth"; out_dir.mkdir(parents=True, exist_ok=True)
stems = sorted(p.stem for p in vda_dir.glob("*.npy"))
assert stems, f"no VDA depth in {vda_dir}"

def load_mask(d, stem, shape):
    p = d / f"{stem}.png"
    if not p.exists():
        return np.zeros(shape, bool)
    m = np.asarray(Image.open(p).convert("L"))
    if m.shape != shape:
        m = np.asarray(Image.fromarray(m).resize((shape[1], shape[0]), Image.NEAREST))
    return m > a.mask_thresh

# precompute sin(lat) per row from first frame's shape
d0 = np.load(vda_dir / f"{stems[0]}.npy"); H, W = d0.shape
lat = np.pi / 2 - (np.arange(H) + 0.5) / H * np.pi
neg_sin = (-np.sin(lat))[:, None]                      # (H,1) >0 downward (lat<0)
downward = (neg_sin > 1e-3)                            # below horizon rows

# --- pass 1: global h = median(VDA * (-sin lat)) over downward water pixels ---
samp = []
for stem in stems[::4]:
    vda = np.load(vda_dir / f"{stem}.npy")
    wm = load_mask(water_dir, stem, (H, W)) & downward & (vda > 0)
    if wm.any():
        ns = np.broadcast_to(neg_sin, (H, W))[wm]
        samp.append(vda[wm] * ns)
h = float(np.median(np.concatenate(samp))) if samp else 100.0
print(f"[fuse] global camera-height h = {h:.1f} m (fit VDA vs plane on water)", flush=True)

# --- pass 2: fuse ---
plane_full = h / np.clip(neg_sin, 1e-3, None)          # (H,1) plane depth by row
plane_full = np.broadcast_to(plane_full, (H, W))
n = 0
for stem in stems:
    vda = np.load(vda_dir / f"{stem}.npy").astype(np.float32)
    wm = load_mask(water_dir, stem, (H, W)) & downward
    sky = load_mask(sky_dir, stem, (H, W))
    fused = vda.copy()
    fused[wm] = plane_full[wm]                          # geometrically-exact on water
    fused[sky] = 0.0                                    # sky invalid
    fused[~np.isfinite(fused)] = 0.0
    fused[fused <= 0] = 0.0
    if a.far_invalid > 0:
        fused[fused > a.far_invalid] = 0.0
    np.save(out_dir / f"{stem}.npy", fused.astype(np.float32))
    n += 1
print(f"[fuse] {n} fused depth maps -> {out_dir}")
