#!/usr/bin/env python3
"""Video monocular depth (Video-Depth-Anything) over a de-rotated ERP frame
sequence, to replace the flat single-frame UniK3D pseudo-depth (§7: that flat
anchor collapsed the fine-tuned depth into a ~100m shell -> LPIPS regression).

VDA gives temporally-consistent RELATIVE depth with real near/far structure.
We:
  - feed the 240 gravity-leveled ERP frames directly (no mp4/decord),
  - convert disparity->depth, orient so sky(top) is far,
  - apply ONE global scale so the whole sequence's median depth = --target-median
    (global, not per-frame, to preserve VDA temporal consistency),
  - mask far/sky (> --far-invalid) and nonpositive -> 0,
  - save per-frame depth .npy matching the rgb resolution (SimPanorama layout).

NOTE (honest caveat): VDA is a perspective model; ERP is 2:1 and pole-distorted,
so this is geometrically approximate at zenith/nadir. It is used as a *structured
anchor*, not ground truth; the photometric loss does the heavy lifting.
"""
import argparse, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--frames-dir", required=True, help="de-rotated ERP jpgs (sorted)")
ap.add_argument("--out-dir", required=True, help="writes <out>/depth/<stem>.npy")
ap.add_argument("--vda", default="/w/p6_unisharp/vda")
ap.add_argument("--encoder", default="vitl")
ap.add_argument("--input-size", type=int, default=518)
ap.add_argument("--target-median", type=float, default=100.0, help="metres; global scale target")
ap.add_argument("--far-invalid", type=float, default=300.0)
ap.add_argument("--out-h", type=int, default=1024)
ap.add_argument("--out-w", type=int, default=2048)
a = ap.parse_args()

sys.path.insert(0, a.vda)
from video_depth_anything.video_depth import VideoDepthAnything

CFG = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}
dev = "cuda" if torch.cuda.is_available() else "cpu"

fdir = Path(a.frames_dir)
jpgs = sorted(fdir.glob("*.jpg"))
assert jpgs, f"no jpgs in {fdir}"
print(f">> loading {len(jpgs)} frames", flush=True)
frames = np.stack([np.asarray(Image.open(p).convert("RGB"), np.uint8) for p in jpgs])  # (N,H,W,3)

print(f">> VDA {a.encoder} infer (input_size={a.input_size}) ...", flush=True)
model = VideoDepthAnything(**CFG[a.encoder], metric=False)
ck = f"{a.vda}/checkpoints/video_depth_anything_{a.encoder}.pth"
model.load_state_dict(torch.load(ck, map_location="cpu"), strict=True)
model = model.to(dev).eval()
out = model.infer_video_depth(frames, target_fps=10, input_size=a.input_size, device=dev, fp32=False)
depths = out[0] if isinstance(out, tuple) else out
depths = np.asarray(depths, dtype=np.float32)          # (N,h,w) VDA disparity-like (near=large)
print(f">> VDA out shape={depths.shape} min={depths.min():.3f} max={depths.max():.3f}", flush=True)

# disparity -> depth (near=large disparity). orient so sky(top band) is FAR.
eps = 1e-6
depth = 1.0 / (depths - depths.min() + eps)            # large disparity -> small depth
H = depth.shape[1]
top = np.median(depth[:, : H // 5]); bot = np.median(depth[:, 4 * H // 5 :])
if top < bot:                                          # sky(top) should be FAR (large depth)
    print(f">> orientation flip (top {top:.2f} < bot {bot:.2f}); using disparity as depth")
    depth = depths - depths.min() + eps
# global scale: whole-sequence median -> target
med = float(np.median(depth))
scale = a.target_median / (med + eps)
depth_m = depth * scale
print(f">> global median {med:.3f} -> scale {scale:.3f}; depth_m med={np.median(depth_m):.1f}m "
      f"p10={np.percentile(depth_m,10):.1f} p90={np.percentile(depth_m,90):.1f}", flush=True)

dep_out = Path(a.out_dir) / "depth"; dep_out.mkdir(parents=True, exist_ok=True)
import torch.nn.functional as F
for i, p in enumerate(jpgs):
    d = torch.from_numpy(depth_m[i])[None, None]
    d = F.interpolate(d, size=(a.out_h, a.out_w), mode="nearest")[0, 0].numpy()
    d[~np.isfinite(d)] = 0.0
    d[d <= 0] = 0.0
    if a.far_invalid > 0:
        d[d > a.far_invalid] = 0.0
    np.save(dep_out / f"{p.stem}.npy", d.astype(np.float32))
print(f"[vda_depth] {len(jpgs)} depth maps -> {dep_out} "
      f"(valid frac ~{np.mean((depth_m>0)&(depth_m<a.far_invalid)):.2f})")
