#!/usr/bin/env python3
"""Overlay the predicted gaussian radial-distance (||xyz||) histogram of the
pretrained vs fine-tuned UniSHARP on one held-out frame. Shows whether the
fine-tune moved the depth distribution off UniK3D's compressed ~indoor track.
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from plyfile import PlyData

ap = argparse.ArgumentParser()
ap.add_argument("--frame", default="00200")
ap.add_argument("--us", default="/w/p6_unisharp/UniSHARP")
ap.add_argument("--out", default="/w/p6_unisharp/ft/runs")
a = ap.parse_args()

def radial(tag):
    p = Path(a.us) / "outputs" / f"dh_{tag}" / f"inputs_dh_{a.frame}" / "gaussians.ply"
    v = PlyData.read(str(p))["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    return np.linalg.norm(xyz, axis=1)

pre, ft = radial("pre"), radial("ft")
for tag, d in [("pretrained", pre), ("finetuned", ft)]:
    print(f"{tag}: n={len(d)} median={np.median(d):.1f}m p10={np.percentile(d,10):.1f} "
          f"p90={np.percentile(d,90):.1f} frac<50m={np.mean(d<50):.2f}")

plt.figure(figsize=(8, 5))
bins = np.linspace(0, 300, 80)
plt.hist(pre, bins=bins, alpha=0.55, label=f"pretrained (med {np.median(pre):.0f}m)", color="#888")
plt.hist(ft, bins=bins, alpha=0.55, label=f"finetuned step3000 (med {np.median(ft):.0f}m)", color="#d62728")
plt.xlabel("gaussian radial distance ||xyz|| (m)"); plt.ylabel("count")
plt.title(f"023 frame {a.frame}: predicted depth distribution, before vs after fine-tune")
plt.legend(); plt.grid(alpha=0.3)
out = Path(a.out) / f"depth_hist_{a.frame}.png"
plt.tight_layout(); plt.savefig(str(out), dpi=110)
print(f"saved {out}")
