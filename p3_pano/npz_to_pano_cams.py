#!/usr/bin/env python3
"""Bridge sliding-window VGGT output (vggt_window_merged.npz: names/R/t/pts) to
pano_cams.json + points.ply for direct-pano training — skips COLMAP binaries.
Each pano has ONE front crop (yaw0,pitch0 -> R_off=I), so R_wp=R, T=t,
C=-R^T t. Usage: npz_to_pano_cams.py <scene_pano_dir> <out_json>"""
import sys, os, re, json, numpy as np
from plyfile import PlyData, PlyElement

scene, out_json = sys.argv[1], sys.argv[2]
d = np.load(os.path.join(scene, "vggt_window_merged.npz"), allow_pickle=True)
names, R, t, pts = d["names"], d["R"].astype(np.float64), d["t"].astype(np.float64), d["pts"].astype(np.float32)
name_re = re.compile(r"pano_(\d+)_")

cams = []
for i, n in enumerate(names):
    m = name_re.search(str(n))
    if not m:
        continue
    idx = int(m.group(1))
    Rwp = R[i]                                   # R_off=I (front crop)
    C = (-Rwp.T @ t[i])
    T = t[i]                                      # = -Rwp @ C
    img = os.path.join(scene, "panoramas", f"pano_{idx:04d}.jpg")
    cams.append({"idx": idx, "image": img, "R_wp": Rwp.tolist(),
                 "T": T.tolist(), "C": C.tolist(), "n_crops": 1,
                 "ref_yaw": 0.0, "ref_pitch": 0.0})
cams.sort(key=lambda c: c["idx"])
centers = np.array([c["C"] for c in cams])
extent = float(np.linalg.norm(centers - centers.mean(0), axis=1).max() * 1.1)

# write init point cloud (x,y,z; trainer fills gray if no color)
os.makedirs(os.path.join(scene, "sparse/0"), exist_ok=True)
ply_path = os.path.join(scene, "sparse/0/points.ply")
verts = np.array([tuple(p) for p in pts], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
PlyData([PlyElement.describe(verts, "vertex")]).write(ply_path)

json.dump({"scene_dir": scene, "point_cloud": ply_path,
           "cameras_extent": extent, "cameras": cams},
          open(out_json, "w"), indent=1)
print(f"OK {len(cams)} pano cameras | extent={extent:.3f} | {len(pts)} init pts | "
      f"centers span {centers.ptp(0).round(2).tolist()}")
