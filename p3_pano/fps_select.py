#!/usr/bin/env python3
"""FPS frame selection (T-G): given a 1-crop-per-pano VGGT model of a dense pool,
farthest-point-sample N panos by camera center (max viewpoint spread). Prints the
selected pano base names (e.g. pano_0007), one per line.

Usage: fps_select.py <pool_scene_dir> <N>     (reads <dir>/sparse/0/images.bin)
"""
import sys, os, re, importlib.util, numpy as np

scene, N = sys.argv[1], int(sys.argv[2])
_cands = ["/w", "/workspace/gaussian-splatting", "/raid/git/gaussian-splatting"]
_root = next((r for r in _cands if os.path.exists(f"{r}/scene/colmap_loader.py")), _cands[0])
spec = importlib.util.spec_from_file_location("cl", f"{_root}/scene/colmap_loader.py")
cl = importlib.util.module_from_spec(spec); spec.loader.exec_module(cl)

ext = cl.read_extrinsics_binary(f"{scene}/sparse/0/images.bin")
names, cents = [], []
for img in ext.values():
    R = cl.qvec2rotmat(img.qvec); t = img.tvec
    C = -R.T @ t                                   # camera center in world
    m = re.match(r"(pano_\d+)", img.name)
    if m:
        names.append(m.group(1)); cents.append(C)
C = np.asarray(cents, float)
n = len(C)
if N >= n:
    for nm in names: print(nm)
    sys.exit(0)

# farthest-point sampling on camera centers
chosen = [0]
d = np.linalg.norm(C - C[0], axis=1)
while len(chosen) < N:
    j = int(d.argmax()); chosen.append(j)
    d = np.minimum(d, np.linalg.norm(C - C[j], axis=1))
for i in sorted(chosen):
    print(names[i])
