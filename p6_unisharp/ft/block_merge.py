#!/usr/bin/env python3
"""Merge block models by ownership along the split axis (see block_split.py).

Each gaussian is kept only by the block whose slab contains it, so the overlap
region is not represented twice (double density there would both waste budget and
create visible seams).

usage: block_merge.py <blocks.json> <out.ply> <block0.ply> <block1.ply> ...
"""
import sys, json
import numpy as np
from plyfile import PlyData, PlyElement

INFO, OUT = sys.argv[1], sys.argv[2]
PLYS = sys.argv[3:]
info = json.load(open(INFO))
axis = np.array(info["axis"]); ctr = np.array(info["ctr"]); edges = np.array(info["edges"])
assert len(PLYS) == len(edges) - 1, f"{len(PLYS)} plys vs {len(edges)-1} blocks"

keep_arrays, total = [], 0
for b, p in enumerate(PLYS):
    v = PlyData.read(p)["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], 1)
    t = (xyz - ctr) @ axis
    lo = -np.inf if b == 0 else edges[b]                  # first/last block keep the tails,
    hi = np.inf if b == len(PLYS) - 1 else edges[b + 1]   # including the sky shell beyond the ends
    m = (t >= lo) & (t < hi)
    keep_arrays.append(v.data[m]); total += len(v.data)
    print(f"block {b}: {m.sum():7d} / {len(v.data):7d} gaussians owned")

data = np.concatenate(keep_arrays)
PlyData([PlyElement.describe(data, "vertex")]).write(OUT)
print(f"merged {len(data)} gaussians (from {total} across blocks) -> {OUT}")
