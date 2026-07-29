#!/usr/bin/env python3
"""Split a scene into blocks along its principal axis (simplified G2PS test).

GGPS's claim is that a single global model underfits a large scene because ERP
gives every camera 360-degree visibility, so frustum-based partitioning fails and
they partition by geometry+gradient instead. This tests the *hypothesis* (does
splitting a long trajectory help?) without reimplementing their algorithm:

  - cameras are projected on the trajectory's principal axis and cut into K parts
  - each block trains on its own cameras plus an overlap margin
  - block_merge.py then keeps, from each block, only the gaussians it owns
    (nearest block centre along the axis), so the merged model has no double
    density in the overlap

usage: block_split.py <cams.json> <out_prefix> [K=2] [overlap=0.15]
"""
import sys, json
import numpy as np

CAMS, OUT = sys.argv[1], sys.argv[2]
K = int(sys.argv[3]) if len(sys.argv) > 3 else 2
OV = float(sys.argv[4]) if len(sys.argv) > 4 else 0.15

meta = json.load(open(CAMS))
cams = meta["cameras"]
C = np.array([c["C"] for c in cams])
ctr = C.mean(0)
# principal axis of the flight path
u, s, vt = np.linalg.svd(C - ctr, full_matrices=False)
axis = vt[0]
t = (C - ctr) @ axis
lo, hi = t.min(), t.max()
# equal-count (quantile) edges, not equal-length: the flight dwells in some areas,
# and blocks should carry comparable supervision, not comparable metres.
edges = np.quantile(t, np.linspace(0, 1, K + 1))
edges[0], edges[-1] = lo, hi
span = (hi - lo) / K

# cam_radius is the full-scene camera radius; the block trainer needs it to tell the
# shared sky shell from near-field geometry (its own camera set is only one block wide).
info = {"axis": axis.tolist(), "ctr": ctr.tolist(), "edges": edges.tolist(),
        "cam_radius": float(np.linalg.norm(C - ctr, axis=1).max())}
json.dump(info, open(f"{OUT}_blocks.json", "w"), indent=1)
print(f"principal axis span {hi-lo:.0f} m -> {K} blocks of {span:.0f} m (overlap {OV:.0%})")

for b in range(K):
    a0, a1 = edges[b] - OV * span, edges[b + 1] + OV * span
    keep = [c for c, ti in zip(cams, t) if a0 <= ti <= a1]
    m = dict(meta); m["cameras"] = keep
    p = f"{OUT}_b{b}.json"
    json.dump(m, open(p, "w"), indent=1)
    core = int(((t >= edges[b]) & (t <= edges[b + 1])).sum())
    print(f"  block {b}: {len(keep):4d} cams ({core} core + {len(keep)-core} overlap) -> {p}")
