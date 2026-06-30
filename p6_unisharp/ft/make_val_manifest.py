#!/usr/bin/env python3
"""Build a run_validation sim manifest (lines: 'scene|src|tgt1,tgt2,...') from a
val pose CSV, pairing by frame gap + metric translation range. Overwrites the
manifest the eval wrapper reads.

  python make_val_manifest.py --scene scene_023hf_val
"""
import argparse, csv, math
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--scene", required=True)
ap.add_argument("--ft", default="/raid/git/gaussian-splatting/p6_unisharp/ft")
ap.add_argument("--min-gap", type=int, default=1)
ap.add_argument("--max-gap", type=int, default=16)
ap.add_argument("--min-tr", type=float, default=0.3)
ap.add_argument("--max-tr", type=float, default=8.0)
ap.add_argument("--max-tgt", type=int, default=4)
a = ap.parse_args()

FT = Path(a.ft)
rows = list(csv.DictReader(open(FT / "poses" / f"{a.scene}.csv")))
pos = {int(r["frame"]): (float(r["x"]), float(r["y"]), float(r["z"])) for r in rows}
frames = sorted(pos)
d = lambda u, v: math.sqrt(sum((u[i] - v[i]) ** 2 for i in range(3)))

lines, npairs = [], 0
for s in frames:
    cands = []
    for t in frames:
        g = abs(t - s)
        if t == s or g < a.min_gap or g > a.max_gap:
            continue
        tr = d(pos[s], pos[t])
        if a.min_tr <= tr <= a.max_tr:
            cands.append((tr, t))
    cands.sort(reverse=True)                       # prefer larger baseline
    tgts = [t for _, t in cands[: a.max_tgt]]
    if tgts:
        lines.append(f"{a.scene}|{s}|{','.join(map(str, tgts))}")
        npairs += len(tgts)

out = FT / "manifests" / f"{a.scene}.txt"
out.write_text("\n".join(lines) + "\n")
print(f"[val_manifest] {a.scene}: {len(lines)} src groups, {npairs} pairs -> {out}")
