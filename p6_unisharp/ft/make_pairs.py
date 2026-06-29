#!/usr/bin/env python3
"""A-tier step 3: build (source, target) training pairs from the pose CSV.

Mirrors SimPanorama/re10k pair selection but on our metric translations: keep a
target if frame gap in [min_gap,max_gap] AND ||C_tgt - C_src|| in [min_tr,max_tr]
metres. Translation-only is honoured because RGB was de-rotated to a common frame.

Input : <pose_csv>  (frame,x,y,z, metres)
Output: <out_jsonl> lines: {"scene","src","tgt","trans_m","gap"}
"""
import argparse, csv, json, math
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--pose", required=True)
ap.add_argument("--scene", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--min-gap", type=int, default=1)
ap.add_argument("--max-gap", type=int, default=8)
ap.add_argument("--min-tr", type=float, default=0.3, help="min baseline (m) for usable parallax")
ap.add_argument("--max-tr", type=float, default=8.0, help="max baseline (m) before overlap dies")
ap.add_argument("--max-tgt-per-src", type=int, default=4)
a = ap.parse_args()

rows = []
with open(a.pose) as f:
    for r in csv.DictReader(f):
        rows.append((int(r["frame"]), float(r["x"]), float(r["y"]), float(r["z"])))
rows.sort()
pos = {fr: (x, y, z) for fr, x, y, z in rows}
frames = [fr for fr, *_ in rows]

def dist(a_, b_):
    return math.sqrt(sum((a_[i] - b_[i]) ** 2 for i in range(3)))

Path(a.out).parent.mkdir(parents=True, exist_ok=True)
npair = 0
nsrc_with = 0
with open(a.out, "w") as out:
    for i, s in enumerate(frames):
        cnt = 0
        cands = []
        for t in frames:
            g = abs(t - s)
            if t == s or g < a.min_gap or g > a.max_gap:
                continue
            tr = dist(pos[s], pos[t])
            if a.min_tr <= tr <= a.max_tr:
                cands.append((tr, t))
        # prefer larger baseline (stronger supervision) up to the cap
        cands.sort(reverse=True)
        for tr, t in cands[: a.max_tgt_per_src]:
            out.write(json.dumps({"scene": a.scene, "src": s, "tgt": t,
                                  "trans_m": round(tr, 3), "gap": abs(t - s)}) + "\n")
            npair += 1; cnt += 1
        if cnt:
            nsrc_with += 1

print(f"[make_pairs] scene={a.scene}: {npair} pairs from {len(frames)} frames "
      f"({nsrc_with} srcs with >=1 tgt) -> {a.out}")
if nsrc_with < len(frames) * 0.5:
    print("  [warn] <50% of frames got a pair — adjust --min-tr/--max-tr or --pos-scale "
          "(translations may be mis-scaled; see task_ft.md §3.2)")
