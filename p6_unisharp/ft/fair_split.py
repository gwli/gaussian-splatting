#!/usr/bin/env python3
"""Build a LEAK-FREE density ablation from a 1140-frame pano_cams json.

Problem this fixes: the default split (test = cams[::8]) samples test views from
the same continuous trajectory, so the gap between a test view and its nearest
training view shrinks as density grows (1.59s @240 -> 0.33s @1140, i.e. ~7.3m ->
~1.5m of drone travel). Denser runs then score higher partly because their test
task got easier, not only because the reconstruction improved.

Fix: one common held-out test set for every density, plus a guard band so the
nearest training view is >= GUARD frames away in ALL variants. Density is then
the only variable.

usage: fair_split.py <cams_1140.json> <out_prefix> [n_test=24] [guard=7]
writes <out_prefix>_test_idx.txt and <out_prefix>_d{1,2,4}.json
"""
import sys, json, os
import numpy as np

CAMS, OUT = sys.argv[1], sys.argv[2]
N_TEST = int(sys.argv[3]) if len(sys.argv) > 3 else 24
GUARD = int(sys.argv[4]) if len(sys.argv) > 4 else 7

meta = json.load(open(CAMS))
cams = meta["cameras"]
idxs = sorted(c["idx"] for c in cams)
lo, hi = idxs[0], idxs[-1]

# evenly spaced test frames, kept away from the sequence ends
step = (hi - lo) / (N_TEST + 1)
test = sorted({int(round(lo + step * (k + 1))) for k in range(N_TEST)} & set(idxs))
banned = {j for t in test for j in range(t - GUARD, t + GUARD + 1)}
pool = [j for j in idxs if j not in banned]

open(f"{OUT}_test_idx.txt", "w").write("\n".join(map(str, test)) + "\n")
by_idx = {c["idx"]: c for c in cams}
print(f"test={len(test)} guard=+-{GUARD}  pool={len(pool)}/{len(idxs)}")

for stride in (1, 2, 4):
    keep = set(pool[::stride]) | set(test)          # test cams must stay in the json
    m = dict(meta)
    m["cameras"] = [by_idx[j] for j in sorted(keep)]
    p = f"{OUT}_d{stride}.json"
    json.dump(m, open(p, "w"), indent=1)
    # nearest-train-neighbour gap actually realised by this variant (in frames)
    tr = np.array(sorted(set(pool[::stride])))
    gaps = [int(np.min(np.abs(tr - t))) for t in test]
    print(f"stride{stride}: train={len(tr)} (+{len(test)} test) "
          f"| nearest train view: min={min(gaps)} med={int(np.median(gaps))} frames -> {p}")
