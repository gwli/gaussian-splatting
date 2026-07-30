#!/usr/bin/env python3
"""Carry training-time pose corrections onto the held-out cameras.

In-training pose optimisation only receives gradients for training frames, so the map
ends up expressed in the corrected training frame while held-out cameras still sit at
their original poses. Evaluating in that state measures the mismatch, not the model.

The trajectory is a continuous flight, so a held-out frame's correction is interpolated
linearly (in dataset-frame index) from its nearest corrected neighbours on each side.
Rotations are small (<1 deg), so interpolating the so3 vector directly is accurate to
second order.

Because the guard band means the nearest trained neighbours are ~7 frames away on each
side, --selfcheck quantifies the extrapolation error directly: it hides each trained
frame in turn, interpolates its delta from neighbours at the same distance, and reports
the disagreement with the delta that frame actually converged to. That number is the
error this step introduces; it is not assumed to be small.

usage: apply_pose_deltas.py <poses.npz> <cams.json> <out_cams.json> [--selfcheck]
"""
import sys, json
import numpy as np

NPZ, CAMS, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
SELFCHECK = "--selfcheck" in sys.argv

z = np.load(NPZ)
idx, trained, dr, dt = z["idx"], z["trained"], z["dr"], z["dt"]
order = np.argsort(idx)
idx, trained, dr, dt = idx[order], trained[order], dr[order], dt[order]


def interp(at, known_i, known_v):
    """linear interpolation of a (...,3) delta at frame index `at`, from known frames"""
    j = np.searchsorted(known_i, at)
    if j == 0: return known_v[0]
    if j >= len(known_i): return known_v[-1]
    i0, i1 = known_i[j - 1], known_i[j]
    w = (at - i0) / max(i1 - i0, 1)
    return (1 - w) * known_v[j - 1] + w * known_v[j]


ti = idx[trained]
if SELFCHECK:
    # hide each trained frame and rebuild it from neighbours at guard-band distance
    er, et = [], []
    for k in range(1, len(ti) - 1):
        keep = np.ones(len(ti), bool); keep[k] = False
        # emulate the guard band: also hide the 7 frames on either side
        lo, hi = max(0, k - 7), min(len(ti), k + 8)
        keep[lo:hi] = False
        if keep.sum() < 2: continue
        pr = interp(ti[k], ti[keep], dr[trained][keep])
        pt = interp(ti[k], ti[keep], dt[trained][keep])
        er.append(np.linalg.norm(pr - dr[trained][k]))
        et.append(np.linalg.norm(pt - dt[trained][k]))
    er, et = np.array(er), np.array(et)
    print(f"[selfcheck] {len(er)} frames, guard band +-7")
    print(f"  rotation    error  median {np.degrees(np.median(er)):.4f} deg  "
          f"p90 {np.degrees(np.percentile(er,90)):.4f} deg")
    print(f"  translation error  median {np.median(et):.4f} m  "
          f"p90 {np.percentile(et,90):.4f} m")
    # for comparison: how large are the corrections themselves?
    mr, mt = np.linalg.norm(dr[trained], axis=1), np.linalg.norm(dt[trained], axis=1)
    print(f"  vs correction size median {np.degrees(np.median(mr)):.4f} deg / "
          f"{np.median(mt):.4f} m  -> interpolation keeps "
          f"{100*(1-np.median(et)/max(np.median(mt),1e-9)):.0f}% of the translation fix")

meta = json.load(open(CAMS))
n = 0
for c in meta["cameras"]:
    k = np.searchsorted(idx, c["idx"])
    if k < len(idx) and idx[k] == c["idx"] and trained[k]:
        pr, pt = dr[k], dt[k]
    else:
        pr = interp(c["idx"], ti, dr[trained]); pt = interp(c["idx"], ti, dt[trained]); n += 1
    w = np.asarray(pr, np.float64); th = np.linalg.norm(w)
    if th > 1e-12:                      # so3 exp
        K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]]) / th
        R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
    else:
        R = np.eye(3)
    Rw = R @ np.array(c["R_wp"], np.float64)
    C = np.array(c["C"], np.float64) + np.asarray(pt, np.float64)
    c["R_wp"] = Rw.tolist(); c["C"] = C.tolist(); c["T"] = (-Rw @ C).tolist()
json.dump(meta, open(OUT, "w"))
print(f"[apply] {len(meta['cameras'])} cameras written ({n} interpolated) -> {OUT}")
