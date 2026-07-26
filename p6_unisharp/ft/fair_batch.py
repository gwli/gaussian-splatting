#!/usr/bin/env python3
"""Prepare leak-free 570-density splits for one scene: writes
  fair<S>_test_idx.txt, fair<S>_d2.json (rig), fair<S>_v360_d2.json (baseline)
Both jsons share the same held-out frames AND the same training frame set, so the
only difference is stitch+poses. Container paths (/w/...) are rewritten to
repo-relative so the host venv can read them.
usage: fair_batch.py <scene>   e.g. 021
"""
import sys, json, subprocess, os
import numpy as np

S = sys.argv[1]
RIG = f"p3_pano/pano_cams_scene_{S}rig1140.json"
BASE = f"p3_pano/pano_cams_scene_{S}d_gps3.json"
OUT = f"p3_pano/fair{S}"

subprocess.run([sys.executable, "p6_unisharp/ft/fair_split.py", RIG, OUT, "24", "7"], check=True)

hold = {int(l) for l in open(f"{OUT}_test_idx.txt") if l.strip()}
d2 = json.load(open(f"{OUT}_d2.json"))
d2["point_cloud"] = d2["point_cloud"].replace("/w/", "")
json.dump(d2, open(f"{OUT}_d2.json", "w"), indent=1)
want = {c["idx"] for c in d2["cameras"]}                 # train subset + held-out

base = json.load(open(BASE))
base["point_cloud"] = base["point_cloud"].replace("/w/", "")
have = {c["idx"] for c in base["cameras"]}
base["cameras"] = [c for c in base["cameras"] if c["idx"] in want]
json.dump(base, open(f"{OUT}_v360_d2.json", "w"), indent=1)

miss = want - have
def gap(js):
    cams = json.load(open(js))["cameras"]
    te = np.array([c["C"] for c in cams if c["idx"] in hold])
    tr = np.array([c["C"] for c in cams if c["idx"] not in hold])
    return float(np.median(np.linalg.norm(te[:, None] - tr[None], axis=2).min(1)))
print(f"[{S}] rig cams={len(d2['cameras'])} v360 cams={len(base['cameras'])} "
      f"(missing in baseline: {len(miss)}) | test->nearest-train med: "
      f"rig={gap(f'{OUT}_d2.json'):.2f} m  v360={gap(f'{OUT}_v360_d2.json'):.2f} m")
