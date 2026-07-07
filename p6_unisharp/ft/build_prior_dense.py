#!/usr/bin/env python3
"""Build a COLMAP prior model for the 1140-frame dense down stream by
time-interpolating the 240-frame down SfM poses (SLERP R, lerp C) in SfM world.
Feeds point_triangulator + free BA on the completed match database.
usage: build_prior_dense.py <out_model_dir>   (paths hardcoded for 023)
"""
import sys, os, math
import numpy as np

SRC = "/w/data/8kpano/scenes/fish023/sparse/1"   # 240-frame down SfM (TXT)
OUT = sys.argv[1]
N240, ND = 240, 1140
os.makedirs(OUT, exist_ok=True)

def q2R(w, x, y, z):
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

poses = {}
for ln in open(f"{SRC}/images.txt"):
    if ln.startswith("#") or not ln.strip(): continue
    f = ln.split()
    if len(f) >= 10 and f[9].endswith(".jpg"):
        q = np.array(list(map(float, f[1:5]))); q /= np.linalg.norm(q)
        t = np.array(list(map(float, f[5:8])))
        j = int(f[9].split("_")[1].split(".")[0])   # 1-based frame index
        R = q2R(*q)
        poses[j] = (q, -R.T @ t)                    # (quat wxyz, center)
assert len(poses) == N240, f"expected 240 poses, got {len(poses)}"

def slerp(q0, q1, a):
    d = float(np.dot(q0, q1))
    if d < 0: q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + a*(q1-q0); return q/np.linalg.norm(q)
    th = math.acos(np.clip(d, -1, 1))
    return (math.sin((1-a)*th)*q0 + math.sin(a*th)*q1)/math.sin(th)

# image ids MUST match the database (feature-extraction order != name order)
import sqlite3
dbp = os.environ.get("MATCH_DB", "/w/data/8kpano/scenes/fish023d/db_matched.db")
db = sqlite3.connect(f"file:{dbp}?mode=ro", uri=True)
name2id = dict(db.execute("select name, image_id from images").fetchall())
assert len(name2id) == ND

# dense frame k (1..1140) -> 240-space position p = (k-1)*240/1140 (0-based)
ilines, flines = [], []
for k in range(1, ND+1):
    p = (k-1)*N240/ND
    j0 = min(int(p), N240-1); j1 = min(j0+1, N240-1); a = p - j0
    q0, C0 = poses[j0+1]; q1, C1 = poses[j1+1]
    q = slerp(q0, q1, a); C = (1-a)*C0 + a*C1
    R = q2R(*q); t = -R @ C
    iid = name2id[f"f_{k:04d}.jpg"]
    pose = f"{q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]}"
    ilines.append((iid, f"{iid} {pose} 1 f_{k:04d}.jpg\n\n"))
    flines.append((iid, f"{iid} 1 {pose} 1 CAMERA 1 {iid}\n"))

with open(f"{OUT}/images.txt", "w") as f:
    f.writelines(l for _, l in sorted(ilines))
with open(f"{OUT}/frames.txt", "w") as f:
    f.writelines(l for _, l in sorted(flines))
with open(f"{OUT}/rigs.txt", "w") as f:
    f.write("1 1 CAMERA 1\n")
with open(f"{OUT}/cameras.txt", "w") as f:
    # dense frames are native 3840x3840: fx,fy,cx,cy scale x2 vs the 1920 calib; k1..k4 are scale-invariant
    f.write("1 OPENCV_FISHEYE 3840 3840 1096.18568235725116 1096.29168563253802 1920 1920 "
            "0.030349368620543618 0.0023128116945658281 -0.0027963710018365905 -0.00035606873525276218\n")
open(f"{OUT}/points3D.txt", "w").close()
print(f"prior model: {ND} interpolated poses -> {OUT}")
