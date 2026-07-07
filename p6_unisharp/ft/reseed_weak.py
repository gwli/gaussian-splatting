#!/usr/bin/env python3
"""Re-seed weakly-constrained frames (few 3D obs) in the dense BA model by
time-interpolating from well-constrained neighbors, keeping good poses as-is.
Writes a new prior model for another tri+BA round.
usage: reseed_weak.py <ba_model_txt> <out_model_dir> [min_obs=200]
"""
import sys, os, math, shutil
import numpy as np

SRC, OUT = sys.argv[1], sys.argv[2]
MIN_OBS = int(sys.argv[3]) if len(sys.argv) > 3 else 200

def q2R(w, x, y, z):
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

def slerp(q0, q1, a):
    d = float(np.dot(q0, q1))
    if d < 0: q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + a*(q1-q0); return q/np.linalg.norm(q)
    th = math.acos(np.clip(d, -1, 1))
    return (math.sin((1-a)*th)*q0 + math.sin(a*th)*q1)/math.sin(th)

# parse ba model: pose + obs count + image id per name
P, O, IID = {}, {}, {}
cur = None
for ln in open(f"{SRC}/images.txt"):
    if ln.startswith("#") or not ln.strip(): continue
    f = ln.split()
    if len(f) >= 10 and f[9].endswith(".jpg"):
        q = np.array(list(map(float, f[1:5]))); q /= np.linalg.norm(q)
        t = np.array(list(map(float, f[5:8])))
        cur = f[9]; P[cur] = (q, -q2R(*q).T @ t); IID[cur] = f[0]; O[cur] = 0
    else:
        O[cur] = sum(1 for i in range(2, len(f), 3) if f[i] != "-1")

ks = sorted(int(n.split("_")[1].split(".")[0]) for n in P)
name = lambda k: f"f_{k:04d}.jpg"
good = [k for k in ks if O[name(k)] >= MIN_OBS]
bad = [k for k in ks if O[name(k)] < MIN_OBS]
print(f"good={len(good)} bad(<{MIN_OBS} obs)={len(bad)}")

ilines, flines = [], []
ga = np.array(good)
for k in ks:
    n = name(k)
    if O[n] >= MIN_OBS:
        q, C = P[n]
    else:
        i = np.searchsorted(ga, k)
        k0 = ga[max(0, i-1)]; k1 = ga[min(len(ga)-1, i)]
        if k0 == k1: q, C = P[name(k0)]
        else:
            a = (k - k0) / (k1 - k0)
            q0, C0 = P[name(k0)]; q1, C1 = P[name(k1)]
            q = slerp(q0, q1, a); C = (1-a)*C0 + a*C1
    R = q2R(*q); t = -R @ C
    iid = IID[n]
    pose = f"{q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]}"
    ilines.append((int(iid), f"{iid} {pose} 1 {n}\n\n"))
    flines.append((int(iid), f"{iid} 1 {pose} 1 CAMERA 1 {iid}\n"))

os.makedirs(OUT, exist_ok=True)
with open(f"{OUT}/images.txt", "w") as f: f.writelines(l for _, l in sorted(ilines))
with open(f"{OUT}/frames.txt", "w") as f: f.writelines(l for _, l in sorted(flines))
shutil.copy(f"{SRC}/cameras.txt", f"{OUT}/cameras.txt")
shutil.copy(f"{SRC}/rigs.txt", f"{OUT}/rigs.txt")
open(f"{OUT}/points3D.txt", "w").close()
print(f"re-seeded model -> {OUT}")
