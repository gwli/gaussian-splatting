#!/usr/bin/env python3
"""GPS-prior COLMAP refinement bridge for 023 front crops (FOV120, 1280^2).
mode=prior: write COLMAP text model from pano_cams_scene_023hf_gps3.json
            (crop frame == pano frame: yaw0/pitch0), for point_triangulator.
mode=fuse : read BA-refined model (TXT), Sim3-align centers back to GPS prior
            (fixes gauge+scale), write pano_cams_scene_023hf_colmap.json.
"""
import sys, json, math, os
import numpy as np

ROOT = "/w"
J = f"{ROOT}/p3_pano/pano_cams_scene_023hf_gps3.json"
F = 369.504

def mat2quat(R):
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1) * 2; w = s / 4; x = (R[2,1]-R[1,2])/s; y = (R[0,2]-R[2,0])/s; z = (R[1,0]-R[0,1])/s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = math.sqrt(1+R[0,0]-R[1,1]-R[2,2])*2; w=(R[2,1]-R[1,2])/s; x=s/4; y=(R[0,1]+R[1,0])/s; z=(R[0,2]+R[2,0])/s
    elif R[1,1] > R[2,2]:
        s = math.sqrt(1+R[1,1]-R[0,0]-R[2,2])*2; w=(R[0,2]-R[2,0])/s; x=(R[0,1]+R[1,0])/s; y=s/4; z=(R[1,2]+R[2,1])/s
    else:
        s = math.sqrt(1+R[2,2]-R[0,0]-R[1,1])*2; w=(R[1,0]-R[0,1])/s; x=(R[0,2]+R[2,0])/s; y=(R[1,2]+R[2,1])/s; z=s/4
    return w, x, y, z

def quat2mat(w,x,y,z):
    n=math.sqrt(w*w+x*x+y*y+z*z); w,x,y,z=w/n,x/n,y/n,z/n
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

mode, out = sys.argv[1], sys.argv[2]
meta = json.load(open(J))

if mode == "prior":
    os.makedirs(out, exist_ok=True)
    open(f"{out}/cameras.txt","w").write(f"1 PINHOLE 1280 1280 {F} {F} 640 640\n")
    with open(f"{out}/images.txt","w") as f:
        for c in meta["cameras"]:
            R = np.array(c["R_wp"]); T = np.array(c["T"])
            w,x,y,z = mat2quat(R)
            name = f"pano_{c['idx']:04d}_y+000_p+00.jpg"
            f.write(f"{c['idx']} {w} {x} {y} {z} {T[0]} {T[1]} {T[2]} 1 {name}\n\n")
    open(f"{out}/points3D.txt","w").close()
    print(f"[prior] {len(meta['cameras'])} images -> {out}")
else:  # fuse <refined_txt_dir>
    ref = {}
    for ln in open(f"{out}/images.txt"):
        if ln.startswith("#") or not ln.strip(): continue
        p = ln.split()
        if len(p) >= 10 and p[9].endswith(".jpg"):
            q = list(map(float, p[1:5])); t = np.array(list(map(float, p[5:8])))
            R = quat2mat(*q); ref[p[9]] = (R, -R.T @ t)   # (R_wc, C)
    prior = {f"pano_{c['idx']:04d}_y+000_p+00.jpg": np.array(c["C"]) for c in meta["cameras"]}
    names = [n for n in prior if n in ref]
    print(f"[fuse] refined {len(ref)}/{len(prior)} images")
    Cr = np.stack([ref[n][1] for n in names]); Cp = np.stack([prior[n] for n in names])
    ms, md = Cr.mean(0), Cp.mean(0); s0, d0 = Cr-ms, Cp-md
    cov = d0.T @ s0 / len(names); U,D,Vt = np.linalg.svd(cov); Sg = np.eye(3)
    if np.linalg.det(U)*np.linalg.det(Vt) < 0: Sg[2,2] = -1
    Rs = U @ Sg @ Vt; sc = np.trace(np.diag(D) @ Sg) / (s0**2).sum() * len(names)
    ts = md - sc * Rs @ ms
    res = np.linalg.norm((sc*(Rs@Cr.T).T+ts) - Cp, axis=1)
    print(f"[fuse] sim3: scale={sc:.3f} residual med={np.median(res):.2f}m p90={np.percentile(res,90):.2f}m")
    byname = {f"pano_{c['idx']:04d}_y+000_p+00.jpg": c for c in meta["cameras"]}
    cams = []
    for c in meta["cameras"]:
        n = f"pano_{c['idx']:04d}_y+000_p+00.jpg"
        if n in ref:
            Rwc, Cc = ref[n]
            Rn = Rwc @ Rs.T          # world(GPS frame) -> cam
            Cn = sc * Rs @ Cc + ts
        else:
            Rn = np.array(c["R_wp"]); Cn = np.array(c["C"])
        cams.append({**c, "R_wp": Rn.tolist(), "C": Cn.tolist(), "T": (-Rn @ Cn).tolist()})
    outj = dict(meta); outj["cameras"] = cams
    json.dump(outj, open(f"{ROOT}/p3_pano/pano_cams_scene_023hf_colmap.json","w"), indent=1)
    print("[fuse] wrote pano_cams_scene_023hf_colmap.json")
