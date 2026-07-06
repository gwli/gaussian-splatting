#!/usr/bin/env python3
"""Rig calibration + SfM-metric poses for 023 dual fisheye.
1) Load down SfM (sparse/1 TXT) and up SfM (<argv1> TXT).
2) Sim3-align down-world -> up-world via matched camera centers; per-frame
   R_rig_i = R_u_i @ R_w @ R_d_i^T (should be constant for a rigid mount);
   average via SVD, report angular spread. Save R_rig + up calib.
3) Sim3-align down-SfM centers -> GPS ENU (metric gauge); write
   pano_cams_scene_023rig.json with ERP frame := down-cam frame via fixed
   R_e2c = [[1,0,0],[0,0,1],[0,-1,0]]  (ERP up = cam -z, COLMAP convention).
"""
import sys, os, math, json, struct
import numpy as np

ROOT = "/w"
DOWN_M = f"{ROOT}/data/8kpano/scenes/fish023/sparse/1"
UP_M = sys.argv[1]

def read_model(p):
    cams = {}
    for ln in open(f"{p}/images.txt"):
        if ln.startswith("#") or not ln.strip(): continue
        f = ln.split()
        if len(f) >= 10 and f[9].endswith(".jpg"):
            w, x, y, z = map(float, f[1:5]); t = np.array(list(map(float, f[5:8])))
            n = math.sqrt(w*w+x*x+y*y+z*z); w, x, y, z = w/n, x/n, y/n, z/n
            R = np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                          [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                          [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
            cams[f[9]] = (R, -R.T @ t)
    cal = open(f"{p}/cameras.txt").read().split("\n")
    cal = [l for l in cal if l and not l.startswith("#")][0].split()
    return cams, list(map(float, cal[4:]))  # fx fy cx cy k1..k4

def umeyama(src, dst):
    ms, md = src.mean(0), dst.mean(0); s0, d0 = src-ms, dst-md
    cov = d0.T @ s0 / len(src); U, D, Vt = np.linalg.svd(cov); Sg = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0: Sg[2, 2] = -1
    R = U @ Sg @ Vt; sc = np.trace(np.diag(D) @ Sg) / (s0**2).sum() * len(src)
    return sc, R, md - sc * R @ ms

dn, cal_d = read_model(DOWN_M)
up, cal_u = read_model(UP_M)
names = sorted(set(dn) & set(up))
print(f"common frames: {len(names)} | cal_down={np.round(cal_d,3)} | cal_up={np.round(cal_u,3)}")
Cd = np.stack([dn[n][1] for n in names]); Cu = np.stack([up[n][1] for n in names])
s_w, R_w, t_w = umeyama(Cd, Cu)
res = np.linalg.norm((s_w*(R_w@Cd.T).T+t_w)-Cu, axis=1)
print(f"down->up world sim3: residual med={np.median(res):.3f} (up-world units)")
# per-frame rig rotation
Ms = [up[n][0] @ R_w @ dn[n][0].T for n in names]
Rm = sum(Ms); U, _, Vt = np.linalg.svd(Rm)
if np.linalg.det(U @ Vt) < 0: U[:, -1] *= -1
R_rig = U @ Vt
ang = [math.degrees(math.acos(np.clip((np.trace(R_rig.T@M)-1)/2, -1, 1))) for M in Ms]
print(f"R_rig spread: med={np.median(ang):.2f}deg p90={np.percentile(ang,90):.2f}deg  (rigid mount => ~<1deg)")
np.savez(f"{ROOT}/p3_pano/rig023.npz", R_rig=R_rig, cal_d=cal_d, cal_u=cal_u)

# ---- metric poses from down SfM + GPS ----
F = f"{ROOT}/data/8kpano/VID_20260326_073432_023.insv"; S = os.path.getsize(F)
xs = S - 15672335; fh = open(F, "rb"); fh.seek(xs+7624073); d = fh.read(12614); G = []
for i in range(0, len(d)-52, 53):
    ts = struct.unpack_from("<Q", d, i)[0] + struct.unpack_from("<H", d, i+8)[0]/1000
    lat = struct.unpack_from("<d", d, i+11)[0]; lon = struct.unpack_from("<d", d, i+20)[0]
    alt = struct.unpack_from("<d", d, i+45)[0]; G.append((ts, lat, lon, alt))
G.sort(); t0, lat0, lon0, alt0 = G[0]
mlat = 111320.0; mlon = 111320.0*math.cos(math.radians(lat0))
grt = np.array([g[0]-t0 for g in G])
gE = np.array([(g[2]-lon0)*mlon for g in G]); gN = np.array([(g[1]-lat0)*mlat for g in G]); gU = np.array([g[3]-alt0 for g in G])
N = 240; ft = np.arange(N)/(N/380.55)
ENU = np.stack([np.interp(ft, grt, gE), np.interp(ft, grt, gU), np.interp(ft, grt, gN)], 1)  # (E,U,N) y-up
idx_of = lambda n: int(n.split("_")[1].split(".")[0])  # f_0001.jpg -> 1
names_d = sorted(dn)
Cd_all = np.stack([dn[n][1] for n in names_d])
T_gps = np.stack([ENU[idx_of(n)-1] for n in names_d])
s_a, R_a, t_a = umeyama(Cd_all, T_gps)
res = np.linalg.norm((s_a*(R_a@Cd_all.T).T+t_a)-T_gps, axis=1)
print(f"downSfM->GPS sim3: scale={s_a:.3f} residual med={np.median(res):.2f}m p90={np.percentile(res,90):.2f}m")
R_e2c = np.array([[1,0,0],[0,0,-1],[0,1,0]], float)  # R_k2c, kernel y-down frame
cams = []
for n in names_d:
    R_i, C_i = dn[n]
    Rwp = R_e2c.T @ R_i @ R_a.T
    C = s_a * R_a @ C_i + t_a
    k = idx_of(n)
    cams.append({"idx": k, "image": f"data/8kpano/scenes/scene_023rig_pano/panoramas/pano_{k:04d}.jpg",
                 "R_wp": Rwp.tolist(), "T": (-Rwp@C).tolist(), "C": C.tolist(),
                 "n_crops": 1, "ref_yaw": 0.0, "ref_pitch": 0.0})
ce = np.array([c["C"] for c in cams])
out = {"scene_dir": "data/8kpano/scenes/scene_023rig_pano",
       "point_cloud": "/w/p3_pano/gps_init_023v2.ply",
       "cameras_extent": float(np.linalg.norm(ce-ce.mean(0), axis=1).max()*1.1), "cameras": cams}
json.dump(out, open(f"{ROOT}/p3_pano/pano_cams_scene_023rig.json", "w"), indent=1)
print(f"wrote pano_cams_scene_023rig.json n={len(cams)}")
