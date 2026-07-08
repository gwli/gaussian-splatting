#!/usr/bin/env python3
"""Dense (1140-frame) rig-chain poses for 023: down-SfM (TXT model, argv1) ->
Sim3 to GPS ENU -> pano_cams json in the kernel y-down convention. Keeps the
photometric R_rig in rig023.npz untouched (stitching is separate).
usage: rig_solve_dense.py <down_sparse_txt_dir> <n_frames> <out_json> <pano_dir_rel> [pano_stride=1]
(pano index for frame k = pano_stride*(k-1)+1, for subsampled SfM over dense panos)
"""
import sys, os, math, json, struct
import numpy as np

ROOT = "/w"
DOWN_M, N = sys.argv[1], int(sys.argv[2])
OUT_JSON, PANO_REL = sys.argv[3], sys.argv[4]
PSTRIDE = int(sys.argv[5]) if len(sys.argv) > 5 else 1

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
    return cams

def umeyama(src, dst):
    ms, md = src.mean(0), dst.mean(0); s0, d0 = src-ms, dst-md
    cov = d0.T @ s0 / len(src); U, D, Vt = np.linalg.svd(cov); Sg = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0: Sg[2, 2] = -1
    R = U @ Sg @ Vt; sc = np.trace(np.diag(D) @ Sg) / (s0**2).sum() * len(src)
    return sc, R, md - sc * R @ ms

dn = read_model(DOWN_M)
print(f"down SfM registered: {len(dn)}/{N}")

# GPS ENU track (same trailer offsets as rig_solve.py, 023-specific)
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
ft = np.arange(N)/(N/380.55)
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
    k = PSTRIDE * (idx_of(n) - 1) + 1
    cams.append({"idx": k, "image": f"{PANO_REL}/pano_{k:04d}.jpg",
                 "R_wp": Rwp.tolist(), "T": (-Rwp@C).tolist(), "C": C.tolist(),
                 "n_crops": 1, "ref_yaw": 0.0, "ref_pitch": 0.0})
ce = np.array([c["C"] for c in cams])
out = {"scene_dir": os.path.dirname(PANO_REL),
       "point_cloud": "/w/p3_pano/gps_init_023v2.ply",
       "cameras_extent": float(np.linalg.norm(ce-ce.mean(0), axis=1).max()*1.1), "cameras": cams}
json.dump(out, open(OUT_JSON, "w"), indent=1)
print(f"wrote {OUT_JSON} n={len(cams)}")
