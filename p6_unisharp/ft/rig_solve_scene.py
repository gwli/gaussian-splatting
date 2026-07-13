#!/usr/bin/env python3
"""Generic rig-chain pose builder for any A1 scene: down-SfM (TXT) -> Sim3 to
GPS ENU (auto trailer-table parse) -> pano_cams json in kernel y-down convention.
usage: rig_solve_scene.py --insv F --dur SEC --model DIR --n N --out JSON --pano-rel REL [--pano-stride 1]
"""
import argparse, os, math, json, struct
import numpy as np

MAGIC = b"8db42d694ccc418790edff439fe026bf"
ap = argparse.ArgumentParser()
ap.add_argument("--insv", required=True); ap.add_argument("--dur", type=float, required=True)
ap.add_argument("--model", required=True); ap.add_argument("--n", type=int, required=True)
ap.add_argument("--out", required=True); ap.add_argument("--pano-rel", required=True)
ap.add_argument("--pano-stride", type=int, default=1)
ap.add_argument("--init-ply", default="")
a = ap.parse_args()

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

# --- GPS from trailer (auto offset-table parse, as gps_hybrid_scene.py) ---
S = os.path.getsize(a.insv); f = open(a.insv, "rb")
f.seek(S-72); hdr = f.read(72); assert hdr[40:72] == MAGIC, "not an A1 trailer"
ES = struct.unpack("<I", hdr[32:36])[0]; xs = S - ES
f.seek(S-72-4096); tail = f.read(4096)
cand = {}
for off in range(len(tail)-10):
    rid, fmt = tail[off], tail[off+1]
    sz, ro = struct.unpack("<II", tail[off+2:off+10])
    if 1 <= rid <= 18 and fmt <= 2 and 0 < sz < ES and ro+sz <= ES:
        cand.setdefault(rid, []).append((ro, sz))
bounds = set()
for lst in cand.values():
    for ro, sz in lst: bounds.add(ro); bounds.add(ro+sz)
tab = {}
for rid, lst in cand.items():
    pick = max(lst, key=lambda e: ((e[0]+e[1]) in bounds)+(e[0] in bounds))
    tab[rid] = (xs+pick[0], pick[1])
assert 7 in tab, f"no GPS record; found {sorted(tab)}"
off, ln = tab[7]; f.seek(off); d = f.read(ln); G = []
for i in range(0, len(d)-52, 53):
    ts = struct.unpack_from("<Q", d, i)[0] + struct.unpack_from("<H", d, i+8)[0]/1000
    if chr(d[i+10]) != 'A': continue
    lat = struct.unpack_from("<d", d, i+11)[0]; lon = struct.unpack_from("<d", d, i+20)[0]
    if chr(d[i+19]) == 'S': lat = -abs(lat)
    if chr(d[i+28]) == 'W': lon = -abs(lon)
    G.append((ts, lat, lon, struct.unpack_from("<d", d, i+45)[0]))
G.sort(); assert len(G) > 10, "too few GPS fixes"
t0, lat0, lon0, alt0 = G[0]
mlat = 111320.0; mlon = 111320.0*math.cos(math.radians(lat0))
grt = np.array([g[0]-t0 for g in G])
gE = np.array([(g[2]-lon0)*mlon for g in G]); gN = np.array([(g[1]-lat0)*mlat for g in G]); gU = np.array([g[3]-alt0 for g in G])
ft = np.arange(a.n)/(a.n/a.dur)
ENU = np.stack([np.interp(ft, grt, gE), np.interp(ft, grt, gU), np.interp(ft, grt, gN)], 1)  # (E,U,N) y-up

dn = read_model(a.model)
print(f"down SfM registered: {len(dn)}/{a.n}")
idx_of = lambda n: int(n.split("_")[1].split(".")[0])
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
    k = a.pano_stride * (idx_of(n) - 1) + 1
    cams.append({"idx": k, "image": f"{a.pano_rel}/pano_{k:04d}.jpg",
                 "R_wp": Rwp.tolist(), "T": (-Rwp@C).tolist(), "C": C.tolist(),
                 "n_crops": 1, "ref_yaw": 0.0, "ref_pitch": 0.0})
ce = np.array([c["C"] for c in cams])

# metric init cloud: GPS track jittered (same recipe as gps_init_023v2)
ply = a.init_ply
if not ply:
    ply = a.out.replace(".json", "_init.ply")
    rng = np.random.default_rng(0)
    base = ENU[rng.integers(0, a.n, 60000)]
    pts = base + rng.normal(0, 1, (60000, 3)) * np.array([25, 12, 25])
    pts[:, 1] -= 30 * rng.random(60000)  # bias below flight path (ground)
    col = np.full((60000, 3), 128, np.uint8)
    with open(ply, "wb") as fh:
        fh.write(b"ply\nformat binary_little_endian 1.0\n")
        fh.write(f"element vertex {len(pts)}\n".encode())
        fh.write(b"property float x\nproperty float y\nproperty float z\n")
        fh.write(b"property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        rec = np.zeros(len(pts), dtype=[("xyz", "f4", 3), ("rgb", "u1", 3)])
        rec["xyz"] = pts.astype(np.float32); rec["rgb"] = col
        fh.write(rec.tobytes())
    print(f"wrote init ply {ply}")
out = {"scene_dir": os.path.dirname(a.pano_rel),
       "point_cloud": ply,
       "cameras_extent": float(np.linalg.norm(ce-ce.mean(0), axis=1).max()*1.1), "cameras": cams}
json.dump(out, open(a.out, "w"), indent=1)
print(f"wrote {a.out} n={len(cams)}")
