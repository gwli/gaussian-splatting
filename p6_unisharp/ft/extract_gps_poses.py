#!/usr/bin/env python3
"""Generalized A1 .insv trailer -> metric 6DoF pano poses (pano_cams_*_gps.json).

Parses the Insta360/Antigravity v3 trailer (72B footer: pad32+size u32+ver u32+
magic32) by locating the 10-byte offset-table entries [id u8][fmt u8][size u32]
[offset u32] just before the footer (validated on 023: entries chain contiguously,
e.g. end(id9)==start(id7)). Extracts:
  GPS  (id=7) : 53B items  ts u64 + ms u16 + fix u8 + lat f64 + NS u8 + lon f64
                + EW u8 + speed f64 + track f64 + alt f64
  Quat (id=14): 36B items  ts u64 + 3*f32 + 4*f32 unit quaternion (30 Hz)

Then GPS->local ENU (metres), per-frame interp at fps=N/DUR, quats nearest,
R_wp from quat (xyzw, world->pano as validated on 023), writes a pano_cams json
(template's image names kept) + a metric random-init ply.

  extract_gps_poses.py --insv <file> --cams <pano_cams_template.json> \
      --dur <video_seconds> --out <out_json> [--n 240]
"""
import argparse, struct, os, json, math
import numpy as np

MAGIC = b"8db42d694ccc418790edff439fe026bf"

def parse_trailer_table(path):
    S = os.path.getsize(path)
    f = open(path, "rb")
    f.seek(S - 72); hdr = f.read(72)
    assert hdr[40:72] == MAGIC, "not an Insta360 trailer"
    extra_size = struct.unpack("<I", hdr[32:36])[0]
    extra_start = S - extra_size
    # scan a window before the footer for stride-10 offset-table entries
    win = 4096
    f.seek(S - 72 - win); tail = f.read(win)
    best = {}
    for off in range(len(tail) - 10):
        rid, fmt = tail[off], tail[off + 1]
        size, roff = struct.unpack_from("<II", tail[off + 2:off + 12][:8], 0) if False else struct.unpack("<II", tail[off + 2:off + 10])
        if 1 <= rid <= 18 and fmt <= 2 and 0 < size < extra_size and roff + size <= extra_size:
            # prefer entries that chain with others (validated layout); collect all, dedupe by id keeping the one whose (offset+size) or offset matches another entry boundary
            best.setdefault(rid, []).append((roff, size, off))
    # choose per-id entry participating in the contiguous chain
    bounds = set()
    for rid, lst in best.items():
        for roff, size, _ in lst:
            bounds.add(roff); bounds.add(roff + size)
    table = {}
    for rid, lst in best.items():
        chained = [e for e in lst if (e[0] + e[1]) in bounds or e[0] in bounds]
        pick = max(chained or lst, key=lambda e: ((e[0] + e[1]) in bounds) + (e[0] in bounds))
        table[rid] = (extra_start + pick[0], pick[1])
    return table, extra_start, S

def read_gps(path, table):
    off, ln = table[7]
    f = open(path, "rb"); f.seek(off); d = f.read(ln)
    G = []
    for i in range(0, len(d) - 52, 53):
        ts = struct.unpack_from("<Q", d, i)[0] + struct.unpack_from("<H", d, i + 8)[0] / 1000.0
        fix = chr(d[i + 10])
        lat = struct.unpack_from("<d", d, i + 11)[0]
        lon = struct.unpack_from("<d", d, i + 20)[0]
        if chr(d[i + 19]) == 'S': lat = -abs(lat)
        if chr(d[i + 28]) == 'W': lon = -abs(lon)
        alt = struct.unpack_from("<d", d, i + 45)[0]
        if fix == 'A':
            G.append((ts, lat, lon, alt))
    G.sort()
    return G

def read_quats(path, table):
    off, ln = table[14]
    f = open(path, "rb"); f.seek(off); d = f.read(ln)
    T, Q = [], []
    for i in range(0, len(d) - 35, 36):
        T.append(struct.unpack_from("<Q", d, i)[0] / 1000.0)
        Q.append(struct.unpack_from("<ffff", d, i + 20))
    T = np.array(T); Q = np.array(Q)
    n = np.linalg.norm(Q, axis=1)
    ok = np.abs(n - 1.0) < 0.05
    return T[ok] - T[ok][0], Q[ok]

def quat2R_xyzw(q):
    x, y, z, w = q / (np.linalg.norm(q) + 1e-9)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--insv", required=True)
    ap.add_argument("--cams", required=True, help="existing pano_cams json (image names template)")
    ap.add_argument("--dur", type=float, required=True, help="video duration seconds")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--init-ply", default=None)
    a = ap.parse_args()

    table, xs, S = parse_trailer_table(a.insv)
    print(f"[trailer] records: " + " ".join(f"id{k}@{v[0]-xs}({v[1]}B)" for k, v in sorted(table.items())))
    assert 7 in table, "no GPS record (id=7) in trailer"
    assert 14 in table, "no quaternion record (id=14) in trailer"

    G = read_gps(a.insv, table)
    print(f"[gps] {len(G)} acquired samples, span {G[-1][0]-G[0][0]:.1f}s")
    t0, lat0, lon0, alt0 = G[0]
    mlat = 111320.0; mlon = 111320.0 * math.cos(math.radians(lat0))
    gt = np.array([g[0] - t0 for g in G])
    gE = np.array([(g[2] - lon0) * mlon for g in G])
    gN = np.array([(g[1] - lat0) * mlat for g in G])
    gU = np.array([g[3] - alt0 for g in G])

    QT, QQ = read_quats(a.insv, table)
    print(f"[quat] {len(QQ)} samples, span {QT[-1]:.1f}s")

    N = a.n; fps = N / a.dur
    ft = np.array([k / fps for k in range(N)])
    C = np.stack([np.interp(ft, gt, gE), np.interp(ft, gt, gN), np.interp(ft, gt, gU)], 1)
    qi = np.clip(np.searchsorted(QT, ft), 0, len(QQ) - 1)

    tmpl = json.load(open(a.cams))
    byidx = {c["idx"]: c for c in tmpl["cameras"]}
    cams = []
    for k in range(N):
        Rwp = quat2R_xyzw(QQ[qi[k]])          # world->pano, xyzw (023-validated)
        c = C[k]; T = -Rwp @ c
        src = byidx.get(k + 1, {})
        cams.append({"idx": k + 1, "image": src.get("image", f"pano_{k+1:04d}.jpg"),
                     "R_wp": Rwp.tolist(), "T": T.tolist(), "C": c.tolist(),
                     "n_crops": src.get("n_crops", 1),
                     "ref_yaw": src.get("ref_yaw", 0.0), "ref_pitch": src.get("ref_pitch", 0.0)})
    out = dict(tmpl); out["cameras"] = cams
    cext = np.array([c["C"] for c in cams])
    out["cameras_extent"] = float(np.linalg.norm(cext - cext.mean(0), axis=1).max() * 1.1)

    if a.init_ply:
        from plyfile import PlyData, PlyElement
        lo, hi = cext.min(0), cext.max(0)
        n = 200000
        xyz = np.stack([np.random.uniform(lo[0]-120, hi[0]+120, n),
                        np.random.uniform(lo[1]-120, hi[1]+120, n),
                        np.random.uniform(lo[2]-60,  hi[2]+10,  n)], 1).astype(np.float32)
        verts = np.empty(n, dtype=[("x","f4"),("y","f4"),("z","f4"),("red","u1"),("green","u1"),("blue","u1")])
        verts["x"], verts["y"], verts["z"] = xyz[:,0], xyz[:,1], xyz[:,2]
        verts["red"] = verts["green"] = verts["blue"] = 160
        PlyData([PlyElement.describe(verts, "vertex")]).write(a.init_ply)
        out["point_cloud"] = a.init_ply
        print(f"[init] metric cloud -> {a.init_ply}")

    json.dump(out, open(a.out, "w"), indent=1)
    print(f"[out] {a.out}  centers span {np.round(cext.ptp(0),1).tolist()} m  extent={out['cameras_extent']:.1f}")

if __name__ == "__main__":
    main()
