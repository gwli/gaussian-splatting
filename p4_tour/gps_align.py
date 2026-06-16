#!/usr/bin/env python3
"""task2 step-3: align a flight GPS track to the reconstructed (VGGT) scene frame,
giving metric scale + north/orientation, and a GPS->scene mapping for GPS POIs.

GPS is in an arbitrary geodetic frame; VGGT poses are in an arbitrary metric
frame. We convert GPS to local ENU meters, then Umeyama-fit a Sim3 from ENU to
the per-pano camera centers (matched by index/time). The Sim3 then maps any GPS
(lat,lon,alt) POI into scene coordinates.

NOTE: the provided scene_023 .insv carries NO GPS (exiftool: 15 QuickTime tags,
no telemetry track) — this drone clip didn't log GPS. So real alignment is N/A
here; `--selftest` validates the math on synthetic GPS, and the path is ready for
any clip that does carry GPS (extract with `exiftool -ee -p '$GPSLatitude,...'`).

Usage:
  gps_align.py <gps.json> <pano_cams.json> <out_align.json>   # gps.json: [{idx,lat,lon,alt}]
  gps_align.py --selftest <pano_cams.json>                    # synthetic recovery test
"""
import sys, json, numpy as np

def umeyama(src, dst):                       # dst ~ s R src + t
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    H = d0.T @ s0 / len(src)
    U, D, Vt = np.linalg.svd(H)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0: S[2, 2] = -1
    R = U @ S @ Vt
    var = (s0 ** 2).sum() / len(src)
    s = np.trace(np.diag(D) @ S) / var
    t = mu_d - s * R @ mu_s
    return s, R, t

def gps_to_enu(lat, lon, alt, lat0, lon0, alt0):
    dN = (lat - lat0) * 111320.0
    dE = (lon - lon0) * 111320.0 * np.cos(np.radians(lat0))
    dU = alt - alt0
    return np.array([dE, dN, dU])

def fit(enu, centers):
    s, R, t = umeyama(enu, centers)
    resid = float(np.linalg.norm((s * (R @ enu.T).T + t) - centers, axis=1).mean())
    north_scene = (R @ np.array([0., 1, 0]))   # ENU North -> scene direction
    return {"scale": float(s), "R": R.tolist(), "t": t.tolist(),
            "resid_m_to_scene": resid, "north_scene": north_scene.tolist(),
            "lat0": None, "lon0": None, "alt0": None}

if sys.argv[1] == "--selftest":
    meta = json.load(open(sys.argv[2]))
    C = np.array([np.array(c["C"], float) for c in meta["cameras"]])
    rng = np.random.default_rng(0)
    # invent a Sim3 mapping a realistic-metric ENU track (tens of m) -> the small
    # scene (extent ~0.6 scene-units) -> scale ~0.02 scene-units/meter.
    s_t = 0.02; A = rng.standard_normal((3, 3)); Rt, _ = np.linalg.qr(A)
    if np.linalg.det(Rt) < 0: Rt[:, 0] = -Rt[:, 0]
    t_t = rng.standard_normal(3) * 0.3
    enu_true = ((np.linalg.inv(Rt) @ (C - t_t).T).T / s_t)        # so s_t,Rt,t_t maps enu->C (enu ~ tens of m)
    NOISE_M = 0.5                                                  # 0.5 m GPS noise (good consumer/RTK)
    enu = enu_true + rng.standard_normal(enu_true.shape) * NOISE_M
    al = fit(enu, C)
    print(f"[selftest] true scale {s_t:.3f}  recovered {al['scale']:.3f}  "
          f"(err {abs(al['scale']-s_t)/s_t*100:.2f}%)")
    print(f"[selftest] alignment residual {al['resid_m_to_scene']:.4f} scene-units "
          f"(= {al['resid_m_to_scene']/al['scale']:.2f} m; GPS noise was {NOISE_M} m)")
    # POI check: a GPS point at enu_true[10] must map back near C[10]
    poi_scene = al["scale"] * (np.array(al["R"]) @ enu_true[10]) + np.array(al["t"])
    print(f"[selftest] GPS->scene POI error: {np.linalg.norm(poi_scene - C[10]):.4f}")
    sys.exit(0)

gps_json, cams_json, out = sys.argv[1], sys.argv[2], sys.argv[3]
gps = json.load(open(gps_json))
meta = json.load(open(cams_json))
cams = {c["idx"]: np.array(c["C"], float) for c in meta["cameras"]}
g = [p for p in gps if p.get("idx") in cams]
lat0, lon0, alt0 = g[0]["lat"], g[0]["lon"], g[0].get("alt", 0)
enu = np.array([gps_to_enu(p["lat"], p["lon"], p.get("alt", 0), lat0, lon0, alt0) for p in g])
C = np.array([cams[p["idx"]] for p in g])
al = fit(enu, C); al.update(lat0=lat0, lon0=lon0, alt0=alt0)
json.dump(al, open(out, "w"), indent=1)
print(f"[gps_align] {len(g)} matched | scale {al['scale']:.4f} | resid {al['resid_m_to_scene']:.4f} -> {out}")
