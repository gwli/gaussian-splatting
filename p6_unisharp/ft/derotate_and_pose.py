#!/usr/bin/env python3
"""A-tier step 1: de-rotate each ERP pano into a common (world) orientation so the
residual between frames is translation-only — the contract UniSHARP's panorama
training (SimPanorama, fixed orientation) expects — and emit a pose CSV.

Input : pano_cams_scene_<S>.json  (per-frame R_wp [world->pano], C [world center])
        panoramas/pano_*.jpg
Output: <out_rgb>/<frame:05d>.jpg   canonical ERP (world-aligned axes)
        <out_pose>                  CSV: frame,x,y,z  (= C * pos_scale, metres)

ERP convention (matches UniK3D Spherical: lon=atan2(x,z), lat=asin(y)):
  pixel(u,v) -> lon=(u+.5)/W*2pi-pi ,  lat=pi/2-(v+.5)/H*pi
  world ray  d_w=[cos lat sin lon, sin lat, cos lat cos lon]
  pano  ray  d_p = R_wp @ d_w   ->  sample source pano at (lon_p,lat_p)
A uniform convention offset is harmless (it only redefines the canonical frame);
what matters is R_wp is applied consistently to every frame.
"""
import argparse, csv, json, re
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--cams", required=True)
ap.add_argument("--panodir", required=True)
ap.add_argument("--out-rgb", required=True)
ap.add_argument("--out-pose", required=True)
ap.add_argument("--pos-scale", type=float, default=1.0,
                help="VGGT->metre scale for C. See task_ft.md §3.2; tune so adjacent "
                     "frame steps match real flight (≈0.1-1m).")
ap.add_argument("--H", type=int, default=1024)
ap.add_argument("--W", type=int, default=2048)
ap.add_argument("--gravity-level", action="store_true",
                help="B-tier: estimate 'up' from the camera trajectory (PCA, smallest "
                     "axis = flight-plane normal) and rotate all frames to flatten the "
                     "horizon, so the canonical ERP is upright (UniSHARP trains upright).")
ap.add_argument("--up-sign", type=int, default=0, choices=[-1, 0, 1],
                help="0=auto (top-vs-bottom brightness: sky should be up); ±1 force.")
ap.add_argument("--level-method", default="horizon", choices=["horizon", "trajectory"],
                help="horizon: fit the sky/ground boundary to a tilted great circle "
                     "(robust to banked flight). trajectory: PCA of camera path normal "
                     "(only valid for level flight).")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
a = ap.parse_args()

dev = torch.device(a.device)
cams = json.load(open(a.cams))["cameras"]
panodir = Path(a.panodir)
out_rgb = Path(a.out_rgb); out_rgb.mkdir(parents=True, exist_ok=True)
Path(a.out_pose).parent.mkdir(parents=True, exist_ok=True)

H, W = a.H, a.W
# canonical output ray grid (world frame), shape [H,W,3]
vv, uu = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")
lon = (uu + 0.5) / W * 2 * np.pi - np.pi
lat = np.pi / 2 - (vv + 0.5) / H * np.pi
clat = torch.cos(lat)
dw = torch.stack([clat * torch.sin(lon), torch.sin(lat), clat * torch.cos(lon)], -1)  # [H,W,3]

def find_pano(idx, image_field):
    # prefer the json's image field, else pano_<idx>.jpg variants
    cand = []
    if image_field:
        cand.append(panodir / Path(image_field).name)
    cand += [panodir / f"pano_{idx:04d}.jpg", panodir / f"pano_{idx:05d}.jpg"]
    for c in cand:
        if c.exists():
            return c
    return None

def remap(Reff_t, src_path):
    """Sample source ERP at pano rays Reff @ dw (Reff = world->pano combined)."""
    dp = torch.einsum("ij,hwj->hwi", Reff_t, dw)         # pano-frame rays [H,W,3]
    lon_p = torch.atan2(dp[..., 0], dp[..., 2])
    lat_p = torch.asin(dp[..., 1].clamp(-1 + 1e-6, 1 - 1e-6))
    u_p = torch.remainder((lon_p + np.pi) / (2 * np.pi) * W, W)
    v_p = (np.pi / 2 - lat_p) / np.pi * H
    grid = torch.stack([u_p / (W - 1) * 2 - 1, v_p / (H - 1) * 2 - 1], -1).unsqueeze(0)
    img = Image.open(src_path).convert("RGB").resize((W, H), Image.BILINEAR)
    t = torch.from_numpy(np.asarray(img, np.float32)).permute(2, 0, 1).unsqueeze(0).to(dev) / 255.0
    out = F.grid_sample(t, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return (out[0].permute(1, 2, 0).clamp(0, 1) * 255).round().byte().cpu().numpy()

def rot_a_to_b(av, bv):
    """3x3 rotation sending unit av -> unit bv (Rodrigues)."""
    av = av / (np.linalg.norm(av) + 1e-12); bv = bv / (np.linalg.norm(bv) + 1e-12)
    v = np.cross(av, bv); c = float(np.dot(av, bv))
    if np.linalg.norm(v) < 1e-8:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))

def estimate_up_horizon(sample_k=12):
    """Average world 'up' from the sky/ground boundary fitted to a tilted great
    circle per frame. Horizon rays r satisfy r·n=0 (n=up); in (lon,lat):
      sin(lon)*n_x + tan(lat)*n_y + cos(lon)*n_z = 0  -> homogeneous LS for n.
    """
    sw, sh = 1024, 512
    lons = (np.arange(sw) + 0.5) / sw * 2 * np.pi - np.pi
    sel = cams[:: max(1, len(cams) // sample_k)][:sample_k]
    ups = []
    for c in sel:
        sp = find_pano(int(c["idx"]), c.get("image"))
        if sp is None:
            continue
        g = np.asarray(Image.open(sp).convert("L").resize((sw, sh)), np.float32)
        r0, r1 = int(0.12 * sh), int(0.75 * sh)          # search band for the boundary
        # per-column row of strongest bright(top)->dark(bottom) transition
        d = 4
        grad = g[r0 + d : r1 - d] - g[r0 + 2 * d : r1]   # luma drop downward
        rows = np.argmax(grad, axis=0) + r0 + d
        strength = np.max(grad, axis=0)
        keep = strength > max(8.0, np.percentile(strength, 40))
        if keep.sum() < 20:
            continue
        lon_k = lons[keep]
        lat_k = np.pi / 2 - (rows[keep] + 0.5) / sh * np.pi
        A = np.stack([np.sin(lon_k), np.tan(np.clip(lat_k, -1.4, 1.4)), np.cos(lon_k)], 1)
        _, _, Vt = np.linalg.svd(A, full_matrices=False)
        n = Vt[-1]
        if n[1] < 0:                                     # orient toward sky (+y in pano)
            n = -n
        Rwp = np.array(c["R_wp"], dtype=np.float64)
        ups.append(Rwp.T @ n)                            # pano-frame up -> world up
    if not ups:
        raise RuntimeError("horizon up-estimate failed (no usable frames)")
    up = np.mean(ups, 0); up /= np.linalg.norm(up) + 1e-12
    spread = float(np.mean([1 - abs(np.dot(up, u / (np.linalg.norm(u) + 1e-12))) for u in ups]))
    print(f"[derotate] horizon up from {len(ups)} frames, consistency spread={spread:.3f}")
    return up

# --- gravity leveling: R_level maps VGGT world -> upright (up -> +Y) ----------
R_level = np.eye(3, dtype=np.float64)
if a.gravity_level:
    if a.level_method == "horizon":
        up = estimate_up_horizon()
    else:
        Cs = np.array([c["C"] for c in cams], dtype=np.float64)
        X = Cs - Cs.mean(0)
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        up = Vt[-1]                                      # flight-plane normal ≈ up
    sign = a.up_sign
    if sign == 0:  # auto: pick sign so sky (brighter) lands on top
        mid = cams[len(cams) // 2]
        sp = find_pano(int(mid["idx"]), mid.get("image"))
        best, sign = None, 1
        for s in (1, -1):
            Rl = rot_a_to_b(up * s, np.array([0.0, 1.0, 0.0]))
            Reff = torch.tensor(np.array(mid["R_wp"], float) @ Rl.T, dtype=torch.float32, device=dev)
            arr = remap(Reff, sp).astype(np.float32)
            top = arr[: H // 4].mean(); bot = arr[3 * H // 4 :].mean()
            score = top - bot                            # want sky(top) brighter
            if best is None or score > best:
                best, sign = score, s
    up = up * sign
    R_level = rot_a_to_b(up, np.array([0.0, 1.0, 0.0]))
    print(f"[derotate] gravity-level ON: up_vggt={np.round(up,3).tolist()} sign={sign}")

R_level_t = torch.tensor(R_level, dtype=torch.float32, device=dev)

rows = []
n_ok = 0
for c in cams:
    idx = int(c["idx"])
    src = find_pano(idx, c.get("image"))
    if src is None:
        print(f"[skip] idx={idx}: no pano file"); continue
    Rwp = torch.tensor(np.array(c["R_wp"], dtype=np.float32), device=dev)   # world->pano
    Reff = Rwp @ R_level_t.T                                                # gravity ray -> pano
    C = R_level @ (np.array(c["C"], dtype=np.float64) * a.pos_scale)        # center in gravity frame
    arr = remap(Reff, src)
    Image.fromarray(arr).save(out_rgb / f"{idx:05d}.jpg", quality=95)
    rows.append((idx, float(C[0]), float(C[1]), float(C[2])))
    n_ok += 1

rows.sort()
with open(a.out_pose, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["frame", "x", "y", "z"]); w.writerows(rows)
print(f"[derotate] {n_ok} frames -> {out_rgb}  |  pose -> {a.out_pose}  "
      f"(pos_scale={a.pos_scale}, gravity_level={a.gravity_level})")
