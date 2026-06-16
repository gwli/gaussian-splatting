#!/usr/bin/env python3
"""task2: virtual tour highlight-video renderer for a trained pano 3DGS model.

Loads a trained INRIA-format .ply into gsplat, derives the scene frame
(center + gravity-up from the gravity-aligned pano camera orientations + radius
from the camera spread), generates cinematic virtual camera paths (orbit /
flythrough-spline / dolly-in), renders frames with gsplat (perspective 16:9 or
9:16, or 360 equirect via the T-F8 fused kernel), and writes a PNG sequence that
run_tour.sh encodes to mp4.

Usage: tour_render.py <ply> <pano_cams.json> <out_frame_dir>
         [--shots orbit,fly,dolly] [--res 1920x1080] [--fps 30] [--secs 8]
         [--hfov 70] [--mode perspective|equirect]
"""
import sys, os, re, json, math, argparse, numpy as np, torch
from plyfile import PlyData
REPO = "/w" if os.path.exists("/w/p3_pano/gsplat_equirect.py") else "/raid/git/gaussian-splatting"
sys.path.insert(0, REPO + "/p3_pano")
from gsplat import rasterization
from gsplat_equirect import render_equirect_fused

ap = argparse.ArgumentParser()
ap.add_argument("ply"); ap.add_argument("cams"); ap.add_argument("outdir")
ap.add_argument("--shots", default="orbit,fly,dolly")
ap.add_argument("--res", default="1920x1080")
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--secs", type=float, default=8.0)        # per shot
ap.add_argument("--hfov", type=float, default=70.0)
ap.add_argument("--mode", default="perspective", choices=["perspective", "equirect"])
ap.add_argument("--keyframes", default=None, help="director JSON: scene-relative cylindrical keys")
ap.add_argument("--poi", default=None, help="'auto' or 'x,y,z;x,y,z' world POIs; renders a POI tour")
ap.add_argument("--split", action="store_true", help="write each shot to its own subdir + manifest (for crossfade)")
ap.add_argument("--poi-gps", dest="poi_gps", default=None, help="'lat,lon,alt;...' GPS POIs (needs --align)")
ap.add_argument("--align", default=None, help="gps_align.py output json (Sim3 GPS->scene)")
ap.add_argument("--poi-pano", dest="poi_pano", default=None, help="'idx:u,v;...' manual POI by pano pixel (u,v in 0..1)")
a = ap.parse_args()
W, H = (int(x) for x in a.res.split("x"))
if a.mode == "equirect":
    W, H = 1024, 512
dev = "cuda"; SH = 3
os.makedirs(a.outdir, exist_ok=True)

# ---------------- load trained gaussians (INRIA ply) ----------------
v = PlyData.read(a.ply)["vertex"]
xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
N = xyz.shape[0]
f_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1).astype(np.float32)          # (N,3)
nrest = sum(1 for p in v.properties if p.name.startswith("f_rest_"))
f_rest = np.stack([v[f"f_rest_{i}"] for i in range(nrest)], 1).astype(np.float32)        # (N,45)
opac = np.asarray(v["opacity"], np.float32)
scl = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1).astype(np.float32)
rot = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], 1).astype(np.float32)
# SH coeffs (N, K, 3): f_dc + f_rest(reshape channel-major (N,3,15)->(N,15,3))
K = nrest // 3 + 1
shN = f_rest.reshape(N, 3, nrest // 3).transpose(0, 2, 1)
sh = np.concatenate([f_dc[:, None, :], shN], 1).astype(np.float32)                       # (N,K,3)
means = torch.tensor(xyz, device=dev)
quats = torch.tensor(rot, device=dev)
scales = torch.exp(torch.tensor(scl, device=dev))
opacities = torch.sigmoid(torch.tensor(opac, device=dev))
colors = torch.tensor(sh, device=dev)
print(f"[tour] loaded {N} gaussians, SH K={K} from {os.path.basename(a.ply)}")

# ---------------- scene frame from pano cameras ----------------
meta = json.load(open(a.cams))
C = np.array([np.array(c["C"], np.float32) for c in meta["cameras"]])                    # (M,3) centers
Rwp = [np.array(c["R_wp"], np.float32) for c in meta["cameras"]]                         # world->view
center = np.median(C, 0)
# gravity up: pano view +y is DOWN (gravity-aligned equirect) -> world down = R^T[0,1,0]
down = np.mean([Rw.T @ np.array([0, 1, 0.]) for Rw in Rwp], 0)
up = -down / (np.linalg.norm(down) + 1e-9)
# horizontal basis perpendicular to up
e1 = np.cross(up, [1, 0, 0.]);
if np.linalg.norm(e1) < 1e-3: e1 = np.cross(up, [0, 1, 0.])
e1 /= np.linalg.norm(e1); e2 = np.cross(up, e1); e2 /= np.linalg.norm(e2)
Ch = C - center
horiz = Ch - np.outer(Ch @ up, up)
radius = float(np.median(np.linalg.norm(horiz, axis=1))) * 1.25 + 1e-3
hmean = float(np.mean(Ch @ up))
print(f"[tour] center={center.round(2)} up={up.round(2)} radius={radius:.2f} h={hmean:.2f}")

def lookat_viewmat(eye, target):
    f = target - eye; f /= np.linalg.norm(f) + 1e-9          # forward (+z cam)
    r = np.cross(f, up); r /= np.linalg.norm(r) + 1e-9       # right (+x cam)
    u = np.cross(r, f)                                       # up
    ydown = -u                                               # +y cam = down
    Rcw = np.stack([r, ydown, f], 1)                         # cam->world cols
    Rwc = Rcw.T
    vm = np.eye(4, dtype=np.float32)
    vm[:3, :3] = Rwc; vm[:3, 3] = -Rwc @ eye
    return torch.tensor(vm, device=dev)

# ---------------- camera paths ----------------
def ease(t):  # smoothstep for gentle accel/decel
    return t * t * (3 - 2 * t)

def shot_orbit(n):
    poses = []
    for i in range(n):
        t = ease(i / max(n - 1, 1)); th = 2 * math.pi * t * 0.9      # ~0.9 revolution
        eye = center + radius * 1.05 * (math.cos(th) * e1 + math.sin(th) * e2) + (hmean + 0.12 * radius) * up
        poses.append(lookat_viewmat(eye, center))                    # stay near capture altitude (well-observed)
    return poses

def shot_dolly(n):
    poses = []
    a0 = center + radius * 1.4 * e1 + (hmean + 0.2 * radius) * up    # far
    a1 = center + radius * 0.45 * e1 + (hmean + 0.05 * radius) * up  # near (reveal)
    for i in range(n):
        t = ease(i / max(n - 1, 1)); eye = (1 - t) * a0 + t * a1
        poses.append(lookat_viewmat(eye, center))
    return poses

def catmull(p, t):  # p: list of 4 points, t in [0,1]
    p0, p1, p2, p3 = p
    t2 = t * t; t3 = t2 * t
    return 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)

def shot_fly(n):
    # Catmull-Rom spline through a spread subset of camera centers (raised slightly)
    idx = np.linspace(0, len(C) - 1, 8).round().astype(int)
    pts = C[idx] + 0.1 * radius * up                         # near the captured flight line (well-observed)
    pts = np.vstack([pts[0], pts, pts[-1]])                 # pad ends
    poses = []
    segs = len(pts) - 3
    for i in range(n):
        t = ease(i / max(n - 1, 1)) * segs; s = min(int(t), segs - 1); lt = t - s
        eye = catmull(pts[s:s + 4], lt)
        nxt = catmull(pts[s:s + 4], min(lt + 0.05, 1.0))
        tgt = eye + (nxt - eye) * 8 + (center - eye) * 0.15   # look ahead, biased to center
        poses.append(lookat_viewmat(eye, tgt))
    return poses

def detect_pois(k=3):
    # auto POIs: densest voxels of the gaussian cloud (structure, not sky/fog)
    P = xyz
    lo, hi = np.percentile(P, 2, 0), np.percentile(P, 98, 0)
    G = 20
    idx = np.clip(((P - lo) / (hi - lo + 1e-9) * G).astype(int), 0, G - 1)
    keymap = {}
    for p, c in zip(P, idx):
        key = tuple(c); keymap.setdefault(key, []).append(p)
    cells = sorted(keymap.values(), key=len, reverse=True)
    # spread the top cells (skip near-duplicate centroids)
    pois = []
    for cell in cells:
        c = np.mean(cell, 0)
        if all(np.linalg.norm(c - q) > 0.3 * radius for q in pois):
            pois.append(c)
        if len(pois) >= k: break
    return pois

def shot_poi_one(poi, n):
    poses = []
    for i in range(n):
        t = ease(i / max(n - 1, 1)); th = math.pi * (0.6 * t - 0.3)   # small arc
        eye = poi + radius * 0.6 * (math.cos(th) * e1 + math.sin(th) * e2) + 0.12 * radius * up
        poses.append(lookat_viewmat(eye, poi))                         # dwell/frame the POI
    return poses

def shot_keyframes(n_total):
    # director keys: scene-relative cylindrical {az(deg), r(xradius), h(xradius), t(sec)}
    keys = json.load(open(a.keyframes))["keys"]
    def eye_of(kf):
        az = math.radians(kf["az"])
        return center + kf["r"] * radius * (math.cos(az) * e1 + math.sin(az) * e2) + \
               (hmean + kf["h"] * radius) * up
    eyes = np.array([eye_of(k) for k in keys])
    eyes = np.vstack([eyes[0], eyes, eyes[-1]])                         # pad for catmull
    ts = [k.get("t", i) for i, k in enumerate(keys)]
    total = sum(ts) if "t" in keys[0] else len(keys)
    poses, segs = [], len(eyes) - 3
    n = int(a.fps * (total if "t" in keys[0] else a.secs * len(keys)))
    for i in range(n):
        u = ease(i / max(n - 1, 1)) * segs; s = min(int(u), segs - 1); lt = u - s
        eye = catmull(eyes[s:s + 4], lt)
        poses.append(lookat_viewmat(eye, center))
    return poses

SHOTS = {"orbit": shot_orbit, "fly": shot_fly, "dolly": shot_dolly}
n_per = int(a.fps * a.secs)

# ---------------- render ----------------
def render(vm):
    if a.mode == "equirect":
        cam_c = torch.tensor(np.linalg.inv(vm.cpu().numpy())[:3, 3], device=dev)
        img, _ = render_equirect_fused(means, quats, scales, opacities, colors, vm, cam_c, W, H, SH)
        return img.clamp(0, 1)
    fx = fy = (W / 2) / math.tan(math.radians(a.hfov) / 2)
    Ks = torch.tensor([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]], dtype=torch.float32, device=dev)[None]
    rc, _, _ = rasterization(means, quats, scales, opacities, colors, vm[None], Ks, W, H,
                             sh_degree=SH, render_mode="RGB")
    return rc[0].clamp(0, 1)

# build the shot list (segment_name -> poses)
def gps_to_scene(lat, lon, alt, al):
    lat0, lon0, alt0 = al["lat0"], al["lon0"], al["alt0"]
    dN = (lat - lat0) * 111320.0
    dE = (lon - lon0) * 111320.0 * math.cos(math.radians(lat0))
    enu = np.array([dE, dN, alt - alt0])
    return al["scale"] * (np.array(al["R"]) @ enu) + np.array(al["t"])

def pano_pixel_to_poi(idx, u, v):
    # manual POI: back-project an equirect pixel (u,v in 0..1) of pano `idx` into
    # the scene by marching the ray to the nearest gaussian (LONLAT convention).
    c = next(c for c in meta["cameras"] if c["idx"] == int(idx))
    R = np.array(c["R_wp"], float); C0 = np.array(c["C"], float)
    lon = (2 * u - 1) * math.pi; lat = (2 * v - 1) * (math.pi / 2)
    d_view = np.array([math.cos(lat) * math.sin(lon), math.sin(lat), math.cos(lat) * math.cos(lon)])
    d = R.T @ d_view; d /= np.linalg.norm(d)            # world ray dir (R is world->view)
    rel = xyz - C0
    tproj = rel @ d
    perp = np.linalg.norm(rel - np.outer(tproj, d), axis=1)
    m = (tproj > 0.02 * radius) & (perp < 0.06 * radius)
    if not m.any():
        return C0 + d * radius
    cand = np.where(m)[0]
    return xyz[cand[np.argmin(tproj[cand])]]            # nearest gaussian along the ray

segments = []
if a.keyframes:
    segments.append(("keyframes", shot_keyframes(0)))
elif a.poi or a.poi_gps or a.poi_pano:
    pois = []
    if a.poi == "auto":
        pois += detect_pois(3)
    elif a.poi:
        pois += [np.array([float(x) for x in p.split(",")]) for p in re.split(r"[;+]", a.poi)]
    if a.poi_gps:
        al = json.load(open(a.align))
        pois += [gps_to_scene(*[float(x) for x in p.split(",")], al) for p in re.split(r"[;+]", a.poi_gps)]
    if a.poi_pano:
        for p in re.split(r"[;+]", a.poi_pano):
            idx, uv = p.split(":"); u, v = (float(x) for x in uv.split(","))
            pois.append(pano_pixel_to_poi(idx, u, v))
    print(f"[tour] {len(pois)} POIs: {[np.round(p,2).tolist() for p in pois]}")
    for j, p in enumerate(pois):
        segments.append((f"poi{j}", shot_poi_one(p, n_per)))
else:
    for name in a.shots.split(","):
        segments.append((name, SHOTS[name](n_per)))

from PIL import Image
fi = 0
manifest = []
for si, (name, poses) in enumerate(segments):
    print(f"[tour] shot '{name}': {len(poses)} frames")
    if a.split:
        segdir = os.path.join(a.outdir, f"seg{si:02d}_{name}")
        os.makedirs(segdir, exist_ok=True)
        manifest.append({"idx": si, "name": name, "nframes": len(poses), "dir": os.path.basename(segdir)})
    with torch.no_grad():
        for j, vm in enumerate(poses):
            img = (render(vm).cpu().numpy() * 255).astype(np.uint8)
            if a.split:
                Image.fromarray(img).save(os.path.join(segdir, f"frame_{j:05d}.png"))
            else:
                Image.fromarray(img).save(os.path.join(a.outdir, f"frame_{fi:05d}.png"))
            fi += 1
if a.split:
    json.dump({"fps": a.fps, "segments": manifest}, open(os.path.join(a.outdir, "segments.json"), "w"), indent=1)
    print(f"[tour] split manifest: {len(manifest)} segments")
print(f"[tour] wrote {fi} frames ({W}x{H}) to {a.outdir}")
