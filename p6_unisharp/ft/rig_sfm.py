#!/usr/bin/env python3
"""Rig-constrained two-lens SfM (pycolmap 4.x).

Motivation (§34): held-out pose error of 0.229deg / 0.167m costs 1.11 dB -- the
largest remaining lever. Until now SfM used the DOWN lens only and threw the UP
lens away. Here both lenses enter as one rigid frame with the photometrically
solved R_rig fixed as the sensor-to-rig extrinsic, which doubles the observations
and constrains rotation (the expensive DOF) via the 360-degree baseline.

Frame k contributes two images: down/f_k.jpg (reference sensor) and up/f_k.jpg.
usage: rig_sfm.py <scene_dir> <out_dir> [--cams-json J] (J selects which frames)
"""
import sys, os, json, argparse, shutil
import numpy as np
import pycolmap

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/8kpano/scenes/fish023rig")
ap.add_argument("--cams-json", default="p3_pano/fair023_d2.json")
ap.add_argument("--rig-npz", default="p3_pano/rig023.npz")
ap.add_argument("--out", default="data/8kpano/scenes/fish023rig/sparse_rig")
ap.add_argument("--overlap", type=int, default=20)
ap.add_argument("--f", type=float, default=547.11)      # self-calibrated @1920 (023)
ap.add_argument("--lenses", default="down,up",
                help="'down,up' = rig-constrained; 'down' = single-lens control "
                     "on the identical frame set (isolates the frame-count factor)")
ap.add_argument("--tag", default="rig")
ap.add_argument("--mask-up", action="store_true",
                help="restrict UP-lens features to the horizon ring (theta 80-100 deg). "
                     "The up lens otherwise images sky/clouds, whose features move and "
                     "corrupt BA (see section 35).")
a = ap.parse_args()
LENSES = a.lenses.split(",")

IMG = os.path.join(a.root, f"images_{a.tag}")   # images_<tag>/<lens>/f_XXXX.jpg
os.makedirs(IMG, exist_ok=True)
idxs = sorted(c["idx"] for c in json.load(open(a.cams_json))["cameras"])

# ---- stage 1: select the frames we evaluate on, into the rig layout ----------
for sub in LENSES:
    d = os.path.join(IMG, sub); os.makedirs(d, exist_ok=True)
    src = os.path.join(a.root, f"{sub}_all")
    for k in idxs:
        s, t = os.path.join(src, f"f_{k:04d}.jpg"), os.path.join(d, f"f_{k:04d}.jpg")
        if not os.path.exists(t):
            os.link(s, t) if os.path.exists(s) else None
n_have = len(os.listdir(os.path.join(IMG, LENSES[0])))
print(f"[rig-sfm] frames selected: {n_have}/{len(idxs)} (lenses: {LENSES})", flush=True)

# ---- stage 2: rig config -----------------------------------------------------
z = np.load(a.rig_npz); R_rig = np.asarray(z["R_rig"], float)   # d_up = R_rig @ d_down
def R2q(R):                                                      # -> [w,x,y,z]
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2; q = [0.25*s, (R[2,1]-R[1,2])/s, (R[0,2]-R[2,0])/s, (R[1,0]-R[0,1])/s]
    else:
        i = int(np.argmax([R[0,0], R[1,1], R[2,2]])); j, k = (i+1) % 3, (i+2) % 3
        s = np.sqrt(1.0 + R[i,i] - R[j,j] - R[k,k]) * 2
        q = [0, 0, 0, 0]; q[0] = (R[k,j]-R[j,k])/s; q[i+1] = 0.25*s
        q[j+1] = (R[j,i]+R[i,j])/s; q[k+1] = (R[k,i]+R[i,k])/s
    q = np.array(q); return q / np.linalg.norm(q)

# NOTE: build the config as pycolmap OBJECTS. Round-tripping plain dicts through
# a JSON file silently drops `camera` and `cam_from_rig` (both parse to None),
# leaving the database on the default SIMPLE_RADIAL model -- fatal for a 200-deg
# fisheye: the mapper then fails to initialise and returns zero reconstructions.
K = [0.0303493686, 0.0023128117, -0.0027963710, -0.0003560687]
def mkcam():
    c = pycolmap.Camera.create_from_model_name(
        camera_id=0, model_name="OPENCV_FISHEYE", focal_length=a.f, width=1920, height=1920)
    c.params = [a.f, a.f, 960.0, 960.0, *K]
    return c
q = R2q(R_rig)                                    # [w,x,y,z]; Rotation3d wants xyzw
up_from_down = pycolmap.Rigid3d(pycolmap.Rotation3d([q[1], q[2], q[3], q[0]]),
                                np.zeros(3))      # ~2 cm lens baseline ignored
cfg = [pycolmap.RigConfig(cameras=[
    pycolmap.RigConfigCamera(ref_sensor=True, image_prefix="down/", camera=mkcam()),
    pycolmap.RigConfigCamera(ref_sensor=False, image_prefix="up/", camera=mkcam(),
                             cam_from_rig=up_from_down),
])]
assert cfg[0].cameras[0].camera is not None and cfg[0].cameras[1].cam_from_rig is not None
print(f"[rig-sfm] rig: up_from_down quat(wxyz)={np.round(q,5).tolist()} | "
      f"model={cfg[0].cameras[0].camera.model.name}", flush=True)

# ---- stage 2b: optional feature masks ---------------------------------------
# COLMAP convention: <mask_path>/<image sub-path>.png, black pixels are ignored.
MASKS = ""
if a.mask_up:
    from PIL import Image
    MASKS = os.path.join(a.root, f"masks_{a.tag}"); os.makedirs(MASKS, exist_ok=True)
    yy, xx = np.mgrid[0:1920, 0:1920]
    rr = np.hypot(yy - 959.5, xx - 959.5)
    def r_of(th_deg):                    # OPENCV_FISHEYE forward model
        t = np.radians(th_deg)
        return a.f * t * (1 + K[0]*t**2 + K[1]*t**4 + K[2]*t**6 + K[3]*t**8)
    ring = ((rr >= r_of(80.0)) & (rr <= r_of(100.0))).astype(np.uint8) * 255
    tmpl = {}
    for sub in LENSES:
        d = os.path.join(MASKS, sub); os.makedirs(d, exist_ok=True)
        m = ring if sub == "up" else np.full((1920, 1920), 255, np.uint8)
        p = os.path.join(MASKS, f"_{sub}.png"); Image.fromarray(m).save(p); tmpl[sub] = p
        for k in idxs:
            t = os.path.join(d, f"f_{k:04d}.jpg.png")
            if not os.path.exists(t): os.link(p, t)
    print(f"[rig-sfm] up-lens mask: horizon ring r={r_of(80):.0f}-{r_of(100):.0f}px "
          f"({100*ring.mean()/255:.1f}% of frame kept)", flush=True)

# ---- stage 3: features + matching + rig-constrained mapping ------------------
db = os.path.join(a.root, f"{a.tag}.db")
if not os.path.exists(db):
    ropts = pycolmap.ImageReaderOptions()
    if MASKS: ropts.mask_path = MASKS
    pycolmap.extract_features(db, IMG, camera_mode=pycolmap.CameraMode.PER_FOLDER,
                              reader_options=ropts)
    print("[rig-sfm] features done", flush=True)
    pycolmap.match_sequential(db, pairing_options=pycolmap.SequentialPairingOptions(
        overlap=a.overlap, quadratic_overlap=True))
    print("[rig-sfm] matching done", flush=True)

if len(LENSES) > 1:                    # single-lens control needs no rig config
    _dbh = pycolmap.Database.open(db)  # pycolmap 4.x: no public constructor
    pycolmap.apply_rig_config(cfg, _dbh)
    _dbh.close()
else:                                  # still force the fisheye model + calibration
    import sqlite3 as _s
    _c = _s.connect(db)
    _c.execute("update cameras set model=5, params=?, prior_focal_length=1",
               (np.array([a.f, a.f, 960.0, 960.0, *K], np.float64).tobytes(),))
    _c.commit(); _c.close()
import sqlite3                          # verify the fisheye model actually landed
_m = sqlite3.connect(db).execute("select camera_id, model from cameras").fetchall()
print(f"[rig-sfm] db cameras (id, model_id): {_m}  (5 == OPENCV_FISHEYE)", flush=True)
assert all(m == 5 for _, m in _m), "camera model not applied -- mapper would fail"
print("[rig-sfm] rig config applied to database", flush=True)

os.makedirs(a.out, exist_ok=True)
opts = pycolmap.IncrementalPipelineOptions()
opts.ba_refine_focal_length = True
opts.ba_refine_extra_params = True
opts.ba_refine_principal_point = False
recs = pycolmap.incremental_mapping(db, IMG, a.out, options=opts)
for i, r in recs.items():
    print(f"[rig-sfm] model {i}: {r.num_reg_images()} images / {len(r.frames)} frames, "
          f"{r.num_points3D()} points", flush=True)
print("RIG_SFM_DONE", flush=True)
