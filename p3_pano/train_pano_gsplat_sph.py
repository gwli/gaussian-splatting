#!/usr/bin/env python3
"""T-F7: direct equirect 3DGS training on the SPHERICAL gsplat rasterizer
(gsplat_equirect.render_equirect): one equirect pass with gsplat's fast tile
compositor. Compare vs LONLAT (native OmniGS) and gsplat-cubemap (T-F6).

Usage: train_pano_gsplat_sph.py <pano_cams.json> <out_dir> [iters=7000] [width=1024]
"""
import sys, os, json, math, random, time, numpy as np, torch
import torch.nn.functional as F
from PIL import Image

REPO = "/w" if os.path.exists("/w/scene/colmap_loader.py") else "/raid/git/gaussian-splatting"
sys.path.insert(0, REPO + "/p3_pano")
from gsplat import DefaultStrategy
from gsplat_equirect import render_equirect, render_equirect_fused
_FUSED = os.environ.get("GSPLAT_EQUIRECT_FUSED", "1") == "1"
_render_fn = render_equirect_fused if _FUSED else render_equirect
print(f"[pano-gsplat-sph] backend = {'FUSED CUDA (T-F8)' if _FUSED else 'hybrid PyTorch-proj (T-F7)'}")

cams_json, out_dir = sys.argv[1], sys.argv[2]
ITERS = int(sys.argv[3]) if len(sys.argv) > 3 else 7000
W = int(sys.argv[4]) if len(sys.argv) > 4 else 1024
H = W // 2
os.makedirs(out_dir, exist_ok=True)
dev = "cuda"; SH_MAX = 3
meta = json.load(open(cams_json))

# init gaussians from VGGT points3D.ply, or resume a trained model (INIT_PLY)
from plyfile import PlyData
INIT_PLY = os.environ.get("INIT_PLY", "")     # resume: full gaussian state from a ply
BLOCK_JSON = os.environ.get("BLOCK_JSON", "")   # blocks.json from block_split.py
BLOCK_ID = int(os.environ.get("BLOCK_ID", "-1"))
BLOCK_MARGIN = float(os.environ.get("BLOCK_MARGIN", "0.15"))
v = PlyData.read(INIT_PLY if INIT_PLY else meta["point_cloud"])["vertex"].data
_bi = None
if INIT_PLY and BLOCK_JSON and BLOCK_ID >= 0:
    _bi = json.load(open(BLOCK_JSON))
    _ax = np.array(_bi["axis"]); _bc = np.array(_bi["ctr"]); _ed = np.array(_bi["edges"])
    _xyz = np.stack([v["x"], v["y"], v["z"]], 1)
    _t = (_xyz - _bc) @ _ax
    _sp = (_ed[-1] - _ed[0]) / (len(_ed) - 1)
    _lo = _ed[BLOCK_ID] - BLOCK_MARGIN * _sp
    _hi = _ed[BLOCK_ID + 1] + BLOCK_MARGIN * _sp
    # the sky shell sits at ~5x the scene radius; anything past 2x the full-scene camera
    # radius is sky and is shared by every block (frozen, not re-optimised).
    _far = np.linalg.norm(_xyz - _bc, axis=1) > 2.0 * _bi["cam_radius"]
    _keep = ((_t >= _lo) & (_t <= _hi)) | _far          # own slab + the shared sky shell
    v = v[_keep]
    print(f"[block {BLOCK_ID}] kept {int(_keep.sum())}/{len(_keep)} gaussians "
          f"(slab {_lo:.0f}..{_hi:.0f} along axis, +{int(_far.sum())} far-field)")
xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
rgb = (np.stack([v["red"], v["green"], v["blue"]], 1).astype(np.float32) / 255.0
       if "red" in v.dtype.names else np.full_like(xyz, 0.5))
N = xyz.shape[0]
# Sky sphere (as in GGPS/PanoLOG create_from_pcd): a shell of far-field gaussians so
# the sky is not represented by stretched near-field ones. Uniform on a sphere of
# radius 5x the scene radius, pale blue-white, 10x the kNN scale. Placed first, and
# (matching their coarse stage) optimised normally rather than locked.
SKYBOX_NUM = int(os.environ.get("SKYBOX_NUM", "0"))
SKYBOX_MULT = float(os.environ.get("SKYBOX_MULT", "5.0"))
if SKYBOX_NUM > 0:
    ctr = xyz.mean(0)
    radius = float(np.linalg.norm(xyz - ctr, axis=1).max())
    rng = np.random.default_rng(0)
    th = 2 * np.pi * rng.random(SKYBOX_NUM)
    ph = np.arccos(1.0 - 2.0 * rng.random(SKYBOX_NUM))
    sky = np.stack([np.cos(th) * np.sin(ph), np.sin(th) * np.sin(ph), np.cos(ph)], 1)
    sky = (sky * radius * SKYBOX_MULT + ctr).astype(np.float32)
    sky_rgb = np.tile(np.array([[0.7, 0.8, 0.95]], np.float32), (SKYBOX_NUM, 1))
    xyz = np.concatenate([sky, xyz]); rgb = np.concatenate([sky_rgb, rgb])
    N = xyz.shape[0]
    print(f"[skybox] {SKYBOX_NUM} sky gaussians at r={radius*SKYBOX_MULT:.1f} "
          f"({SKYBOX_MULT}x scene radius {radius:.1f}); total init {N}")

from scipy.spatial import cKDTree
dd, _ = cKDTree(xyz).query(xyz, k=4)
_d2 = np.clip((dd[:, 1:] ** 2).mean(1), 1e-8, None)
if SKYBOX_NUM > 0:
    _d2[:SKYBOX_NUM] *= 10.0                      # their skybox scale boost
    _d2[SKYBOX_NUM:] = np.minimum(_d2[SKYBOX_NUM:], 10.0)
scales0 = np.log(np.sqrt(_d2))[:, None].repeat(3, 1).astype(np.float32)
def RGB2SH(c): return (c - 0.5) / 0.28209479177387814
sh0 = torch.tensor(RGB2SH(rgb), dtype=torch.float32, device=dev)[:, None, :]
shN = torch.zeros((N, (SH_MAX + 1) ** 2 - 1, 3), dtype=torch.float32, device=dev)
extent = float(meta["cameras_extent"])
print(f"[pano-gsplat-sph] init {N} pts | extent {extent:.3f} | equirect {W}x{H}")

splats = torch.nn.ParameterDict({
    "means":     torch.nn.Parameter(torch.tensor(xyz, device=dev)),
    "scales":    torch.nn.Parameter(torch.tensor(scales0, device=dev)),
    "quats":     torch.nn.Parameter(torch.tensor([1., 0, 0, 0], device=dev).repeat(N, 1)),
    "opacities": torch.nn.Parameter(torch.logit(torch.full((N,), 0.1, device=dev))),
    "sh0":       torch.nn.Parameter(sh0),
    "shN":       torch.nn.Parameter(shN),
}).to(dev)
if INIT_PLY:                                    # overwrite with the trained state
    _nr = len([n for n in v.dtype.names if n.startswith("f_rest_")])
    _fr = np.stack([v[f"f_rest_{i}"] for i in range(_nr)], 1).astype(np.float32)
    with torch.no_grad():
        splats["sh0"].copy_(torch.tensor(np.stack([v[f"f_dc_{i}"] for i in range(3)], 1),
                                         dtype=torch.float32, device=dev)[:, None, :])
        splats["shN"].copy_(torch.tensor(_fr.reshape(len(_fr), 3, _nr // 3).transpose(0, 2, 1).copy(),
                                         device=dev))
        splats["opacities"].copy_(torch.tensor(np.asarray(v["opacity"], np.float32), device=dev))
        splats["scales"].copy_(torch.tensor(np.stack([v[f"scale_{i}"] for i in range(3)], 1),
                                            dtype=torch.float32, device=dev))
        splats["quats"].copy_(torch.tensor(np.stack([v[f"rot_{i}"] for i in range(4)], 1),
                                           dtype=torch.float32, device=dev))
    print(f"[resume] loaded {N} gaussians from {INIT_PLY}")

# SKY_FREEZE_R: freeze gaussians farther than R x scene radius from the camera centroid
# (GGPS locks the sky shell during block refinement). Selected geometrically, not by
# index, so it survives the densifier reordering entries.
SKY_FREEZE_R = float(os.environ.get("SKY_FREEZE_R", "0"))

_LRS = float(os.environ.get("LR_SCALE_POS", "1.0"))     # their c4: 0.000064/0.00016 = 0.4
lrs = {"means": 0.00016 * extent * _LRS, "scales": 0.005 * (0.4 if _LRS < 1 else 1.0),
       "quats": 0.001, "opacities": 0.05, "sh0": 0.0025, "shN": 0.0025 / 20}
opt = {k: torch.optim.Adam([{"params": splats[k], "lr": lr}], eps=1e-15) for k, lr in lrs.items()}
# resume the Adam moments saved next to INIT_PLY, so a fine-tune continues the optimiser
# instead of restarting it from zero momentum. Block runs subset the moments with the
# same mask as the gaussians, keeping state and parameters aligned.
_ck = os.path.join(os.path.dirname(INIT_PLY), "optim.pt") if INIT_PLY else ""
if os.environ.get("COLD_OPTIM", "0") == "1": _ck = ""   # ablation: force a cold optimiser
if _ck and os.path.exists(_ck):
    _sd = torch.load(_ck, map_location=dev, weights_only=False)
    _msk = torch.tensor(_keep, device=dev) if (BLOCK_JSON and BLOCK_ID >= 0) else None
    for k, o in opt.items():
        st = _sd["opt"][k]
        if _msk is not None:
            for s in st["state"].values():
                for m in ("exp_avg", "exp_avg_sq"):
                    if m in s: s[m] = s[m][_msk]
        o.load_state_dict(st)
        for g in o.param_groups: g["lr"] = lrs[k]      # keep this run's lr, not the saved one
    print(f"[resume] restored Adam moments from {_ck}"
          + (f" (subset to {int(_keep.sum())} gaussians)" if _msk is not None else ""))
elif INIT_PLY:
    print(f"[resume] no optim.pt next to INIT_PLY -- optimiser starts cold "
          f"(measured neutral, see task_ft.md 40)")
# densification knobs (env): lower GROW_GRAD2D -> grow more gaussians; higher
# REFINE_STOP_FRAC -> keep densifying longer. Defaults = gsplat DefaultStrategy.
_GG = float(os.environ.get("GROW_GRAD2D", "0.0002"))
_RSF = float(os.environ.get("REFINE_STOP_FRAC", "0.5"))
MASK_DIR = os.environ.get("MASK_DIR", "")  # per-pano weight masks (dynamic-content downweight)
# UniK3D depth priors (gen_depth.py). GGPS enables depth only in the refine stage,
# annealing the weight down; DEPTH_W/DEPTH_W_END mirror their 1.0 -> 0.001.
DEPTH_DIR = os.environ.get("DEPTH_DIR", "")
DEPTH_W = float(os.environ.get("DEPTH_W", "0.05"))
DEPTH_W_END = float(os.environ.get("DEPTH_W_END", "0.001"))
# The prior disagrees with the model by a scene-dependent amount (log residual 0.81 on
# 027 vs 1.51 on 021), so a fixed DEPTH_W is not a fixed strength: the per-scene optima
# spread 2.7x (0.15/0.15/0.4) while weight x residual only spreads 1.7x. DEPTH_NORM
# targets a constant *effective* strength by dividing by the step-0 residual, which is
# what makes a single setting usable across scenes we cannot sweep individually.
DEPTH_NORM = float(os.environ.get("DEPTH_NORM", "0"))
_dnorm = None
GT_ON_GPU = os.environ.get("GT_ON_GPU", "0") == "1"   # legacy behaviour; off = CPU cache
ABSGRAD = os.environ.get("ABSGRAD", "0") == "1"

def gt_of(cam):                            # uint8 (H,W,3) -> float (3,H,W) on device
    return cam["gt"].to(dev, non_blocking=True).permute(2, 0, 1).float() / 255.0

_RESET = int(os.environ.get("RESET_EVERY", "3000"))
strat = DefaultStrategy(verbose=False, refine_stop_iter=int(ITERS * _RSF),
                        reset_every=_RESET, refine_every=100, grow_grad2d=_GG,
                        absgrad=ABSGRAD)
print(f"[pano-gsplat-sph] densify: grow_grad2d={_GG} refine_stop={int(ITERS*_RSF)}")
strat.check_sanity(splats, opt); state = strat.initialize_state(scene_scale=extent)

def load_cam(c, i):
    R = np.array(c["R_wp"], np.float32); T = np.array(c["T"], np.float32)
    vm = np.eye(4, dtype=np.float32); vm[:3, :3] = R; vm[:3, 3] = T
    im = Image.open(c["image"]).convert("RGB").resize((W, H), Image.LANCZOS)
    # GT lives on the CPU as uint8 and is uploaded per iteration. Keeping every
    # frame on the GPU as float32 was what capped training at 1024x512:
    # 390 views at 4096x2048 would need 39 GB (0.6 GB as uint8 on the host).
    gt = torch.from_numpy(np.asarray(im).copy())                     # (H,W,3) uint8, cpu
    if GT_ON_GPU: gt = gt.to(dev)
    wmask = None
    if MASK_DIR:
        mp = os.path.join(MASK_DIR, f"pano_{c['idx']:04d}.png")
        if os.path.exists(mp):
            wm = Image.open(mp).convert("L").resize((W, H), Image.BILINEAR)
            wmask = torch.tensor(np.asarray(wm), dtype=torch.float32, device=dev)[None] / 255.0  # (1,H,W)
    dprior = None
    if DEPTH_DIR:
        dp = os.path.join(DEPTH_DIR, f"{c['idx']:05d}.npy")
        if os.path.exists(dp):     # (2,h,w) = radial metres + confidence, kept at 1024x512
            dprior = torch.tensor(np.load(dp).astype(np.float32), device=dev)
    return {"vm": torch.tensor(vm, device=dev), "C": torch.tensor(np.array(c["C"], np.float32), device=dev),
            "R": torch.tensor(R, device=dev), "i": i, "wmask": wmask, "dprior": dprior,
            "idx_ds": int(c["idx"]),      # dataset frame index, for pose-delta export
            "gt": gt, "name": f"pano_{c['idx']:04d}"}

cams = [load_cam(c, i) for i, c in enumerate(meta["cameras"])]
# TEST_IDX_FILE: explicit held-out pano indices (one per line). Required for fair
# cross-density comparison -- the default cams[::8] makes test views land closer
# to train views as density grows (leak), which flatters denser runs.
_tif = os.environ.get("TEST_IDX_FILE", "")
if _tif:
    hold = {int(l) for l in open(_tif) if l.strip()}
    test = [c for c, m in zip(cams, meta["cameras"]) if m["idx"] in hold]
    train = [c for c, m in zip(cams, meta["cameras"]) if m["idx"] not in hold]
    print(f"[split] held-out from {_tif}: {len(test)} test / {len(train)} train")
else:
    test = cams[::8]; train = [c for i, c in enumerate(cams) if i % 8 != 0]
print(f"[pano-gsplat-sph] {len(cams)} cams -> {len(train)} train / {len(test)} test")

_cc = torch.stack([c["C"] for c in cams])           # camera centroid + radius, for SKY_FREEZE_R
_ctr, _rs = _cc.mean(0), float((_cc - _cc.mean(0)).norm(dim=1).max())
if _bi is not None:      # a block sees only its own cameras; the sky shell is global
    _ctr = torch.tensor(_bi["ctr"], dtype=_cc.dtype, device=_cc.device)
    _rs = float(_bi["cam_radius"])
if SKY_FREEZE_R > 0:
    print(f"[sky-lock] freezing gaussians beyond {SKY_FREEZE_R}x scene radius ({SKY_FREEZE_R*_rs:.0f} m)")

def freeze_far_grads():
    """Zero the gradients of far-field (sky-shell) gaussians so refinement leaves
    them untouched -- the analogue of GGPS's skybox_locked in the block stage.
    Caveat: the densifier can still split them; only the parameters are locked."""
    if SKY_FREEZE_R <= 0: return
    with torch.no_grad():
        far = (splats["means"].detach() - _ctr).norm(dim=1) > SKY_FREEZE_R * _rs
        if far.any():
            for k, prm in splats.items():
                if prm.grad is not None: prm.grad[far] = 0

# optional in-training pose refinement (BA): POSE_OPT=1 -> per-camera so3
# rotation delta (pano frame, left-mul) + metric translation delta on C.
# EXP_OPT: per-image colour affine (3-channel gain + bias) on the RENDER, trained only.
# The drone runs auto-exposure, and the held-out diagnostic shows a per-frame photometric
# residual worth +0.34 dB (027) to +0.80 dB (021) that is additive with the pose residual,
# i.e. a genuinely separate error. Without this term the map has to absorb AE drift into
# geometry and colour. Evaluation uses the identity transform, so no test image is touched.
# intermediate checkpoints, for monitoring only -- see the caveat at the call site
SAVE_AT = {int(x) for x in os.environ.get("SAVE_AT", "").split(",") if x.strip()}
EXP_OPT = os.environ.get("EXP_OPT", "0") == "1"
POSE_OPT = os.environ.get("POSE_OPT", "0") == "1"
POSE_START = int(os.environ.get("POSE_START", "500"))
pose_dr = torch.zeros(len(cams), 3, device=dev, requires_grad=POSE_OPT)
pose_dt = torch.zeros(len(cams), 3, device=dev, requires_grad=POSE_OPT)
pose_opt = torch.optim.Adam([
    {"params": [pose_dr], "lr": float(os.environ.get("POSE_LR_R", "1e-3"))},
    {"params": [pose_dt], "lr": float(os.environ.get("POSE_LR_T", "3e-2"))}]) if POSE_OPT else None
if POSE_OPT:
    print(f"[pose-opt] ON start={POSE_START} lr_r={os.environ.get('POSE_LR_R','1e-3')} lr_t={os.environ.get('POSE_LR_T','3e-2')}")
exp_g = torch.ones(len(cams), 3, device=dev, requires_grad=EXP_OPT)
exp_b = torch.zeros(len(cams), 3, device=dev, requires_grad=EXP_OPT)
exp_opt = torch.optim.Adam([{"params": [exp_g, exp_b],
                             "lr": float(os.environ.get("EXP_LR", "3e-3"))}]) if EXP_OPT else None
if EXP_OPT:
    print(f"[exp-opt] ON lr={os.environ.get('EXP_LR','3e-3')} (train-only; eval uses identity)")

def so3exp(w):
    th = w.norm() + 1e-12; k = w / th
    z = torch.zeros((), device=w.device)
    K = torch.stack([torch.stack([z, -k[2], k[1]]),
                     torch.stack([k[2], z, -k[0]]),
                     torch.stack([-k[1], k[0], z])])
    return torch.eye(3, device=w.device) + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)

def render(cam, sh_deg, use_pose=True):
    colors = torch.cat([splats["sh0"], splats["shN"]], 1)
    # pose deltas and the depth channel are independent choices -- branching on them
    # separately meant POSE_OPT silently skipped info["depth"] and the depth loss died.
    if POSE_OPT and use_pose:
        i = cam["i"]
        Rp = so3exp(pose_dr[i]) @ cam["R"]
        Cp = cam["C"] + pose_dt[i]
        vm = torch.cat([torch.cat([Rp, -(Rp @ Cp)[:, None]], 1),
                        torch.tensor([[0, 0, 0, 1.0]], device=dev)], 0)
    else:
        vm, Cp = cam["vm"], cam["C"]
    out = _render_fn(splats["means"], splats["quats"], torch.exp(splats["scales"]),
                     torch.sigmoid(splats["opacities"]), colors, vm, Cp, W, H, sh_deg,
                     absgrad=ABSGRAD, **({"with_depth": True} if DEPTH_DIR else {}))
    if DEPTH_DIR:
        img, dep, info = out
        info["depth"] = dep            # (H,W) expected radial range, for the depth prior
    else:
        img, info = out
    return img.permute(2, 0, 1).clamp(0, 1), info     # (3,H,W)

def ssim(a, b):
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    k = torch.ones(3, 1, 11, 11, device=a.device) / 121.0
    ma = F.conv2d(a, k, padding=5, groups=3); mb = F.conv2d(b, k, padding=5, groups=3)
    va = F.conv2d(a * a, k, padding=5, groups=3) - ma ** 2
    vb = F.conv2d(b * b, k, padding=5, groups=3) - mb ** 2
    vab = F.conv2d(a * b, k, padding=5, groups=3) - ma * mb
    return (((2 * ma * mb + C1) * (2 * vab + C2)) / ((ma ** 2 + mb ** 2 + C1) * (va + vb + C2))).mean()

# save INRIA-format ply (for ksplat / viewers) — matches scene.gaussian_model.save_ply
from plyfile import PlyData as _PD, PlyElement as _PE
def save_state(it):
  with torch.no_grad():
      xyz = splats["means"].detach().cpu().numpy()
      f_dc = splats["sh0"].detach().transpose(1, 2).flatten(1).cpu().numpy()    # (N,3)
      f_rest = splats["shN"].detach().transpose(1, 2).flatten(1).cpu().numpy()  # (N,45)
      opac = splats["opacities"].detach().cpu().numpy().reshape(-1, 1)
      scal = splats["scales"].detach().cpu().numpy()
      rot = splats["quats"].detach().cpu().numpy()
      Ng = xyz.shape[0]
      fields = (["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(3)] +
                [f"f_rest_{i}" for i in range(f_rest.shape[1])] + ["opacity"] +
                [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)])
      arr = np.concatenate([xyz, np.zeros((Ng, 3), np.float32), f_dc, f_rest, opac, scal, rot], 1).astype(np.float32)
      elems = np.empty(Ng, dtype=[(f, "f4") for f in fields])
      for i, f in enumerate(fields):
          elems[f] = arr[:, i]
      pc_dir = os.path.join(out_dir, f"point_cloud/iteration_{it}")
      os.makedirs(pc_dir, exist_ok=True)
      _PD([_PE.describe(elems, "vertex")]).write(os.path.join(pc_dir, "point_cloud.ply"))
      print(f"[PLY] saved {Ng} gaussians -> {pc_dir}/point_cloud.ply")
      # Adam moments alongside the ply. Without these a "fine-tune" restarts the optimiser
      # from zero momentum, which measurably damages the model (-0.87 dB, see task_ft.md 39)
      # and gets misread as the fine-tuning method failing.
      torch.save({"n": Ng, "opt": {k: o.state_dict() for k, o in opt.items()}},
                 os.path.join(pc_dir, "optim.pt"))
      print(f"[CKPT] saved optimiser state -> {pc_dir}/optim.pt")
      if POSE_OPT:
          # Held-out cameras never receive a gradient, so their deltas stay zero while the
          # map follows the corrected training poses. Persisting the deltas (with each
          # camera's dataset idx and whether it was trained) lets apply_pose_deltas.py
          # interpolate corrections onto the held-out cameras afterwards.
          _tr = {id(c) for c in train}
          np.savez(os.path.join(pc_dir, "poses.npz"),
                   idx=np.array([c["idx_ds"] for c in cams], np.int64),
                   trained=np.array([id(c) in _tr for c in cams], bool),
                   dr=pose_dr.detach().cpu().numpy(), dt=pose_dt.detach().cpu().numpy())
          _dn = pose_dt.detach().norm(dim=1)
          print(f"[POSE] saved deltas -> {pc_dir}/poses.npz  "
                f"|dt| med {float(_dn.median()):.3f} m max {float(_dn.max()):.3f} m", flush=True)


torch.manual_seed(0); stack = []; t0 = time.time(); ema = None
for step in range(ITERS):
    sh_deg = min(SH_MAX, step // (ITERS // (SH_MAX + 1) + 1))
    if not stack: stack = train.copy(); random.Random(step).shuffle(stack)
    cam = stack.pop()
    img, info = render(cam, sh_deg)
    strat.step_pre_backward(params=splats, optimizers=opt, state=state, step=step, info=info)
    gt = gt_of(cam)
    if EXP_OPT:      # applied to the render, so the map is not asked to explain AE drift
        _i = cam["i"]
        img = (img * exp_g[_i][:, None, None] + exp_b[_i][:, None, None]).clamp(0, 1)
    if cam.get("wmask") is not None:  # dynamic-content downweight (0.1 floor keeps geometry grounded)
        wt = cam["wmask"].clamp(min=0.1)
        loss = 0.8 * ((img - gt).abs() * wt).sum() / (wt.sum() * 3) + 0.2 * (1.0 - ssim((img * wt)[None], (gt * wt)[None]))
    else:
        loss = 0.8 * (img - gt).abs().mean() + 0.2 * (1.0 - ssim(img[None], gt[None]))
    if DEPTH_DIR and DEPTH_W > 0:
        # UniK3D prior. Monocular range on aerial panoramas is not trustworthy in
        # absolute scale, so both maps are compared in log space after removing a
        # per-image scale (the mean log offset) -- this supervises relative geometry
        # only. Weighted by UniK3D's own confidence; weight decays 1.0 -> DEPTH_W_END
        # over training, as in their c4 config.
        dp = cam.get("dprior")
        if dp is not None:
            rend = info["depth"][None, None]
            rend = F.interpolate(rend, size=dp.shape[-2:], mode="bilinear",
                                 align_corners=False)[0, 0]
            m = (dp[0] > 0) & (rend > 0)
            if m.any():
                lr_, lp_ = rend[m].clamp_min(1e-3).log(), dp[0][m].log()
                cw = dp[1][m]
                dloss = (((lr_ - lr_.mean()) - (lp_ - lp_.mean())).abs()
                         * cw).sum() / cw.sum().clamp_min(1e-6)
                if DEPTH_NORM > 0 and _dnorm is None:
                    _dnorm = DEPTH_NORM / max(float(dloss), 1e-6)   # constant strength
                w0 = _dnorm if _dnorm is not None else DEPTH_W
                w_now = w0 * (DEPTH_W_END / w0) ** (step / max(ITERS - 1, 1))
                if step == 0:      # prove the term is live, not silently zero
                    print(f"[depth] {int(m.sum())} valid px, log-residual {float(dloss):.4f}, "
                          f"w {w_now:.4f} -> photometric {float(loss):.4f}", flush=True)
                loss = loss + w_now * dloss
    if POSE_OPT:  # anchor deltas to the GPS/IMU prior (sigma_r~2deg, sigma_t~2m)
        i = cam["i"]
        loss = loss + float(os.environ.get("POSE_REG", "0.01")) * (
            (pose_dr[i] / 0.035).square().sum() + (pose_dt[i] / 2.0).square().sum())
    loss.backward()
    freeze_far_grads()
    for o in opt.values(): o.step(); o.zero_grad(set_to_none=True)
    if EXP_OPT:
        exp_opt.step(); exp_opt.zero_grad(set_to_none=True)
        with torch.no_grad():   # gauge: the mean exposure stays identity, so the whole
            _ti = torch.tensor([c["i"] for c in train], device=dev)   # map cannot drift
            exp_g.data[_ti] /= exp_g.data[_ti].mean(0).clamp_min(1e-6)
            exp_b.data[_ti] -= exp_b.data[_ti].mean(0)
    if POSE_OPT:
        if step >= POSE_START: pose_opt.step()
        pose_opt.zero_grad(set_to_none=True)
        if os.environ.get("POSE_GAUGE", "0") == "1" and step >= POSE_START:
            with torch.no_grad():  # gauge fix: zero-mean deltas over train cams
                ti = torch.tensor([c["i"] for c in train], device=dev)
                pose_dt.data[ti] -= pose_dt.data[ti].mean(0)
                pose_dr.data[ti] -= pose_dr.data[ti].mean(0)
    strat.step_post_backward(params=splats, optimizers=opt, state=state, step=step, info=info, packed=False)
    ema = loss.item() if ema is None else 0.9 * ema + 0.1 * loss.item()
    if (step + 1) in SAVE_AT:
        # NOT equivalent to a run of this length: the depth-weight anneal and the SH
        # ramp are indexed by step/ITERS, so a 7k checkpoint of a 30k run is a different
        # model than a 7k run (20.306 vs 23.664 on 021). Use these for monitoring, and
        # run each length separately when comparing lengths.
        save_state(step + 1)
    if step % 500 == 0 or step == ITERS - 1:
        print(f"  it {step:5d}  loss {ema:.4f}  N={splats['means'].shape[0]}  "
              f"{(step+1)/(time.time()-t0):.1f} it/s", flush=True)
torch.cuda.synchronize(); dt = time.time() - t0

save_state(ITERS)

def psnr(a, b): return float(-10 * torch.log10(((a - b) ** 2).mean()))
try:
    sys.path.insert(0, REPO); from lpipsPyTorch import lpips as _lpips; HAVE = True
except Exception:
    HAVE = False
ps, ss, lp = [], [], []
with torch.no_grad():
    for cam in test:
        img, _ = render(cam, SH_MAX); gt = gt_of(cam)
        ps.append(psnr(img, gt)); ss.append(float(ssim(img[None], gt[None])))
        if HAVE: lp.append(float(_lpips(img[None], gt[None], net_type="vgg")))
res = {"scene": os.path.basename(os.path.dirname(out_dir)) or out_dir,
       "method": "direct-pano-gsplat-sphere", "backend": "gsplat-equirect", "iterations": ITERS,
       "train_res": [W, H], "n_gaussians": int(splats["means"].shape[0]), "n_train": len(train),
       "n_test": len(test), "iter_s": round(ITERS / dt, 1), "train_s": round(dt, 1),
       "PSNR": round(float(np.mean(ps)), 3), "SSIM": round(float(np.mean(ss)), 4),
       "LPIPS": (round(float(np.mean(lp)), 4) if lp else None)}
json.dump(res, open(os.path.join(out_dir, "results.json"), "w"), indent=1)
print(f"[EVAL] PSNR={res['PSNR']} SSIM={res['SSIM']} LPIPS={res['LPIPS']} | {res['iter_s']} it/s, {res['train_s']}s")
print(f"[DONE] {res}")
