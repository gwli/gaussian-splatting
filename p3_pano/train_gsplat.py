#!/usr/bin/env python3
"""T-F2: end-to-end 3DGS training on the gsplat backend (nerfstudio gsplat
1.5.3) with its native DefaultStrategy densification, to validate the 3.42x
micro-bench speedup as a real end-to-end win at comparable quality.

Self-contained: loads a COLMAP/VGGT scene (sparse/0 + images) via colmap_loader
by file path (avoids scene/__init__ -> simple_knn import), inits per-point
scales with a scipy cKDTree kNN, trains with L1+SSIM, holds out every 8th camera
(matching the repo's --eval llffhold=8), and reports held-out PSNR/SSIM + the
training throughput (iter/s) so it can be compared to the INRIA backend.

Usage: train_gsplat.py <scene_dir> [iters=7000] [--no-eval]
  scene_dir has images/ and sparse/0/{cameras,images,points3D}.bin
"""
import sys, os, time, math, json, importlib.util, numpy as np, torch
import torch.nn.functional as F
from PIL import Image

REPO = "/w" if os.path.exists("/w/scene/colmap_loader.py") else "/raid/git/gaussian-splatting"
_spec = importlib.util.spec_from_file_location("colmap_loader", f"{REPO}/scene/colmap_loader.py")
_cl = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_cl)

from gsplat import rasterization, DefaultStrategy

scene_dir = sys.argv[1]
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 7000
DO_EVAL = "--no-eval" not in sys.argv
dev = "cuda"
SH_MAX = 3

# ---------------- load COLMAP scene ----------------
def focal2fov(focal, pixels): return 2 * math.atan(pixels / (2 * focal))

cam_intr = _cl.read_intrinsics_binary(f"{scene_dir}/sparse/0/cameras.bin")
cam_extr = _cl.read_extrinsics_binary(f"{scene_dir}/sparse/0/images.bin")

cams = []
for img_id, ext in sorted(cam_extr.items(), key=lambda kv: kv[1].name):
    intr = cam_intr[ext.camera_id]
    R = _cl.qvec2rotmat(ext.qvec)           # world->cam
    T = ext.tvec.astype(np.float32)
    W, H = intr.width, intr.height
    model = intr.model
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
        fx = fy = intr.params[0]; cx, cy = intr.params[1], intr.params[2]
    elif model in ("PINHOLE", "OPENCV"):
        fx, fy, cx, cy = intr.params[0], intr.params[1], intr.params[2], intr.params[3]
    else:
        fx = fy = intr.params[0]; cx, cy = W / 2, H / 2
    viewmat = np.eye(4, dtype=np.float32)
    viewmat[:3, :3] = R; viewmat[:3, 3] = T
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    cams.append(dict(name=ext.name, viewmat=viewmat, K=K, W=W, H=H,
                     path=f"{scene_dir}/images/{ext.name}"))

# every-8th holdout (llffhold=8) when eval
test = [c for i, c in enumerate(cams) if DO_EVAL and i % 8 == 0]
train = [c for i, c in enumerate(cams) if not (DO_EVAL and i % 8 == 0)]
print(f"[gsplat] {len(cams)} cams -> {len(train)} train / {len(test)} test | iters={ITERS}")

def load_img(c):
    im = Image.open(c["path"]).convert("RGB")
    if im.size != (c["W"], c["H"]): im = im.resize((c["W"], c["H"]))
    return torch.from_numpy(np.asarray(im, np.float32) / 255.0).to(dev)  # (H,W,3)

# ---------------- init gaussians from sparse points ----------------
xyz, rgb, _ = _cl.read_points3D_binary(f"{scene_dir}/sparse/0/points3D.bin") \
    if os.path.exists(f"{scene_dir}/sparse/0/points3D.bin") else (None, None, None)
if xyz is None:
    from plyfile import PlyData
    p = PlyData.read(f"{scene_dir}/sparse/0/points3D.ply")["vertex"]
    xyz = np.stack([p["x"], p["y"], p["z"]], 1).astype(np.float32)
    rgb = np.stack([p["red"], p["green"], p["blue"]], 1).astype(np.float32) / 255.0
else:
    xyz = xyz.astype(np.float32); rgb = (rgb.astype(np.float32) / 255.0)
N = xyz.shape[0]
print(f"[gsplat] init {N} points")

# camera-extent (nerf++ norm) for LR scaling + strategy scene_scale
cam_centers = np.array([(-c["viewmat"][:3, :3].T @ c["viewmat"][:3, 3]) for c in train])
center = cam_centers.mean(0)
extent = float(np.linalg.norm(cam_centers - center, axis=1).max() * 1.1)
print(f"[gsplat] cameras_extent={extent:.3f}")

# init per-point scale = log(mean dist to 3 nearest neighbors)
from scipy.spatial import cKDTree
d, _ = cKDTree(xyz).query(xyz, k=4)
dist2 = np.clip((d[:, 1:] ** 2).mean(1), 1e-8, None)
scales0 = np.log(np.sqrt(dist2))[:, None].repeat(3, 1).astype(np.float32)

def RGB2SH(c): return (c - 0.5) / 0.28209479177387814
sh0 = torch.tensor(RGB2SH(rgb), dtype=torch.float32, device=dev)[:, None, :]   # (N,1,3)
shN = torch.zeros((N, (SH_MAX + 1) ** 2 - 1, 3), dtype=torch.float32, device=dev)

splats = torch.nn.ParameterDict({
    "means":     torch.nn.Parameter(torch.tensor(xyz, device=dev)),
    "scales":    torch.nn.Parameter(torch.tensor(scales0, device=dev)),
    "quats":     torch.nn.Parameter(torch.tensor([1., 0, 0, 0], device=dev).repeat(N, 1)),
    "opacities": torch.nn.Parameter(torch.logit(torch.full((N,), 0.1, device=dev))),
    "sh0":       torch.nn.Parameter(sh0),
    "shN":       torch.nn.Parameter(shN),
}).to(dev)

# INRIA-matched LRs (means scaled by extent)
lrs = {"means": 0.00016 * extent, "scales": 0.005, "quats": 0.001,
       "opacities": 0.05, "sh0": 0.0025, "shN": 0.0025 / 20}
opt = {k: torch.optim.Adam([{"params": splats[k], "lr": lr, "name": k}], eps=1e-15)
       for k, lr in lrs.items()}

strat = DefaultStrategy(verbose=False, refine_stop_iter=int(ITERS * 0.5),
                        reset_every=3000, refine_every=100)
strat.check_sanity(splats, opt)
state = strat.initialize_state(scene_scale=extent)

def ssim(a, b):  # a,b: (1,3,H,W)
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    k = torch.ones(3, 1, 11, 11, device=a.device) / (11 * 11)
    mu_a = F.conv2d(a, k, padding=5, groups=3); mu_b = F.conv2d(b, k, padding=5, groups=3)
    va = F.conv2d(a * a, k, padding=5, groups=3) - mu_a ** 2
    vb = F.conv2d(b * b, k, padding=5, groups=3) - mu_b ** 2
    vab = F.conv2d(a * b, k, padding=5, groups=3) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + C1) * (2 * vab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (va + vb + C2))
    return s.mean()

def render(cam, sh_deg):
    vm = torch.tensor(cam["viewmat"], device=dev)[None]
    K = torch.tensor(cam["K"], device=dev)[None]
    colors = torch.cat([splats["sh0"], splats["shN"]], 1)
    rc, ra, info = rasterization(
        means=splats["means"], quats=splats["quats"], scales=torch.exp(splats["scales"]),
        opacities=torch.sigmoid(splats["opacities"]), colors=colors,
        viewmats=vm, Ks=K, width=cam["W"], height=cam["H"],
        sh_degree=sh_deg, render_mode="RGB", packed=False, absgrad=strat.absgrad)
    return rc[0].clamp(0, 1), info   # (H,W,3)

# ---------------- train ----------------
torch.manual_seed(0)
order = list(range(len(train)))
t0 = time.time(); ema = None
for step in range(ITERS):
    sh_deg = min(SH_MAX, step // (ITERS // (SH_MAX + 1) + 1))
    cam = train[order[step % len(train)]]
    if step % len(train) == 0:
        import random; random.Random(step).shuffle(order)
    gt = load_img(cam)
    out, info = render(cam, sh_deg)
    strat.step_pre_backward(params=splats, optimizers=opt, state=state, step=step, info=info)
    pred_c = out.permute(2, 0, 1)[None]; gt_c = gt.permute(2, 0, 1)[None]
    l1 = (out - gt).abs().mean()
    loss = 0.8 * l1 + 0.2 * (1.0 - ssim(pred_c, gt_c))
    loss.backward()
    for o in opt.values(): o.step(); o.zero_grad(set_to_none=True)
    strat.step_post_backward(params=splats, optimizers=opt, state=state, step=step,
                             info=info, packed=False)
    ema = loss.item() if ema is None else 0.9 * ema + 0.1 * loss.item()
    if step % 500 == 0 or step == ITERS - 1:
        print(f"  it {step:5d}  loss {ema:.4f}  N={splats['means'].shape[0]}  "
              f"{(step+1)/(time.time()-t0):.1f} it/s")
torch.cuda.synchronize()
dt = time.time() - t0
ips = ITERS / dt
print(f"[gsplat] trained {ITERS} it in {dt:.1f}s = {ips:.1f} it/s | final N={splats['means'].shape[0]}")

# ---------------- eval ----------------
res = {"backend": "gsplat", "iters": ITERS, "iter_s": round(ips, 1),
       "n_gaussians": int(splats["means"].shape[0]), "train_s": round(dt, 1)}
if test:
    psnrs, ssims = [], []
    with torch.no_grad():
        for c in test:
            gt = load_img(c); out, _ = render(c, SH_MAX)
            mse = ((out - gt) ** 2).mean()
            psnrs.append(float(-10 * torch.log10(mse)))
            ssims.append(float(ssim(out.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None])))
    res["psnr"] = round(float(np.mean(psnrs)), 2)
    res["ssim"] = round(float(np.mean(ssims)), 3)
    print(f"[gsplat] HELD-OUT: PSNR {res['psnr']}  SSIM {res['ssim']}  over {len(test)} cams")
outp = f"{scene_dir}/gsplat_result.json"
json.dump(res, open(outp, "w"), indent=2)
print(f"[gsplat] wrote {outp}: {res}")
