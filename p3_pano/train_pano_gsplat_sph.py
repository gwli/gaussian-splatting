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

# init gaussians from VGGT points3D.ply
from plyfile import PlyData
v = PlyData.read(meta["point_cloud"])["vertex"]
xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
rgb = (np.stack([v["red"], v["green"], v["blue"]], 1).astype(np.float32) / 255.0
       if "red" in v.data.dtype.names else np.full_like(xyz, 0.5))
N = xyz.shape[0]
from scipy.spatial import cKDTree
dd, _ = cKDTree(xyz).query(xyz, k=4)
scales0 = np.log(np.sqrt(np.clip((dd[:, 1:] ** 2).mean(1), 1e-8, None)))[:, None].repeat(3, 1).astype(np.float32)
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
lrs = {"means": 0.00016 * extent, "scales": 0.005, "quats": 0.001,
       "opacities": 0.05, "sh0": 0.0025, "shN": 0.0025 / 20}
opt = {k: torch.optim.Adam([{"params": splats[k], "lr": lr}], eps=1e-15) for k, lr in lrs.items()}
strat = DefaultStrategy(verbose=False, refine_stop_iter=int(ITERS * 0.5), reset_every=3000, refine_every=100)
strat.check_sanity(splats, opt); state = strat.initialize_state(scene_scale=extent)

def load_cam(c):
    R = np.array(c["R_wp"], np.float32); T = np.array(c["T"], np.float32)
    vm = np.eye(4, dtype=np.float32); vm[:3, :3] = R; vm[:3, 3] = T
    im = Image.open(c["image"]).convert("RGB").resize((W, H), Image.LANCZOS)
    gt = torch.tensor(np.asarray(im), dtype=torch.float32, device=dev).permute(2, 0, 1) / 255.0
    return {"vm": torch.tensor(vm, device=dev), "C": torch.tensor(np.array(c["C"], np.float32), device=dev),
            "gt": gt, "name": f"pano_{c['idx']:04d}"}

cams = [load_cam(c) for c in meta["cameras"]]
test = cams[::8]; train = [c for i, c in enumerate(cams) if i % 8 != 0]
print(f"[pano-gsplat-sph] {len(cams)} cams -> {len(train)} train / {len(test)} test")

def render(cam, sh_deg):
    colors = torch.cat([splats["sh0"], splats["shN"]], 1)
    img, info = _render_fn(splats["means"], splats["quats"], torch.exp(splats["scales"]),
                           torch.sigmoid(splats["opacities"]), colors, cam["vm"], cam["C"],
                           W, H, sh_deg)
    return img.permute(2, 0, 1).clamp(0, 1), info     # (3,H,W)

def ssim(a, b):
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    k = torch.ones(3, 1, 11, 11, device=a.device) / 121.0
    ma = F.conv2d(a, k, padding=5, groups=3); mb = F.conv2d(b, k, padding=5, groups=3)
    va = F.conv2d(a * a, k, padding=5, groups=3) - ma ** 2
    vb = F.conv2d(b * b, k, padding=5, groups=3) - mb ** 2
    vab = F.conv2d(a * b, k, padding=5, groups=3) - ma * mb
    return (((2 * ma * mb + C1) * (2 * vab + C2)) / ((ma ** 2 + mb ** 2 + C1) * (va + vb + C2))).mean()

torch.manual_seed(0); stack = []; t0 = time.time(); ema = None
for step in range(ITERS):
    sh_deg = min(SH_MAX, step // (ITERS // (SH_MAX + 1) + 1))
    if not stack: stack = train.copy(); random.Random(step).shuffle(stack)
    cam = stack.pop()
    img, info = render(cam, sh_deg)
    strat.step_pre_backward(params=splats, optimizers=opt, state=state, step=step, info=info)
    gt = cam["gt"]
    loss = 0.8 * (img - gt).abs().mean() + 0.2 * (1.0 - ssim(img[None], gt[None]))
    loss.backward()
    for o in opt.values(): o.step(); o.zero_grad(set_to_none=True)
    strat.step_post_backward(params=splats, optimizers=opt, state=state, step=step, info=info, packed=False)
    ema = loss.item() if ema is None else 0.9 * ema + 0.1 * loss.item()
    if step % 500 == 0 or step == ITERS - 1:
        print(f"  it {step:5d}  loss {ema:.4f}  N={splats['means'].shape[0]}  "
              f"{(step+1)/(time.time()-t0):.1f} it/s", flush=True)
torch.cuda.synchronize(); dt = time.time() - t0

def psnr(a, b): return float(-10 * torch.log10(((a - b) ** 2).mean()))
try:
    sys.path.insert(0, REPO); from lpipsPyTorch import lpips as _lpips; HAVE = True
except Exception:
    HAVE = False
ps, ss, lp = [], [], []
with torch.no_grad():
    for cam in test:
        img, _ = render(cam, SH_MAX); gt = cam["gt"]
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
