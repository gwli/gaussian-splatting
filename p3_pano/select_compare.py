#!/usr/bin/env python3
"""Frame-selection 3-way comparison (same budget, same fixed test set).

One dense candidate pool (poses from a single VGGT) -> three strategies each pick
the SAME number of TRAIN panos, all evaluated on the SAME held-out TEST panos.
Isolates the selection algorithm (no re-stitch / re-VGGT confound).

  uniform  : evenly spaced in the pool (current pipeline behaviour)
  fps      : farthest-point sampling on camera centers (max viewpoint spread)
  adaptive : error-guided greedy active learning (seed -> train -> add the
             highest-error candidate viewpoints -> retrain), spends the budget
             where the model is currently worst.

Backend: fused equirect-gsplat (T-F8). Usage:
  select_compare.py <pool_cams.json> [n_train=78] [n_test=12] [iters=7000] [sel_iters=2000]
"""
import sys, os, json, math, random, time, numpy as np, torch
import torch.nn.functional as F
from PIL import Image
REPO = "/w" if os.path.exists("/w/scene/colmap_loader.py") else "/raid/git/gaussian-splatting"
sys.path.insert(0, REPO + "/p3_pano")
from gsplat import DefaultStrategy
from gsplat_equirect import render_equirect_fused

pool_json = sys.argv[1]
N_TRAIN = int(sys.argv[2]) if len(sys.argv) > 2 else 78
N_TEST = int(sys.argv[3]) if len(sys.argv) > 3 else 12
ITERS = int(sys.argv[4]) if len(sys.argv) > 4 else 7000
SEL_ITERS = int(sys.argv[5]) if len(sys.argv) > 5 else 2000
dev = "cuda"; SH_MAX = 3; W = 1024; H = 512
meta = json.load(open(pool_json))
cams_all = meta["cameras"]
NP = len(cams_all)
extent = float(meta["cameras_extent"])
print(f"[select] pool={NP} cams | train={N_TRAIN} test={N_TEST} | iters={ITERS} sel_iters={SEL_ITERS}")

# ---- init point cloud (shared across all runs) ----
from plyfile import PlyData
v = PlyData.read(meta["point_cloud"])["vertex"]
xyz0 = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
rgb0 = (np.stack([v["red"], v["green"], v["blue"]], 1).astype(np.float32) / 255.0
        if "red" in v.data.dtype.names else np.full_like(xyz0, 0.5))
from scipy.spatial import cKDTree
dd, _ = cKDTree(xyz0).query(xyz0, k=4)
scl0 = np.log(np.sqrt(np.clip((dd[:, 1:] ** 2).mean(1), 1e-8, None)))[:, None].repeat(3, 1).astype(np.float32)
def RGB2SH(c): return (c - 0.5) / 0.28209479177387814

# ---- cameras: poses + GT panos (cache GT on GPU) ----
def vm_of(c):
    R = np.array(c["R_wp"], np.float32); T = np.array(c["T"], np.float32)
    m = np.eye(4, dtype=np.float32); m[:3, :3] = R; m[:3, 3] = T
    return torch.tensor(m, device=dev)
VM = [vm_of(c) for c in cams_all]
CEN = np.array([np.array(c["C"], np.float32) for c in cams_all])           # (NP,3) for FPS
CC = [torch.tensor(c["C"], dtype=torch.float32, device=dev) for c in cams_all]
GT = []
for c in cams_all:
    im = Image.open(c["image"]).convert("RGB").resize((W, H), Image.LANCZOS)
    GT.append(torch.tensor(np.asarray(im), dtype=torch.float32, device=dev).permute(2, 0, 1) / 255.0)

def ssim(a, b):
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    k = torch.ones(3, 1, 11, 11, device=a.device) / 121.0
    ma = F.conv2d(a, k, padding=5, groups=3); mb = F.conv2d(b, k, padding=5, groups=3)
    va = F.conv2d(a*a, k, padding=5, groups=3) - ma**2; vb = F.conv2d(b*b, k, padding=5, groups=3) - mb**2
    vab = F.conv2d(a*b, k, padding=5, groups=3) - ma*mb
    return (((2*ma*mb+C1)*(2*vab+C2))/((ma**2+mb**2+C1)*(va+vb+C2))).mean()

def new_splats():
    N = xyz0.shape[0]
    sp = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.tensor(xyz0, device=dev)),
        "scales": torch.nn.Parameter(torch.tensor(scl0, device=dev)),
        "quats": torch.nn.Parameter(torch.tensor([1.,0,0,0], device=dev).repeat(N,1)),
        "opacities": torch.nn.Parameter(torch.logit(torch.full((N,), 0.1, device=dev))),
        "sh0": torch.nn.Parameter(torch.tensor(RGB2SH(rgb0), dtype=torch.float32, device=dev)[:,None,:]),
        "shN": torch.nn.Parameter(torch.zeros((N,(SH_MAX+1)**2-1,3), device=dev)),
    }).to(dev)
    lrs = {"means":0.00016*extent,"scales":0.005,"quats":0.001,"opacities":0.05,"sh0":0.0025,"shN":0.0025/20}
    opt = {k: torch.optim.Adam([{"params": sp[k], "lr": lr}], eps=1e-15) for k,lr in lrs.items()}
    return sp, opt

def render(sp, i, sh_deg):
    colors = torch.cat([sp["sh0"], sp["shN"]], 1)
    img, info = render_equirect_fused(sp["means"], sp["quats"], torch.exp(sp["scales"]),
        torch.sigmoid(sp["opacities"]), colors, VM[i], CC[i], W, H, sh_deg)
    return img.permute(2,0,1).clamp(0,1), info

def train(train_idx, iters):
    sp, opt = new_splats()
    strat = DefaultStrategy(verbose=False, refine_stop_iter=int(iters*0.5), reset_every=3000, refine_every=100)
    strat.check_sanity(sp, opt); state = strat.initialize_state(scene_scale=extent)
    stack = []
    for step in range(iters):
        sh = min(SH_MAX, step // (iters//(SH_MAX+1)+1))
        if not stack: stack = list(train_idx); random.Random(step).shuffle(stack)
        i = stack.pop()
        img, info = render(sp, i, sh)
        strat.step_pre_backward(params=sp, optimizers=opt, state=state, step=step, info=info)
        loss = 0.8*(img-GT[i]).abs().mean() + 0.2*(1.0-ssim(img[None], GT[i][None]))
        loss.backward()
        for o in opt.values(): o.step(); o.zero_grad(set_to_none=True)
        strat.step_post_backward(params=sp, optimizers=opt, state=state, step=step, info=info, packed=False)
    return sp

def psnr(a,b): return float(-10*torch.log10(((a-b)**2).mean()))
def eval_psnr(sp, idx):
    with torch.no_grad():
        return float(np.mean([psnr(render(sp,i,SH_MAX)[0], GT[i]) for i in idx]))
def cand_errors(sp, idx):              # mean L1 error per candidate viewpoint
    with torch.no_grad():
        return {i: float((render(sp,i,SH_MAX)[0]-GT[i]).abs().mean()) for i in idx}

# ---- fixed test set: evenly spaced; remaining = candidates ----
test_idx = sorted(set(np.linspace(0, NP-1, N_TEST).round().astype(int).tolist()))
cand = [i for i in range(NP) if i not in test_idx]
print(f"[select] test={test_idx}")

def sel_uniform():
    return sorted(np.array(cand)[np.linspace(0, len(cand)-1, N_TRAIN).round().astype(int)].tolist())

def sel_fps():
    pts = CEN[cand]; chosen = [0]                         # seed: first candidate
    d = np.linalg.norm(pts - pts[0], axis=1)
    while len(chosen) < N_TRAIN:
        j = int(d.argmax()); chosen.append(j)
        d = np.minimum(d, np.linalg.norm(pts - pts[j], axis=1))
    return sorted(np.array(cand)[chosen].tolist())

def sel_adaptive():
    seed_n = max(N_TRAIN//3, 8)
    sel = set(np.array(cand)[np.linspace(0, len(cand)-1, seed_n).round().astype(int)].tolist())
    while len(sel) < N_TRAIN:
        sp = train(sorted(sel), SEL_ITERS)
        rest = [i for i in cand if i not in sel]
        errs = cand_errors(sp, rest)
        k = min(max((N_TRAIN-len(sel))//2, 8), N_TRAIN-len(sel))
        add = sorted(errs, key=errs.get, reverse=True)[:k]      # highest-error candidates
        sel.update(add)
        print(f"  [adaptive] |sel|={len(sel)} added {len(add)} (max err {max(errs.values()):.4f})")
    return sorted(sel)

results = {}
for name, selector in [("uniform", sel_uniform), ("fps", sel_fps), ("adaptive", sel_adaptive)]:
    t0 = time.time(); tr = selector()
    sp = train(tr, ITERS)
    p = eval_psnr(sp, test_idx); s = float(np.mean([ssim(render(sp,i,SH_MAX)[0][None], GT[i][None]).item() for i in test_idx]))
    results[name] = {"psnr": round(p,3), "ssim": round(s,4), "n_train": len(tr), "sel_s": round(time.time()-t0,1)}
    print(f"[RESULT] {name:9s} PSNR={p:.3f} SSIM={s:.4f}  (n_train={len(tr)}, {results[name]['sel_s']}s)")

print("\n=== 3-WAY (same 12 test, same 78 budget, fused T-F8) ===")
for k in ["uniform","fps","adaptive"]:
    print(f"  {k:9s} PSNR {results[k]['psnr']:.3f}  SSIM {results[k]['ssim']:.4f}")
json.dump({"test_idx": test_idx, "results": results}, open(os.path.join(os.path.dirname(pool_json), "select_compare_result.json"), "w"), indent=1)
