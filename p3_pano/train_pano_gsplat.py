#!/usr/bin/env python3
"""T-F6: direct equirectangular 3DGS training on the gsplat backend.

gsplat (nerfstudio 1.5.3) is pinhole-only — it has no equirect projection, so it
can't drop into train_pano.py's LONLAT rasterizer directly. Instead we render 6
pinhole CUBE FACES per panorama with gsplat (one batched C=6 rasterization) and
resample them into an equirectangular image with a differentiable grid_sample,
using the EXACT same view->lon/lat convention as the OmniGS LONLAT rasterizer
(auxiliary.h point3ToLonlatPixel: lon=atan2(x,z), lat=asin(y)). Loss + grads flow
back through the resampling into gsplat, and gsplat's native DefaultStrategy
drives densification. This gives the fast pinhole backend on direct-pano training.

Usage: train_pano_gsplat.py <pano_cams.json> <out_dir> [iters=7000] [width=1024] [face=512]
"""
import sys, os, json, math, random, importlib.util, numpy as np, torch
import torch.nn.functional as F
from PIL import Image

REPO = "/w" if os.path.exists("/w/scene/colmap_loader.py") else "/raid/git/gaussian-splatting"
from gsplat import rasterization, DefaultStrategy

cams_json, out_dir = sys.argv[1], sys.argv[2]
ITERS = int(sys.argv[3]) if len(sys.argv) > 3 else 7000
W = int(sys.argv[4]) if len(sys.argv) > 4 else 1024
FACE = int(sys.argv[5]) if len(sys.argv) > 5 else 512
H = W // 2
os.makedirs(out_dir, exist_ok=True)
dev = "cuda"; SH_MAX = 3
meta = json.load(open(cams_json))

# ---------------- cube-face geometry (pano-view frame, +y is DOWN) ----------------
def face_R(forward, up):                 # v_face = R @ v_pano ; face looks along +z=forward
    z = np.asarray(forward, float); z /= np.linalg.norm(z)
    x = np.cross(up, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], 0).astype(np.float32)

FACES = [                                # 6 faces covering the full sphere
    face_R((0, 0, 1),  (0, -1, 0)),      # 0 front +z
    face_R((0, 0, -1), (0, -1, 0)),      # 1 back  -z
    face_R((1, 0, 0),  (0, -1, 0)),      # 2 right +x
    face_R((-1, 0, 0), (0, -1, 0)),      # 3 left  -x
    face_R((0, 1, 0),  (0, 0, 1)),       # 4 down  +y
    face_R((0, -1, 0), (0, 0, -1)),      # 5 up    -y
]
FACES_t = torch.tensor(np.stack(FACES), device=dev)            # (6,3,3)
# pinhole intrinsics for a 90deg face
f = FACE / 2.0
K_face = torch.tensor([[f, 0, FACE / 2], [0, f, FACE / 2], [0, 0, 1]],
                      dtype=torch.float32, device=dev)

# ---------------- precompute equirect <- cube sampling (the LONLAT convention) ----
yy, xx = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")
lon = ((xx + 0.5) / (W / 2) - 1.0) * math.pi          # px = (lon/pi + 1)*W/2
lat = ((yy + 0.5) / (H / 2) - 1.0) * (math.pi / 2)    # py = (lat*2/pi + 1)*H/2
d = torch.stack([torch.cos(lat) * torch.sin(lon),     # x  (matches lon=atan2(x,z))
                 torch.sin(lat),                       # y  (lat=asin(y))
                 torch.cos(lat) * torch.cos(lon)], -1) # z              (H,W,3) view-space dir
# per-face sampling grid + ownership mask
face_grid, face_mask = [], []
for fi in range(6):
    df = torch.einsum("ij,hwj->hwi", FACES_t[fi], d)  # direction in face frame
    z = df[..., 2].clamp_min(1e-6)
    gx, gy = df[..., 0] / z, df[..., 1] / z           # normalized [-1,1] over the 90deg face
    face_grid.append(torch.stack([gx, gy], -1))
    inside = (df[..., 2] > 0) & (gx.abs() <= 1.0001) & (gy.abs() <= 1.0001)
    face_mask.append(inside)
# assign each equirect pixel to the face whose forward is most aligned (unique owner)
align = torch.stack([(d * torch.tensor(FACES[fi][2], device=dev)).sum(-1) for fi in range(6)], 0)
owner = align.argmax(0)                                # (H,W) in 0..5
face_grid = torch.stack(face_grid, 0)                  # (6,H,W,2)
owner_oh = torch.stack([(owner == fi) for fi in range(6)], 0).float()[:, None]  # (6,1,H,W)


def assemble_equirect(faces_img):                      # faces_img: (6,3,FACE,FACE) -> (3,H,W)
    sampled = F.grid_sample(faces_img, face_grid, mode="bilinear",
                            align_corners=False, padding_mode="border")  # (6,3,H,W)
    return (sampled * owner_oh).sum(0)                 # owner mask is one-hot -> pick


# ---------------- init gaussians from VGGT points3D.ply ----------------
from plyfile import PlyData
ply = PlyData.read(meta["point_cloud"]); v = ply["vertex"]
xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
if "red" in v.data.dtype.names:
    rgb = np.stack([v["red"], v["green"], v["blue"]], 1).astype(np.float32) / 255.0
else:
    rgb = np.full_like(xyz, 0.5)
N = xyz.shape[0]
from scipy.spatial import cKDTree
dd, _ = cKDTree(xyz).query(xyz, k=4)
scales0 = np.log(np.sqrt(np.clip((dd[:, 1:] ** 2).mean(1), 1e-8, None)))[:, None].repeat(3, 1).astype(np.float32)

def RGB2SH(c): return (c - 0.5) / 0.28209479177387814
sh0 = torch.tensor(RGB2SH(rgb), dtype=torch.float32, device=dev)[:, None, :]
shN = torch.zeros((N, (SH_MAX + 1) ** 2 - 1, 3), dtype=torch.float32, device=dev)
extent = float(meta["cameras_extent"])
print(f"[pano-gsplat] init {N} pts | extent {extent:.3f} | equirect {W}x{H} face {FACE}")

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
strat = DefaultStrategy(verbose=False, refine_stop_iter=int(ITERS * 0.5),
                        reset_every=3000, refine_every=100)
strat.check_sanity(splats, opt); state = strat.initialize_state(scene_scale=extent)

# ---------------- cameras ----------------
def load_cam(c):
    R = np.array(c["R_wp"], np.float32)       # world->pano-view rotation
    T = np.array(c["T"], np.float32)
    im = Image.open(c["image"]).convert("RGB").resize((W, H), Image.LANCZOS)
    gt = torch.tensor(np.asarray(im), dtype=torch.float32, device=dev).permute(2, 0, 1) / 255.0
    return {"R": R, "T": T, "gt": gt, "name": f"pano_{c['idx']:04d}"}

cams = [load_cam(c) for c in meta["cameras"]]
test = cams[::8]; train = [c for i, c in enumerate(cams) if i % 8 != 0]
print(f"[pano-gsplat] {len(cams)} cams -> {len(train)} train / {len(test)} test")


def render_pano(cam, sh_deg):
    R = torch.tensor(cam["R"], device=dev); T = torch.tensor(cam["T"], device=dev)
    # 6 face world->cam: rot = R_face @ R_wp , trans = R_face @ T
    rot = FACES_t @ R[None]                                # (6,3,3)
    trans = torch.einsum("fij,j->fi", FACES_t, T)          # (6,3)
    vm = torch.zeros(6, 4, 4, device=dev); vm[:, :3, :3] = rot; vm[:, :3, 3] = trans; vm[:, 3, 3] = 1
    colors = torch.cat([splats["sh0"], splats["shN"]], 1)
    rc, _, info = rasterization(
        means=splats["means"], quats=splats["quats"], scales=torch.exp(splats["scales"]),
        opacities=torch.sigmoid(splats["opacities"]), colors=colors,
        viewmats=vm, Ks=K_face[None].expand(6, 3, 3), width=FACE, height=FACE,
        sh_degree=sh_deg, render_mode="RGB", packed=False, absgrad=strat.absgrad)
    faces_img = rc.permute(0, 3, 1, 2).clamp(0, 1)         # (6,3,FACE,FACE)
    return assemble_equirect(faces_img), info              # (3,H,W)


def ssim(a, b):
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    k = torch.ones(3, 1, 11, 11, device=a.device) / 121.0
    mu_a = F.conv2d(a, k, padding=5, groups=3); mu_b = F.conv2d(b, k, padding=5, groups=3)
    va = F.conv2d(a * a, k, padding=5, groups=3) - mu_a ** 2
    vb = F.conv2d(b * b, k, padding=5, groups=3) - mu_b ** 2
    vab = F.conv2d(a * b, k, padding=5, groups=3) - mu_a * mu_b
    return (((2 * mu_a * mu_b + C1) * (2 * vab + C2)) /
            ((mu_a ** 2 + mu_b ** 2 + C1) * (va + vb + C2))).mean()


# ---------------- train ----------------
import time
torch.manual_seed(0)
stack = []; t0 = time.time(); ema = None
for step in range(ITERS):
    sh_deg = min(SH_MAX, step // (ITERS // (SH_MAX + 1) + 1))
    if not stack:
        stack = train.copy(); random.Random(step).shuffle(stack)
    cam = stack.pop()
    img, info = render_pano(cam, sh_deg)
    strat.step_pre_backward(params=splats, optimizers=opt, state=state, step=step, info=info)
    gt = cam["gt"]
    l1 = (img - gt).abs().mean()
    loss = 0.8 * l1 + 0.2 * (1.0 - ssim(img[None], gt[None]))
    loss.backward()
    for o in opt.values(): o.step(); o.zero_grad(set_to_none=True)
    strat.step_post_backward(params=splats, optimizers=opt, state=state, step=step,
                             info=info, packed=False)
    ema = loss.item() if ema is None else 0.9 * ema + 0.1 * loss.item()
    if step % 500 == 0 or step == ITERS - 1:
        print(f"  it {step:5d}  loss {ema:.4f}  N={splats['means'].shape[0]}  "
              f"{(step+1)/(time.time()-t0):.1f} it/s", flush=True)
torch.cuda.synchronize(); dt = time.time() - t0

# ---------------- eval ----------------
def psnr(a, b): return float(-10 * torch.log10(((a - b) ** 2).mean()))
try:
    sys.path.insert(0, REPO); from lpipsPyTorch import lpips as _lpips; HAVE_LPIPS = True
except Exception:
    HAVE_LPIPS = False
ps, ss, lp = [], [], []
with torch.no_grad():
    for cam in test:
        img, _ = render_pano(cam, SH_MAX); img = img.clamp(0, 1); gt = cam["gt"]
        ps.append(psnr(img, gt)); ss.append(float(ssim(img[None], gt[None])))
        if HAVE_LPIPS: lp.append(float(_lpips(img[None], gt[None], net_type="vgg")))
res = {"scene": os.path.basename(os.path.dirname(out_dir)) or out_dir,
       "method": "direct-pano-gsplat", "backend": "gsplat-cubemap", "iterations": ITERS,
       "train_res": [W, H], "face": FACE, "n_gaussians": int(splats["means"].shape[0]),
       "n_train": len(train), "n_test": len(test), "iter_s": round(ITERS / dt, 1),
       "train_s": round(dt, 1), "PSNR": round(float(np.mean(ps)), 3),
       "SSIM": round(float(np.mean(ss)), 4), "LPIPS": (round(float(np.mean(lp)), 4) if lp else None)}
json.dump(res, open(os.path.join(out_dir, "results.json"), "w"), indent=1)
print(f"[EVAL] PSNR={res['PSNR']} SSIM={res['SSIM']} LPIPS={res['LPIPS']} "
      f"over {len(test)} panos | {res['iter_s']} it/s, {res['train_s']}s")
print(f"[DONE] {res}")
