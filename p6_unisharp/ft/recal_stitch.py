#!/usr/bin/env python3
"""Calibrated dual-fisheye -> ERP re-stitcher (replaces v360 nominal mapping).
Forward model (COLMAP OPENCV_FISHEYE, self-calibrated on 023 down lens, 1920^2):
  r = f*theta*(1 + k1 t^2 + k2 t^4 + k3 t^6 + k4 t^8),  f=(548.093+548.146)/2
Assumes both lenses share the model (same physical lens type).
mode=cal   : grid-search per-lens roll & phi-sign on frame 120 by correlating
             against the existing v360 pano (globally correct) -> prints best.
mode=batch : given rolls/signs, stitch N frames to 4096x2048 ERPs.
"""
import sys, os, math
import numpy as np, torch, torch.nn.functional as Fn
from PIL import Image

F_CAL = (548.09284117862558 + 548.14584281626901) / 2
K = (0.030349368620543618, 0.0023128116945658281, -0.0027963710018365905, -0.00035606873525276218)
CX = CY = 960.0; IMW = 1920
TH_MAX = math.radians(100.35)
dev = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = "/w"
DOWN = f"{ROOT}/data/8kpano/scenes/fish023/images"      # stream0 = down
UP   = f"{ROOT}/data/8kpano/scenes/fish023/images_up"   # stream1 = up

def lut(H, W, axis_y, roll_deg, phi_sign):
    """ERP grid -> fisheye pixel LUT + validity(theta)."""
    vv, uu = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")
    lon = (uu + 0.5) / W * 2 * math.pi - math.pi
    lat = math.pi / 2 - (vv + 0.5) / H * math.pi
    d = torch.stack([torch.cos(lat) * torch.sin(lon), torch.sin(lat), torch.cos(lat) * torch.cos(lon)], -1)
    z = torch.tensor([0.0, axis_y, 0.0], device=dev)
    x0 = torch.tensor([1.0, 0.0, 0.0], device=dev); y0 = torch.linalg.cross(z, x0)
    r = math.radians(roll_deg)
    xl = math.cos(r) * x0 + math.sin(r) * y0; yl = -math.sin(r) * x0 + math.cos(r) * y0
    dz = (d @ z).clamp(-1, 1); th = torch.acos(dz)
    phi = torch.atan2(d @ yl, d @ xl) * phi_sign
    t2 = th * th
    rr = F_CAL * th * (1 + K[0]*t2 + K[1]*t2**2 + K[2]*t2**3 + K[3]*t2**4)
    px = CX + rr * torch.cos(phi); py = CY + rr * torch.sin(phi)
    gx = px / (IMW - 1) * 2 - 1; gy = py / (IMW - 1) * 2 - 1
    grid = torch.stack([gx, gy], -1)
    return grid, th

def load(p, size=None):
    im = Image.open(p).convert("RGB")
    if size: im = im.resize((size, size), Image.BILINEAR)
    return torch.from_numpy(np.asarray(im, np.float32)).permute(2, 0, 1)[None].to(dev) / 255

def stitch(fd, fu, H, W, roll_d, sign_d, roll_u, sign_u):
    gd, thd = lut(H, W, -1.0, roll_d, sign_d)
    gu, thu = lut(H, W,  1.0, roll_u, sign_u)
    a = Fn.grid_sample(fd, gd[None], mode="bilinear", padding_mode="zeros", align_corners=True)[0]
    b = Fn.grid_sample(fu, gu[None], mode="bilinear", padding_mode="zeros", align_corners=True)[0]
    # feathered weights by theta (valid < TH_MAX; blend where both valid)
    wd = ((TH_MAX - thd) / 0.18).clamp(0, 1) * (thd < TH_MAX)
    wu = ((TH_MAX - thu) / 0.18).clamp(0, 1) * (thu < TH_MAX)
    s = (wd + wu).clamp(min=1e-6)
    return (a * wd + b * wu) / s

if sys.argv[1] == "cal":
    ref = load(f"{ROOT}/data/8kpano/scenes/scene_023hf_pano/panoramas/pano_0120.jpg")
    ref = Fn.interpolate(ref, size=(256, 512), mode="bilinear")[0]
    fd = load(f"{DOWN}/f_0120.jpg"); fu = load(f"{UP}/f_0120.jpg")
    H, W = 256, 512
    best = {}
    for tag, axis, half in (("down", -1.0, slice(128, 256)), ("up", 1.0, slice(0, 128))):
        sc = []
        for sign in (1, -1):
            for roll in range(0, 360, 10):
                g, th = lut(H, W, axis, roll, sign)
                img = Fn.grid_sample(fd if tag == "down" else fu, g[None], mode="bilinear",
                                     padding_mode="zeros", align_corners=True)[0]
                v = (th < TH_MAX)[half]
                a = img[:, half, :][:, v]; b = ref[:, half, :][:, v]
                a = a - a.mean(); b = b - b.mean()
                c = float((a * b).sum() / (a.norm() * b.norm() + 1e-9))
                sc.append((c, roll, sign))
        sc.sort(reverse=True)
        best[tag] = sc[0]
        print(f"{tag}: corr={sc[0][0]:.3f} roll={sc[0][1]} sign={sc[0][2]} | 2nd {sc[1]}")
    # fine search +-10 in 2deg
    for tag, axis in (("down", -1.0), ("up", 1.0)):
        c0, r0, s0 = best[tag]; half = slice(128, 256) if tag == "down" else slice(0, 128)
        sc = []
        for roll in range(r0 - 10, r0 + 11, 2):
            g, th = lut(H, W, axis, roll % 360, s0)
            img = Fn.grid_sample(fd if tag == "down" else fu, g[None], mode="bilinear",
                                 padding_mode="zeros", align_corners=True)[0]
            v = (th < TH_MAX)[half]
            a = img[:, half, :][:, v]; b = ref[:, half, :][:, v]
            a = a - a.mean(); b = b - b.mean()
            sc.append((float((a*b).sum()/(a.norm()*b.norm()+1e-9)), roll % 360, s0))
        sc.sort(reverse=True); print(f"{tag} fine: corr={sc[0][0]:.3f} roll={sc[0][1]} sign={sc[0][2]}")
        best[tag] = sc[0]
    # save preview at chosen params
    out = stitch(fd, fu, 512, 1024, best["down"][1], best["down"][2], best["up"][1], best["up"][2])
    Image.fromarray((out.permute(1,2,0).clamp(0,1)*255).byte().cpu().numpy()).save(f"{ROOT}/p6_unisharp/ft/recal_preview.jpg")
    print("preview saved; params:", best["down"][1], best["down"][2], best["up"][1], best["up"][2])
elif sys.argv[1] == "batch":  # batch <roll_d> <sign_d> <roll_u> <sign_u> <outdir> [N]
    rd, sd, ru, su = float(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
    outdir = sys.argv[6]; N = int(sys.argv[7]) if len(sys.argv) > 7 else 240
    os.makedirs(outdir, exist_ok=True)
    H, W = 2048, 4096
    gd, thd = lut(H, W, -1.0, rd, sd); gu, thu = lut(H, W, 1.0, ru, su)
    wd = ((TH_MAX - thd) / 0.18).clamp(0, 1) * (thd < TH_MAX)
    wu = ((TH_MAX - thu) / 0.18).clamp(0, 1) * (thu < TH_MAX)
    s = (wd + wu).clamp(min=1e-6)
    for k in range(1, N + 1):
        fd = load(f"{DOWN}/f_{k:04d}.jpg"); fu = load(f"{UP}/f_{k:04d}.jpg")
        a = Fn.grid_sample(fd, gd[None], mode="bilinear", padding_mode="zeros", align_corners=True)[0]
        b = Fn.grid_sample(fu, gu[None], mode="bilinear", padding_mode="zeros", align_corners=True)[0]
        out = (a * wd + b * wu) / s
        Image.fromarray((out.permute(1,2,0).clamp(0,1)*255).byte().cpu().numpy()).save(
            f"{outdir}/pano_{k:04d}.jpg", quality=92)
        if k % 60 == 0: print(k, flush=True)
    print(f"[batch] {N} calibrated ERPs -> {outdir}")

# mode=calup: align UP lens to DOWN lens via the shared overlap ring (texture-rich
# horizon band), no external reference needed.
if sys.argv[1] == "calup":
    fd = load(f"{DOWN}/f_0120.jpg"); fu = load(f"{UP}/f_0120.jpg")
    H, W = 384, 768
    gd, thd = lut(H, W, -1.0, 0, -1)
    ring = (thd > math.radians(80)) & (thd < math.radians(100))
    a = Fn.grid_sample(fd, gd[None], mode="bilinear", padding_mode="zeros", align_corners=True)[0]
    sc = []
    for sign in (1, -1):
        for roll in range(0, 360, 2):
            gu, thu = lut(H, W, 1.0, roll, sign)
            m = ring & (thu < TH_MAX)
            if m.sum() < 500: continue
            b = Fn.grid_sample(fu, gu[None], mode="bilinear", padding_mode="zeros", align_corners=True)[0]
            x = a[:, m] - a[:, m].mean(); y = b[:, m] - b[:, m].mean()
            sc.append((float((x*y).sum()/(x.norm()*y.norm()+1e-9)), roll, sign))
    sc.sort(reverse=True)
    print("UP-lens ring alignment top3:", [(round(c,3), r, s) for c, r, s in sc[:3]])

if sys.argv[1] == "synctest":  # ring corr at fixed (180,-1) across frames
    H, W = 384, 768
    gd, thd = lut(H, W, -1.0, 0, -1)
    gu, thu = lut(H, W, 1.0, 180, -1)
    ring = (thd > math.radians(80)) & (thd < math.radians(100)) & (thu < TH_MAX)
    for k in (30, 60, 120, 180, 230):
        fd = load(f"{DOWN}/f_{k:04d}.jpg"); fu = load(f"{UP}/f_{k:04d}.jpg")
        a = Fn.grid_sample(fd, gd[None], align_corners=True)[0][:, ring]
        b = Fn.grid_sample(fu, gu[None], align_corners=True)[0][:, ring]
        x = a - a.mean(); y = b - b.mean()
        print(f"frame {k}: ring corr={float((x*y).sum()/(x.norm()*y.norm()+1e-9)):.3f}")
