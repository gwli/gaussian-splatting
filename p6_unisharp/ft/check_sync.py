#!/usr/bin/env python3
"""Test for a temporal offset between the two fisheye streams.

Motivation: rig BA estimated a down->up translation of ~1.5 m, but the physical
lens baseline is ~2-3 cm. 1.5 m / 4.68 m/s ~ 0.32 s ~ one frame at 3 fps, i.e.
the two streams may be offset by a frame. If so, every stitched ERP glues two
hemispheres captured at different instants.

Method: sample both lenses on the shared overlap ring (theta 80-100 deg) under
R_rig and correlate down[k] against up[k+d] for d in {-2..2}. The peak tells us
the true pairing.
usage: check_sync.py [root] [n_frames]
"""
import sys, os, math
import numpy as np, torch, torch.nn.functional as F
from PIL import Image

ROOT = sys.argv[1] if len(sys.argv) > 1 else "data/8kpano/scenes/fish023rig"
NK = int(sys.argv[2]) if len(sys.argv) > 2 else 12
dev = "cuda" if torch.cuda.is_available() else "cpu"
z = np.load("p3_pano/rig023.npz")
R_rig = torch.tensor(z["R_rig"], dtype=torch.float32, device=dev)
cal = [547.11, 547.11, 960.0, 960.0, 0.0303493686, 0.0023128117, -0.0027963710, -0.0003560687]

H, W = 512, 1024                                   # sampling grid on the sphere
vv, uu = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")
lat = (vv + 0.5) / H * math.pi - math.pi / 2
lon = (uu + 0.5) / W * 2 * math.pi - math.pi
d_erp = torch.stack([torch.cos(lat)*torch.sin(lon), torch.sin(lat), torch.cos(lat)*torch.cos(lon)], -1)
R_e2c = torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=torch.float32, device=dev)
d_dn = torch.einsum("ij,hwj->hwi", R_e2c, d_erp)
d_up = torch.einsum("ij,hwj->hwi", R_rig, d_dn)

def proj(d):
    fx, fy, cx, cy, k1, k2, k3, k4 = cal
    x, y, zc = d[..., 0], d[..., 1], d[..., 2]
    hyp = torch.sqrt(x*x + y*y).clamp(min=1e-9)
    th = torch.atan2(hyp, zc); t2 = th*th
    r = th*(1 + k1*t2 + k2*t2**2 + k3*t2**3 + k4*t2**4)
    u = fx*r*x/hyp + cx; v = fy*r*y/hyp + cy
    return torch.stack([u/(1920-1)*2-1, v/(1920-1)*2-1], -1), th

gd, thd = proj(d_dn); gu, thu = proj(d_up)
ring = (thd > math.radians(80)) & (thd < math.radians(100)) & (thu < math.radians(100))
print(f"overlap ring: {int(ring.sum())} sample points")

def load(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert("L"), np.float32))[None, None].to(dev) / 255

ks = np.linspace(60, 1080, NK).astype(int)
res = {}
for dk in (-2, -1, 0, 1, 2):
    cs = []
    for k in ks:
        pd = f"{ROOT}/down_all/f_{k:04d}.jpg"; pu = f"{ROOT}/up_all/f_{k+dk:04d}.jpg"
        if not (os.path.exists(pd) and os.path.exists(pu)): continue
        a = F.grid_sample(load(pd), gd[None], align_corners=True)[0, 0][ring]
        b = F.grid_sample(load(pu), gu[None], align_corners=True)[0, 0][ring]
        a = a - a.mean(); b = b - b.mean()
        cs.append(float((a*b).sum() / (a.norm()*b.norm() + 1e-9)))
    res[dk] = float(np.mean(cs))
    print(f"  up offset {dk:+d} frame ({dk*0.334:+.2f} s): ring corr = {res[dk]:.4f}  (n={len(cs)})")
best = max(res, key=res.get)
print(f"\nbest pairing: down[k] <-> up[k{best:+d}]"
      f"{'  -> streams are SYNCHRONISED as extracted' if best == 0 else '  -> STREAMS ARE OFFSET'}")
