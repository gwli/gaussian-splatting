#!/usr/bin/env python3
"""Compare models trained at different resolutions on a COMMON yardstick.

PSNR is not comparable across evaluation resolutions (higher-res GT carries more
high-frequency content, so the same model scores lower). This renders every model
at a fixed high resolution and reports, per model:
  PSNR@hi   -- native high-res comparison
  PSNR@lo   -- both render and GT bilinearly downsampled to `lo` (common reference)
  sharp@hi  -- grad(render)/grad(GT) on the lower 75% rows
usage: eval_res.py <cams.json> <test_idx.txt> <hiW> <loW> <ply1> [<ply2> ...]
"""
import sys, os, json
import numpy as np, torch, torch.nn.functional as F
from PIL import Image

REPO = os.getcwd()
sys.path.insert(0, REPO + "/p3_pano")
from gsplat_equirect import render_equirect_fused
from plyfile import PlyData

cams_json, tif, hiW, loW = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
plys = sys.argv[5:]
hiH, loH = hiW // 2, loW // 2
dev = "cuda"
cams = json.load(open(cams_json))["cameras"]
hold = {int(l) for l in open(tif) if l.strip()}
test = [c for c in cams if c["idx"] in hold]

def load_ply(p):
    v = PlyData.read(p)["vertex"]; names = v.data.dtype.names
    nrest = len([n for n in names if n.startswith("f_rest_")])
    fr = np.stack([v[f"f_rest_{i}"] for i in range(nrest)], 1).astype(np.float32)
    return dict(
        xyz=torch.tensor(np.stack([v["x"], v["y"], v["z"]], 1), dtype=torch.float32, device=dev),
        colors=torch.cat([torch.tensor(np.stack([v[f"f_dc_{i}"] for i in range(3)], 1),
                                       dtype=torch.float32, device=dev)[:, None, :],
                          torch.tensor(fr.reshape(len(fr), 3, nrest // 3).transpose(0, 2, 1).copy(), device=dev)], 1),
        opac=torch.sigmoid(torch.tensor(np.asarray(v["opacity"], np.float32), device=dev)),
        scal=torch.exp(torch.tensor(np.stack([v[f"scale_{i}"] for i in range(3)], 1), dtype=torch.float32, device=dev)),
        quat=torch.tensor(np.stack([v[f"rot_{i}"] for i in range(4)], 1), dtype=torch.float32, device=dev))

def grad(x):
    gx = torch.abs(x[:, 1:] - x[:, :-1]); gy = torch.abs(x[1:, :] - x[:-1, :])
    return gx[:-1] + gy[:, :-1]

sys.path.insert(0, REPO)
from lpipsPyTorch import lpips as _lpips
def _ssim(a, b):                      # same 11x11 uniform-window SSIM as the trainer
    C1, C2 = 0.01**2, 0.03**2
    k = torch.ones(3, 1, 11, 11, device=a.device) / 121.0
    ma = F.conv2d(a, k, padding=5, groups=3); mb = F.conv2d(b, k, padding=5, groups=3)
    va = F.conv2d(a*a, k, padding=5, groups=3) - ma**2
    vb = F.conv2d(b*b, k, padding=5, groups=3) - mb**2
    vab = F.conv2d(a*b, k, padding=5, groups=3) - ma*mb
    return float((((2*ma*mb+C1)*(2*vab+C2))/((ma**2+mb**2+C1)*(va+vb+C2))).mean())
print(f"{'model':22s} {'N':>9} {'PSNR@'+str(loW):>10} {'SSIM@'+str(loW):>10} {'LPIPS@'+str(loW):>11} {'sharp@'+str(hiW):>10}")
for p in plys:
    G = load_ply(p)
    hi, lo, sh, ss, lp = [], [], [], [], []
    with torch.no_grad():
        for c in test:
            R = np.array(c["R_wp"], np.float32); T = np.array(c["T"], np.float32)
            vm = np.eye(4, dtype=np.float32); vm[:3, :3] = R; vm[:3, 3] = T
            img, _ = render_equirect_fused(G["xyz"], G["quat"], G["scal"], G["opac"], G["colors"],
                                           torch.tensor(vm, device=dev),
                                           torch.tensor(np.array(c["C"], np.float32), device=dev), hiW, hiH, 3)
            img = img.clamp(0, 1).permute(2, 0, 1)
            gt = torch.tensor(np.asarray(Image.open(c["image"]).convert("RGB").resize((hiW, hiH), Image.LANCZOS)),
                              dtype=torch.float32, device=dev).permute(2, 0, 1) / 255
            hi.append(float(-10 * torch.log10(((img - gt) ** 2).mean())))
            il = F.interpolate(img[None], size=(loH, loW), mode="bilinear", antialias=True)[0]
            gl = F.interpolate(gt[None], size=(loH, loW), mode="bilinear", antialias=True)[0]
            lo.append(float(-10 * torch.log10(((il - gl) ** 2).mean())))
            r, g = img.mean(0)[hiH // 4:], gt.mean(0)[hiH // 4:]
            sh.append(100 * float(grad(r).mean() / (grad(g).mean() + 1e-8)))
            ss.append(_ssim(il[None], gl[None])); lp.append(float(_lpips(il[None], gl[None], net_type='vgg')))
    print(f"{p.split('/runs/')[1].split('/')[0]:22s} {len(G['xyz']):9d} "
          f"{np.mean(lo):10.3f} {np.mean(ss):10.4f} {np.mean(lp):11.4f} {np.mean(sh):9.1f}%")
