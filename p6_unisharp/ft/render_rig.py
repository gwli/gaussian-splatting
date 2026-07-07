#!/usr/bin/env python3
"""Post-train verification for rig-stitched ERP runs.
Renders test cams from a saved INRIA ply via gsplat equirect, writes
render|GT stacks, measures (a) global pitch offset render->GT via vertical
cross-correlation of row-mean profiles, (b) sharpness retention on the
lower-75% rows (grad(render)/grad(GT)).  usage: render_rig.py <cams.json> <ply> <out_prefix>
"""
import sys, os, json, numpy as np, torch
from PIL import Image

REPO = "/w" if os.path.exists("/w/p3_pano") else "/raid/git/gaussian-splatting"
sys.path.insert(0, REPO + "/p3_pano")
from gsplat_equirect import render_equirect_fused

cams_json, ply_path, out_prefix = sys.argv[1], sys.argv[2], sys.argv[3]
W = 1024; H = W // 2; dev = "cuda"
meta = json.load(open(cams_json))

from plyfile import PlyData
v = PlyData.read(ply_path)["vertex"]
names = v.data.dtype.names
xyz = torch.tensor(np.stack([v["x"], v["y"], v["z"]], 1), dtype=torch.float32, device=dev)
sh0 = torch.tensor(np.stack([v[f"f_dc_{i}"] for i in range(3)], 1), dtype=torch.float32, device=dev)[:, None, :]
nrest = len([n for n in names if n.startswith("f_rest_")])
fr = np.stack([v[f"f_rest_{i}"] for i in range(nrest)], 1).astype(np.float32)
shN = torch.tensor(fr.reshape(len(fr), 3, nrest // 3).transpose(0, 2, 1).copy(), device=dev)
opac = torch.sigmoid(torch.tensor(np.asarray(v["opacity"], np.float32), device=dev))
scal = torch.exp(torch.tensor(np.stack([v[f"scale_{i}"] for i in range(3)], 1), dtype=torch.float32, device=dev))
quat = torch.tensor(np.stack([v[f"rot_{i}"] for i in range(4)], 1), dtype=torch.float32, device=dev)
colors = torch.cat([sh0, shN], 1)
print(f"loaded {len(xyz)} gaussians, shN={nrest//3} coefs")

cams = meta["cameras"]
test_idx = list(range(0, len(cams), 8))[:32]
def grad(x):
    gx = np.abs(np.diff(x, axis=1)); gy = np.abs(np.diff(x, axis=0))
    return gx[:-1, :] + gy[:, :-1]

pitches, rets = [], []
for j, ti in enumerate(test_idx):
    c = cams[ti]
    R = np.array(c["R_wp"], np.float32); T = np.array(c["T"], np.float32)
    vm = np.eye(4, dtype=np.float32); vm[:3, :3] = R; vm[:3, 3] = T
    with torch.no_grad():
        img, _ = render_equirect_fused(xyz, quat, scal, opac, colors,
                                       torch.tensor(vm, device=dev),
                                       torch.tensor(np.array(c["C"], np.float32), device=dev), W, H, 3)
    ren = (img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    gt = np.asarray(Image.open(os.path.join(REPO, c["image"])).convert("RGB").resize((W, H), Image.LANCZOS))
    rg, gg = ren.mean(2).astype(np.float32), gt.mean(2).astype(np.float32)
    # pitch offset: best vertical shift of row-mean luminance profile
    pr, pg = rg.mean(1), gg.mean(1)
    pr, pg = pr - pr.mean(), pg - pg.mean()
    sh = range(-40, 41)
    cc = [float((np.roll(pr, s) * pg).sum()) for s in sh]
    best = list(sh)[int(np.argmax(cc))]
    pitches.append(best * 180.0 / H)  # rows -> degrees
    # sharpness retention on lower 75%
    lo = H // 4
    grn, ggt = grad(rg[lo:]), grad(gg[lo:])
    rets.append(100 * grn.mean() / (ggt.mean() + 1e-6))
    if j < 4:
        Image.fromarray(np.concatenate([ren, gt], 0)).save(f"{out_prefix}_{c['idx']:04d}.png")
print(f"pitch offset (deg): med={np.median(pitches):+.2f} p90={np.percentile(np.abs(pitches),90):.2f}")
print(f"sharpness retention lower-75%: med={np.median(rets):.1f}% mean={np.mean(rets):.1f}%")
