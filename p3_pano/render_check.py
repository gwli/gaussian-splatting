#!/usr/bin/env python3
"""Sanity-render a trained pano .ply from a (held-out) training camera with the
T-F8 fused equirect kernel, save render | ground-truth stacked for eyeball check.
Usage: render_check.py <ply> <pano_cams.json> <out.png> [pick=test1] [W=2048]"""
import sys, os, json, numpy as np, torch
from plyfile import PlyData
from PIL import Image
sys.path.insert(0, "/w/p3_pano")
from gsplat_equirect import render_equirect_fused

ply_p, cams_p, out_p = sys.argv[1], sys.argv[2], sys.argv[3]
pick = sys.argv[4] if len(sys.argv) > 4 else "test1"
W = int(sys.argv[5]) if len(sys.argv) > 5 else 2048
H = W // 2
dev = "cuda"

v = PlyData.read(ply_p)["vertex"]
xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
op = v["opacity"].astype(np.float32)[:, None]
sc = np.stack([v[f"scale_{i}"] for i in range(3)], 1).astype(np.float32)
rot = np.stack([v[f"rot_{i}"] for i in range(4)], 1).astype(np.float32)
fdc = np.stack([v[f"f_dc_{i}"] for i in range(3)], 1).astype(np.float32)
rk = sorted([k for k in v.data.dtype.names if k.startswith("f_rest_")], key=lambda s: int(s.split("_")[-1]))
frest = np.stack([v[k] for k in rk], 1).astype(np.float32)
N = xyz.shape[0]
shN = frest.reshape(N, 3, len(rk) // 3).transpose(0, 2, 1)
sh = np.concatenate([fdc[:, None, :], shN], 1).astype(np.float32)   # (N,K,3)
K = sh.shape[1]; sh_deg = int(round(K ** 0.5)) - 1

means = torch.tensor(xyz, device=dev)
quats = torch.nn.functional.normalize(torch.tensor(rot, device=dev), dim=-1)
scales = torch.exp(torch.tensor(sc, device=dev))
opac = torch.sigmoid(torch.tensor(op, device=dev)).squeeze(1)
shc = torch.tensor(sh, device=dev)
print(f"[check] {N} gaussians, SH K={K} (deg {sh_deg})")

meta = json.load(open(cams_p))
cams = meta["cameras"]
test = cams[::8]
cam = test[int(pick[4:])] if pick.startswith("test") else cams[int(pick)]
R = np.array(cam["R_wp"], np.float32); T = np.array(cam["T"], np.float32)
vm = np.eye(4, dtype=np.float32); vm[:3, :3] = R; vm[:3, 3] = T
vm = torch.tensor(vm, device=dev); C = torch.tensor(np.array(cam["C"], np.float32), device=dev)

with torch.no_grad():
    img, _ = render_equirect_fused(means, quats, scales, opac, shc, vm, C, W, H, sh_deg)
ren = (img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
gt = np.asarray(Image.open(cam["image"]).convert("RGB").resize((W, H), Image.LANCZOS))
stack = np.concatenate([ren, gt], 0)   # render on top, GT below
Image.fromarray(stack).save(out_p)
err = np.abs(ren.astype(float) - gt.astype(float)).mean()
print(f"[check] cam idx={cam['idx']} | mean|render-GT|={err:.1f}/255 | saved {out_p}")
