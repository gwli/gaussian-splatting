#!/usr/bin/env python3
"""Static-region PSNR: evaluate two plys on the same test cams, excluding
dynamic pixels (dyn_mask weight<0.5) from the metric.
usage: eval_masked.py <cams.json> <mask_dir> <ply_a> <ply_b>
"""
import sys, os, json, numpy as np, torch
from PIL import Image

REPO = "/w" if os.path.exists("/w/p3_pano") else "/raid/git/gaussian-splatting"
sys.path.insert(0, REPO + "/p3_pano")
from gsplat_equirect import render_equirect_fused
from plyfile import PlyData

cams_json, mask_dir = sys.argv[1], sys.argv[2]
plys = sys.argv[3:5]
W = 1024; H = W // 2; dev = "cuda"
meta = json.load(open(cams_json))
cams = meta["cameras"]
test_idx = list(range(0, len(cams), 8))

def load_ply(p):
    v = PlyData.read(p)["vertex"]; names = v.data.dtype.names
    xyz = torch.tensor(np.stack([v["x"], v["y"], v["z"]], 1), dtype=torch.float32, device=dev)
    sh0 = torch.tensor(np.stack([v[f"f_dc_{i}"] for i in range(3)], 1), dtype=torch.float32, device=dev)[:, None, :]
    nrest = len([n for n in names if n.startswith("f_rest_")])
    fr = np.stack([v[f"f_rest_{i}"] for i in range(nrest)], 1).astype(np.float32)
    shN = torch.tensor(fr.reshape(len(fr), 3, nrest // 3).transpose(0, 2, 1).copy(), device=dev)
    return dict(xyz=xyz, colors=torch.cat([sh0, shN], 1),
                opac=torch.sigmoid(torch.tensor(np.asarray(v["opacity"], np.float32), device=dev)),
                scal=torch.exp(torch.tensor(np.stack([v[f"scale_{i}"] for i in range(3)], 1), dtype=torch.float32, device=dev)),
                quat=torch.tensor(np.stack([v[f"rot_{i}"] for i in range(4)], 1), dtype=torch.float32, device=dev))

for ply in plys:
    G = load_ply(ply)
    ps_full, ps_stat = [], []
    with torch.no_grad():
        for ti in test_idx:
            c = cams[ti]
            R = np.array(c["R_wp"], np.float32); T = np.array(c["T"], np.float32)
            vm = np.eye(4, dtype=np.float32); vm[:3, :3] = R; vm[:3, 3] = T
            img, _ = render_equirect_fused(G["xyz"], G["quat"], G["scal"], G["opac"], G["colors"],
                                           torch.tensor(vm, device=dev),
                                           torch.tensor(np.array(c["C"], np.float32), device=dev), W, H, 3)
            img = img.clamp(0, 1).permute(2, 0, 1)
            gt = torch.tensor(np.asarray(Image.open(os.path.join(REPO, c["image"])).convert("RGB")
                                         .resize((W, H), Image.LANCZOS)), dtype=torch.float32, device=dev).permute(2, 0, 1) / 255
            se = (img - gt) ** 2
            ps_full.append(float(-10 * torch.log10(se.mean())))
            mp = os.path.join(mask_dir, f"pano_{c['idx']:04d}.png")
            wm = torch.tensor(np.asarray(Image.open(mp).convert("L").resize((W, H), Image.BILINEAR)),
                              dtype=torch.float32, device=dev)[None] / 255.0
            stat = wm > 0.5
            ps_stat.append(float(-10 * torch.log10(se[stat.expand_as(se)].mean())))
    print(f"{os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(ply))))}/{ply.split('/')[-4]}: "
          f"full={np.mean(ps_full):.3f}  static-only={np.mean(ps_stat):.3f}")
