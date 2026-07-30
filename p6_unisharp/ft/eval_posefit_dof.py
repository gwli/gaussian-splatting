#!/usr/bin/env python3
"""Decisive diagnostic: is the remaining error geometric (pose) or photometric
(scene not static)?

Freeze a trained model and optimise ONLY the held-out camera's own pose against
its GT image. The recovered PSNR upper-bounds what better pose estimation could
buy; whatever stays missing is content the model cannot explain at any pose
(moving water/vegetation/traffic, changing sun, exposure drift).

usage: eval_posefit.py <cams.json> <test_idx.txt> <ply> [W=2048] [steps=300]
"""
import sys, os, json
import numpy as np, torch
from PIL import Image

sys.path.insert(0, os.getcwd() + "/p3_pano")
from gsplat_equirect import render_equirect_fused
from plyfile import PlyData

cams_json, tif, ply = sys.argv[1], sys.argv[2], sys.argv[3]
W = int(sys.argv[4]) if len(sys.argv) > 4 else 2048
STEPS = int(sys.argv[5]) if len(sys.argv) > 5 else 300
DOF = sys.argv[6] if len(sys.argv) > 6 else "both"   # both | rot | trans | exp | rot+exp | all
# "exp" fits a per-image affine on colour (per-channel gain + bias) instead of a pose.
# The drone runs auto-exposure -- §26 already had to fix an AE seam in the rig chain --
# so a per-frame photometric offset is a live alternative explanation for what pose
# fitting has been absorbing. Comparing the two tells us which one the residual is.
H = W // 2; dev = "cuda"

v = PlyData.read(ply)["vertex"]; names = v.data.dtype.names
nrest = len([n for n in names if n.startswith("f_rest_")])
fr = np.stack([v[f"f_rest_{i}"] for i in range(nrest)], 1).astype(np.float32)
xyz = torch.tensor(np.stack([v["x"], v["y"], v["z"]], 1), dtype=torch.float32, device=dev)
colors = torch.cat([torch.tensor(np.stack([v[f"f_dc_{i}"] for i in range(3)], 1), dtype=torch.float32, device=dev)[:, None, :],
                    torch.tensor(fr.reshape(len(fr), 3, nrest // 3).transpose(0, 2, 1).copy(), device=dev)], 1)
opac = torch.sigmoid(torch.tensor(np.asarray(v["opacity"], np.float32), device=dev))
scal = torch.exp(torch.tensor(np.stack([v[f"scale_{i}"] for i in range(3)], 1), dtype=torch.float32, device=dev))
quat = torch.tensor(np.stack([v[f"rot_{i}"] for i in range(4)], 1), dtype=torch.float32, device=dev)

def so3exp(w):
    th = w.norm() + 1e-12; k = w / th; z = torch.zeros((), device=w.device)
    K = torch.stack([torch.stack([z, -k[2], k[1]]), torch.stack([k[2], z, -k[0]]), torch.stack([-k[1], k[0], z])])
    return torch.eye(3, device=w.device) + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)

def psnr(a, b): return float(-10 * torch.log10(((a - b) ** 2).mean()))

cams = json.load(open(cams_json))["cameras"]
hold = {int(l) for l in open(tif) if l.strip()}
test = [c for c in cams if c["idx"] in hold]
before, after, drot, dtrans, dexp = [], [], [], [], []
for c in test:
    R0 = torch.tensor(np.array(c["R_wp"], np.float32), device=dev)
    C0 = torch.tensor(np.array(c["C"], np.float32), device=dev)
    gt = torch.tensor(np.asarray(Image.open(c["image"]).convert("RGB").resize((W, H), Image.LANCZOS)),
                      dtype=torch.float32, device=dev).permute(2, 0, 1) / 255
    dr = torch.zeros(3, device=dev, requires_grad=True)
    dt = torch.zeros(3, device=dev, requires_grad=True)
    eg = torch.ones(3, device=dev, requires_grad=True)     # per-channel gain
    eb = torch.zeros(3, device=dev, requires_grad=True)    # per-channel bias
    groups = []
    if DOF in ("both", "rot", "rot+exp", "all"):   groups.append({"params": [dr], "lr": 3e-4})
    if DOF in ("both", "trans", "all"):            groups.append({"params": [dt], "lr": 3e-2})
    if DOF in ("exp", "rot+exp", "all"):           groups.append({"params": [eg, eb], "lr": 3e-3})
    opt = torch.optim.Adam(groups)
    def render():
        Rp = so3exp(dr) @ R0; Cp = C0 + dt
        vm = torch.cat([torch.cat([Rp, -(Rp @ Cp)[:, None]], 1), torch.tensor([[0, 0, 0, 1.0]], device=dev)], 0)
        img, _ = render_equirect_fused(xyz, quat, scal, opac, colors, vm, Cp, W, H, 3)
        img = img.permute(2, 0, 1)
        if DOF in ("exp", "rot+exp", "all"):
            img = img * eg[:, None, None] + eb[:, None, None]
        return img.clamp(0, 1)
    with torch.no_grad():
        before.append(psnr(render(), gt))
    for _ in range(STEPS):
        loss = (render() - gt).abs().mean()
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        after.append(psnr(render(), gt))
        drot.append(float(torch.rad2deg(dr.norm()))); dtrans.append(float(dt.norm()))
        dexp.append(float((eg - 1).abs().mean() + eb.abs().mean()))
b, a = np.array(before), np.array(after)
print(f"held-out views: {len(test)}  |  dof={DOF}")
print(f"PSNR before pose fit: {b.mean():.3f}   after: {a.mean():.3f}   gain: {a.mean()-b.mean():+.3f} dB")
print(f"pose correction applied: rot med={np.median(drot):.3f} deg   trans med={np.median(dtrans):.3f} m"
      f"   exposure med={np.median(dexp):.4f}")
