#!/usr/bin/env python3
"""T-D4: run VGGT on >300-frame scenes via overlapping sliding windows.

VGGT attends across all frames at once (O(N^2) memory, ~300-frame ceiling on
80GB). This processes the scene in overlapping windows, then aligns each window
to the first via a Sim3 (Umeyama on shared-frame camera centers) and merges
into ONE COLMAP model (sparse/0 + points.ply) usable by 3DGS / make_pano_dataset.

Usage: vggt_window.py <scene_dir> [win=250] [overlap=50] [conf=1.5]
  reads <scene_dir>/images, writes <scene_dir>/sparse/0
"""
import sys, os, glob, numpy as np, torch
sys.path.insert(0, "/workspace/gaussian-splatting/p2_vggt/vggt")
import torch.nn.functional as F
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images_square

scene = sys.argv[1]
WIN  = int(sys.argv[2]) if len(sys.argv) > 2 else 250
OVL  = int(sys.argv[3]) if len(sys.argv) > 3 else 50
CONF = float(sys.argv[4]) if len(sys.argv) > 4 else 1.5
RES = 518
dev = "cuda"; dtype = torch.bfloat16

paths = sorted(glob.glob(os.path.join(scene, "images", "*")))
N = len(paths)
print(f"{N} images | window={WIN} overlap={OVL}")

model = VGGT()
_wpath = next(p for p in ["/wcache/model.pt", "/wcache/hub/checkpoints/model.pt",
                          "/wsrc/model.pt"] if os.path.exists(p))
model.load_state_dict(torch.load(_wpath, map_location="cpu"))
model.eval().to(dev)

def run_window(win_paths):
    imgs, coords = load_and_preprocess_images_square(win_paths, 1024)
    imgs = imgs.to(dev)
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
        im = F.interpolate(imgs, size=(RES, RES), mode="bilinear", align_corners=False)[None]
        toks, ps = model.aggregator(im)
        pose_enc = model.camera_head(toks)[-1]
        extr, intr = pose_encoding_to_extri_intri(pose_enc, im.shape[-2:])
        depth, conf = model.depth_head(toks, im, ps)
    extr = extr.squeeze(0).cpu().numpy()        # (S,3,4) world2cam
    intr = intr.squeeze(0).cpu().numpy()
    depth = depth.squeeze(0).cpu().numpy()
    conf = conf.squeeze(0).cpu().numpy()
    pts = unproject_depth_map_to_point_map(depth, extr, intr)   # (S,H,W,3) world
    return extr, conf, pts

def center(R, t):                # cam center in world
    return -R.T @ t

def umeyama(src, dst):           # sim3: dst ~ s R src + t  (src,dst: Nx3)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    H = d0.T @ s0 / len(src)
    U, D, Vt = np.linalg.svd(H)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0: S[2, 2] = -1
    R = U @ S @ Vt
    var = (s0 ** 2).sum() / len(src)
    s = np.trace(np.diag(D) @ S) / var
    t = mu_d - s * R @ mu_s
    return s, R, t

# global accumulators
g_names, g_R, g_t, g_intr = [], [], [], []      # per-image (world2cam in global frame)
g_pts, g_rgb = [], []
seen = set()
name_to_C = {}                                   # global camera centers (for alignment)

starts = list(range(0, max(1, N - OVL), WIN - OVL))
for wi, st in enumerate(starts):
    wp = paths[st:st + WIN]
    if not wp: continue
    names = [os.path.basename(p) for p in wp]
    extr, conf, pts = run_window(wp)
    S = extr.shape[0]
    Cs = np.array([center(extr[i, :, :3], extr[i, :, 3]) for i in range(S)])

    if wi == 0:
        s, R, t = 1.0, np.eye(3), np.zeros(3)
    else:
        shared = [(i, n) for i, n in enumerate(names) if n in name_to_C]
        if len(shared) < 3:
            print(f"  window {wi}: only {len(shared)} shared frames, skipping (gap too big)")
            continue
        src = np.array([Cs[i] for i, _ in shared])
        dst = np.array([name_to_C[n] for _, n in shared])
        s, R, t = umeyama(src, dst)
        err = np.linalg.norm((s * (R @ src.T).T + t) - dst, axis=1).mean()
        print(f"  window {wi}: aligned on {len(shared)} frames, sim3 s={s:.3f} resid={err:.4f}")

    # transform + accumulate cameras
    for i, n in enumerate(names):
        Rc, tc = extr[i, :, :3], extr[i, :, 3]
        C = Cs[i]
        Cg = s * (R @ C) + t
        Rcg = Rc @ R.T
        tcg = -Rcg @ Cg
        name_to_C[n] = Cg
        if n in seen: continue
        seen.add(n)
        g_names.append(n); g_R.append(Rcg); g_t.append(tcg)
    # transform + accumulate points (conf-masked)
    H, Wd = conf.shape[1], conf.shape[2]
    m = conf >= CONF
    P = pts[m]                                   # (K,3)
    if P.shape[0]:
        Pg = (s * (R @ P.T).T + t)
        g_pts.append(Pg)

print(f"merged: {len(g_names)} cameras, {sum(p.shape[0] for p in g_pts)} raw points")
np.savez(os.path.join(scene, "vggt_window_merged.npz"),
         names=np.array(g_names), R=np.array(g_R), t=np.array(g_t),
         pts=np.concatenate(g_pts)[:300000] if g_pts else np.zeros((0,3)))
print("saved vggt_window_merged.npz (poses + global point cloud)")
