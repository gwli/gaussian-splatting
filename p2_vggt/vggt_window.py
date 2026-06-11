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

# ---- Pass 1: run every window, keep raw per-window reconstructions ----
starts = list(range(0, max(1, N - OVL), WIN - OVL))
raw = []
for wi, st in enumerate(starts):
    wp = paths[st:st + WIN]
    if not wp: continue
    names = [os.path.basename(p) for p in wp]
    extr, conf, pts = run_window(wp)
    S = extr.shape[0]
    Cs = np.array([center(extr[i, :, :3], extr[i, :, 3]) for i in range(S)])
    raw.append(dict(names=names, extr=extr, conf=conf, pts=pts, centers=Cs))
    print(f"  window {wi}: {S} frames")

# ---- Pass 2: GLOBAL Sim3 pose-graph alignment (T-F3). Spanning-tree init from
# adjacent windows + loop-closure refinement over ALL window pairs. Falls back to
# the legacy sequential Umeyama if the solver is unavailable. ----
xforms = None
try:
    from global_sim3 import optimize_global_sim3
    xforms = optimize_global_sim3(
        [{"names": r["names"], "centers": r["centers"]} for r in raw],
        iters=3000, lr=0.02, verbose=True)
except Exception as e:
    print(f"  [global_sim3 unavailable: {e}] -> sequential Umeyama fallback")
    xforms, name_to_C = [], {}
    for wi, r in enumerate(raw):
        names, Cs = r["names"], r["centers"]
        if wi == 0:
            s, R, t = 1.0, np.eye(3), np.zeros(3)
        else:
            sh = [(i, n) for i, n in enumerate(names) if n in name_to_C]
            if len(sh) < 3:
                s, R, t = xforms[-1]
            else:
                src = np.array([Cs[i] for i, _ in sh]); dst = np.array([name_to_C[n] for _, n in sh])
                s, R, t = umeyama(src, dst)
        for i, n in enumerate(names):
            name_to_C[n] = s * (R @ Cs[i]) + t
        xforms.append((s, R, t))

# ---- Pass 3: apply per-window Sim3, dedup cameras, accumulate conf-masked pts ----
g_names, g_R, g_t, g_pts = [], [], [], []
seen = set()
for r, (s, R, t) in zip(raw, xforms):
    names, extr, conf, pts, Cs = r["names"], r["extr"], r["conf"], r["pts"], r["centers"]
    for i, n in enumerate(names):
        if n in seen: continue
        seen.add(n)
        Rc = extr[i, :, :3]
        Cg = s * (R @ Cs[i]) + t
        Rcg = Rc @ R.T
        g_names.append(n); g_R.append(Rcg); g_t.append(-Rcg @ Cg)
    m = conf >= CONF
    P = pts[m]
    if P.shape[0]:
        g_pts.append(s * (R @ P.T).T + t)

print(f"merged: {len(g_names)} cameras, {sum(p.shape[0] for p in g_pts)} raw points")
np.savez(os.path.join(scene, "vggt_window_merged.npz"),
         names=np.array(g_names), R=np.array(g_R), t=np.array(g_t),
         pts=np.concatenate(g_pts)[:300000] if g_pts else np.zeros((0,3)))
print("saved vggt_window_merged.npz (poses + global point cloud)")
