#!/usr/bin/env python3
"""T-F3 validation: global Sim3 solve vs sequential pairwise Umeyama on a
synthetic windowed trajectory with per-window arbitrary scale/rot/trans + noise
(mimicking VGGT's per-window metric ambiguity). Reports RMS-to-ground-truth.
"""
import numpy as np
from global_sim3 import optimize_global_sim3

rng = np.random.default_rng(0)


def umeyama(src, dst):                       # dst ~ s R src + t
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    H = d0.T @ s0 / len(src)
    U, D, Vt = np.linalg.svd(H)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var = (s0 ** 2).sum() / len(src)
    s = np.trace(np.diag(D) @ S) / var
    t = mu_d - s * R @ mu_s
    return s, R, t


def rand_sim3(scale_lo=0.3, scale_hi=3.0):
    s = rng.uniform(scale_lo, scale_hi)
    A = rng.standard_normal((3, 3)); Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    t = rng.standard_normal(3) * 5
    return s, Q, t


# ---- ground-truth trajectory: a CLOSED LOOP (drone returns to start) ----
# A planar circle (closed) with mild altitude change — so the last frames are
# spatially near the first, creating loop-closure overlap between the last and
# first windows that sequential chain-alignment cannot exploit.
M = 200
u = np.linspace(0, 2 * np.pi, M, endpoint=False)
P_gt = np.stack([10 * np.cos(u), 10 * np.sin(u), 0.5 * np.sin(2 * u)], 1)
names = [f"f{i:04d}" for i in range(M)]

# ---- overlapping windows that WRAP AROUND the loop (last window overlaps first) ----
WIN, OVL, NOISE = 40, 15, 0.02
windows = []
starts = list(range(0, M, WIN - OVL))
for st in starts:
    idx = [(st + k) % M for k in range(WIN)]          # wrap-around -> loop closure
    s, R, t = rand_sim3()
    g = P_gt[idx]
    local = (np.linalg.inv(R) @ ((g - t).T)).T / s
    local = local + rng.standard_normal(local.shape) * NOISE
    windows.append({"names": [names[i] for i in idx], "centers": local})
print(f"GT {M} cams (closed loop) -> {len(windows)} windows (size {WIN}, overlap {OVL}, wrap-around)")


def reconstruct_sequential(windows):
    """Old approach: align each window to the running global frame via Umeyama."""
    name_to_C, est = {}, {}
    for wi, w in enumerate(windows):
        C = np.asarray(w["centers"]); nm = w["names"]
        if wi == 0:
            s, R, t = 1.0, np.eye(3), np.zeros(3)
        else:
            sh = [(i, n) for i, n in enumerate(nm) if n in name_to_C]
            src = np.array([C[i] for i, _ in sh]); dst = np.array([name_to_C[n] for _, n in sh])
            s, R, t = umeyama(src, dst)
        for i, n in enumerate(nm):
            Cg = s * (R @ C[i]) + t
            name_to_C[n] = Cg; est[n] = Cg
    return est


def reconstruct_global(windows):
    xf = optimize_global_sim3(windows, iters=2000, lr=0.05, verbose=True)
    est = {}
    for w, (s, R, t) in zip(windows, xf):
        C = np.asarray(w["centers"])
        for i, n in enumerate(w["names"]):
            est[n] = s * (R @ C[i]) + t
    return est


def rms_to_gt(est):
    nm = [n for n in names if n in est]
    P = np.array([est[n] for n in nm])
    G = np.array([P_gt[names.index(n)] for n in nm])
    s, R, t = umeyama(P, G)            # align estimate to GT (global Sim3 gauge)
    Pa = (s * (R @ P.T).T + t)
    return np.linalg.norm(Pa - G, axis=1).mean(), np.linalg.norm(Pa - G, axis=1).max()


seq = reconstruct_sequential(windows)
glob = reconstruct_global(windows)
sm, sx = rms_to_gt(seq)
gm, gx = rms_to_gt(glob)
print(f"\nsequential Umeyama : mean err {sm:.4f}  max {sx:.4f}")
print(f"global Sim3 solve  : mean err {gm:.4f}  max {gx:.4f}")
print(f"=> global is {sm/gm:.1f}x lower mean error" if gm > 0 else "")
