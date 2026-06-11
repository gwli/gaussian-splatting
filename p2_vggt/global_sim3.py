#!/usr/bin/env python3
"""T-F3: global Sim3 pose-graph alignment for sliding-window VGGT (finishes T-D4).

vggt_window.py merges per-window VGGT reconstructions by aligning each window to
the running global frame with a *sequential, pairwise* Umeyama Sim3. For windows
that form a pure chain this is already near-optimal; but when the flight revisits
a place (loop closure) non-adjacent windows share frames, and greedy sequential
alignment can't use those constraints -> accumulated drift (scale 1.23 -> 0.026
across 7 windows in the T-D4 run).

This solves a global Sim3 **pose graph**: per-window absolute Sim3 {S_w}
(S_0 = I) chosen to agree with every pairwise *relative* Sim3 measurement
S_ab (estimated by Umeyama on the frames shared by windows a,b). Because the
loss is on relative Sim3s (whose scale is pinned by the measurement), there is
no scale-collapse mode — unlike a naive absolute-position loss. Loop closures
are used automatically (any window pair sharing >=3 frames adds an edge).

Operates on camera centers only (no feature tracks) -> works on our sparse
360-derived data, where classic feature BA can't get inliers (see T-F4).

API: optimize_global_sim3(windows, iters, lr) -> list[(s, R(3x3), t(3))]
     windows: [{"names": [str], "centers": (S,3) array}]   local camera centers
"""
import numpy as np
import torch


def umeyama_np(src, dst):                    # dst ~ s R src + t  (src,dst: Nx3)
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


def _so3_log_np(R):                          # 3x3 -> axis-angle (3,)
    c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    th = np.arccos(c)
    if th < 1e-6:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w / (2 * np.sin(th)) * th


def _compose(A, B):                          # apply B then A: X -> A(B(X))
    sA, RA, tA = A; sB, RB, tB = B
    return (sA * sB, RA @ RB, sA * (RA @ tB) + tA)


def _so3_exp(omega):                         # (B,3) -> (B,3,3)
    theta = omega.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    k = omega / theta
    K = torch.zeros(omega.shape[0], 3, 3, device=omega.device, dtype=omega.dtype)
    K[:, 0, 1] = -k[:, 2]; K[:, 0, 2] = k[:, 1]
    K[:, 1, 0] = k[:, 2];  K[:, 1, 2] = -k[:, 0]
    K[:, 2, 0] = -k[:, 1]; K[:, 2, 1] = k[:, 0]
    I = torch.eye(3, device=omega.device, dtype=omega.dtype)[None]
    s, c = torch.sin(theta)[..., None], torch.cos(theta)[..., None]
    return I + s * K + (1 - c) * (K @ K)


def optimize_global_sim3(windows, iters=3000, lr=0.02, device="cpu", verbose=True,
                         min_shared=3):
    """Returns per-window (s, R, t) with X_global = s * R @ X_win + t."""
    W = len(windows)
    by_win = [{n: np.asarray(c, np.float64) for n, c in zip(w["names"], w["centers"])}
              for w in windows]

    # relative Sim3 measurements S_ab: maps window-b local -> window-a local
    # (so that a-local center ~ S_ab applied to b-local center on shared frames).
    edges = []
    for a in range(W):
        for b in range(a + 1, W):
            shared = [n for n in by_win[a] if n in by_win[b]]
            if len(shared) < min_shared:
                continue
            src = np.array([by_win[b][n] for n in shared])   # b-local
            dst = np.array([by_win[a][n] for n in shared])   # a-local
            s, R, t = umeyama_np(src, dst)
            edges.append((a, b, s, R, t, len(shared)))
    n_loop = sum(1 for e in edges if e[1] - e[0] > 1)
    if verbose:
        print(f"[gsim3] {W} windows, {len(edges)} edges "
              f"({n_loop} loop-closure / non-adjacent)")
    if not edges:
        return [(1.0, np.eye(3), np.zeros(3))] * W

    # --- initialize absolute Sim3 from a spanning tree of ADJACENT edges
    # (= the sequential chain). Good init; the optimizer then refines using the
    # extra loop-closure edges. Without this, zero-init sticks in a bad minimum.
    rel = {(a, b): (s, R, t_) for a, b, s, R, t_, _ in edges}
    init = {0: (1.0, np.eye(3), np.zeros(3))}
    for w in range(1, W):
        if (w - 1, w) in rel:                       # S_w = S_{w-1} ∘ S_rel(w-1<-w)
            init[w] = _compose(init[w - 1], rel[(w - 1, w)])
        else:
            init[w] = init.get(w - 1, (1.0, np.eye(3), np.zeros(3)))
    ls0 = np.array([np.log(max(init[w][0], 1e-6)) for w in range(W)], np.float32)
    om0 = np.array([_so3_log_np(init[w][1]) for w in range(W)], np.float32)
    t0 = np.array([init[w][2] for w in range(W)], np.float32)

    log_s = torch.tensor(ls0, device=device, requires_grad=True)
    omega = torch.tensor(om0, device=device, requires_grad=True)
    t = torch.tensor(t0, device=device, requires_grad=True)
    opt = torch.optim.Adam([log_s, omega, t], lr=lr)

    # pre-stack edge measurements
    ea = torch.tensor([e[0] for e in edges], device=device)
    eb = torch.tensor([e[1] for e in edges], device=device)
    m_s = torch.tensor([e[2] for e in edges], dtype=torch.float32, device=device)
    m_R = torch.tensor(np.array([e[3] for e in edges]), dtype=torch.float32, device=device)
    m_t = torch.tensor(np.array([e[4] for e in edges]), dtype=torch.float32, device=device)
    wgt = torch.tensor([e[5] for e in edges], dtype=torch.float32, device=device).sqrt()
    I3 = torch.eye(3, device=device)

    def sim3(idx):
        s = torch.exp(log_s)
        s = torch.where(torch.arange(W, device=device) == 0, torch.ones_like(s), s)
        R = _so3_exp(omega)
        R = torch.where((torch.arange(W, device=device) == 0)[:, None, None], I3[None], R)
        tt = torch.where((torch.arange(W, device=device) == 0)[:, None],
                         torch.zeros_like(t), t)
        return s[idx], R[idx], tt[idx]

    for it in range(iters):
        opt.zero_grad()
        sa, Ra, ta = sim3(ea)
        sb, Rb, tb = sim3(eb)
        # predicted relative b->a : S_a^{-1} ∘ S_b   (maps b-local -> a-local)
        # S_a^{-1}: s=1/sa, R=Ra^T, t=-(1/sa)Ra^T ta
        sai = 1.0 / sa; Rai = Ra.transpose(1, 2); tai = -(sai[:, None]) * torch.einsum("bij,bj->bi", Rai, ta)
        # compose S_a^{-1} ∘ S_b : s = sai*sb, R = Rai@Rb, t = sai*Rai@tb + tai
        ps = sai * sb
        pR = Rai @ Rb
        pt = sai[:, None] * torch.einsum("bij,bj->bi", Rai, tb) + tai
        # residual vs measurement
        loss_s = ((torch.log(ps) - torch.log(m_s)) * wgt) ** 2
        loss_R = (((pR - m_R) ** 2).sum((-1, -2)) * wgt ** 2)
        loss_t = (((pt - m_t) ** 2).sum(-1) * wgt ** 2)
        loss = (loss_s + loss_R + loss_t).mean()
        loss.backward()
        opt.step()
        if verbose and (it % 1000 == 0 or it == iters - 1):
            print(f"[gsim3]  it {it:5d}  loss {loss.item():.5f}")

    with torch.no_grad():
        s = torch.exp(log_s); s[0] = 1.0
        R = _so3_exp(omega); R[0] = I3
        tt = t.clone(); tt[0] = 0
        return [(float(s[w]), R[w].cpu().numpy(), tt[w].cpu().numpy()) for w in range(W)]
