#!/usr/bin/env python3
"""Per-camera pose-graph BA for sliding-window VGGT (the level above global_sim3).

global_sim3 aligns each WINDOW rigidly (one Sim3 per window) and the merge then
picks each shared camera's pose from the FIRST window it appears in. But shared
cameras live in the window OVERLAP = the *tail* of window A (where VGGT drift is
largest) and the *head* of window B. First-window-pick therefore selects the
worst estimate -> seam drift.

This optimizes ONE global pose per unique camera (R_i, C_i) directly, given the
per-window Sim3 from global_sim3:
  - data term : every (window, camera) observation pulls g_i toward that window's
    global pose, WEIGHTED by centrality (margin to the window's ends) so a
    camera is trusted most where it sits mid-window, least at the drifty seam.
  - smoothness: consecutive cameras' relative pose matches the best window's
    observed relative pose (chains the trajectory through the seams).
Centers + rotations only (no feature tracks) — same sparse-360 regime as global_sim3.

API: optimize_pose_graph(raw, win_sim3, ...) -> (names list, R (N,3,3), C (N,3))
  raw: [{"names":[str], "extr":(S,3,4) world2cam, "centers":(S,3)}]
  win_sim3: [(s,R,t)] per window  (X_global = s R X_win + t)
"""
import numpy as np, torch


def _so3_exp(omega):                         # (B,3)->(B,3,3)
    th = omega.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    k = omega / th
    K = torch.zeros(omega.shape[0], 3, 3, device=omega.device, dtype=omega.dtype)
    K[:, 0, 1] = -k[:, 2]; K[:, 0, 2] = k[:, 1]
    K[:, 1, 0] = k[:, 2];  K[:, 1, 2] = -k[:, 0]
    K[:, 2, 0] = -k[:, 1]; K[:, 2, 1] = k[:, 0]
    I = torch.eye(3, device=omega.device, dtype=omega.dtype)[None]
    s, c = torch.sin(th)[..., None], torch.cos(th)[..., None]
    return I + s * K + (1 - c) * (K @ K)


def _so3_log_np(R):
    c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0); th = np.arccos(c)
    if th < 1e-6: return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w / (2 * np.sin(th)) * th


def optimize_pose_graph(raw, win_sim3, iters=4000, lr=0.01, lam_rel=1.0,
                        device="cpu", verbose=True):
    # 1) gather weighted global observations per unique camera
    obs = {}                                   # name -> list of (Rg(3,3), Cg(3), w)
    order = []                                 # global camera order (first appearance)
    for (s, R, t), r in zip(win_sim3, raw):
        names, extr, Cs = r["names"], r["extr"], r["centers"]
        S = len(names)
        for i, n in enumerate(names):
            Rc = extr[i, :, :3]
            Cg = s * (R @ Cs[i]) + t
            Rg = Rc @ R.T                      # world->cam in global frame
            margin = min(i, S - 1 - i) + 1.0   # centrality within window (>=1)
            obs.setdefault(n, [])
            if n not in obs or len(obs[n]) == 0: order.append(n)
            obs[n].append((Rg, Cg, float(margin)))
    names_g = sorted(obs.keys(), key=lambda n: order.index(n) if n in order else 1e9)
    # de-dup order list (order may repeat); rebuild stable order by first window/index
    names_g = []
    seen = set()
    for r in raw:
        for n in r["names"]:
            if n not in seen: seen.add(n); names_g.append(n)
    idx = {n: i for i, n in enumerate(names_g)}
    N = len(names_g)

    # 2) init each camera = max-margin (most central) observation
    R0 = np.zeros((N, 3, 3)); C0 = np.zeros((N, 3))
    for n in names_g:
        best = max(obs[n], key=lambda o: o[2])
        R0[idx[n]] = best[0]; C0[idx[n]] = best[1]

    # 3) relative-pose edges between consecutive cameras, from their best shared window
    #    (a window observing both i and i+1; choose the one maximizing min centrality)
    rel = []                                   # (i, j, dR(3,3), dC(3), weight)
    win_lookup = []
    for (s, R, t), r in zip(win_sim3, raw):
        nm = {n: k for k, n in enumerate(r["names"])}
        win_lookup.append((nm, r, s, R, t))
    for a in range(N - 1):
        na, nb = names_g[a], names_g[a + 1]
        best = None
        for (nm, r, s, R, t) in win_lookup:
            if na in nm and nb in nm:
                ia, ib = nm[na], nm[nb]; Sw = len(r["names"])
                mrg = min(min(ia, Sw - 1 - ia), min(ib, Sw - 1 - ib)) + 1.0
                if best is None or mrg > best[0]:
                    Ra, Rb = r["extr"][ia, :, :3], r["extr"][ib, :, :3]
                    Ca = s * (R @ r["centers"][ia]) + t
                    Cb = s * (R @ r["centers"][ib]) + t
                    dR = (Rb @ R.T) @ (Ra @ R.T).T   # global rel rotation i->i+1
                    best = (mrg, dR, Cb - Ca)
        if best is not None:
            rel.append((a, a + 1, best[1], best[2], best[0]))

    if verbose:
        print(f"[pgo] {N} cameras, {sum(len(v) for v in obs.values())} obs, "
              f"{len(rel)} relative edges")

    # 4) torch optimization
    om = torch.tensor(np.array([_so3_log_np(R0[i]) for i in range(N)]), dtype=torch.float32,
                      device=device, requires_grad=True)
    C = torch.tensor(C0, dtype=torch.float32, device=device, requires_grad=True)
    opt = torch.optim.Adam([om, C], lr=lr)

    # stack data obs
    di, dR, dC, dw = [], [], [], []
    for n in names_g:
        for (Rg, Cg, w) in obs[n]:
            di.append(idx[n]); dR.append(Rg); dC.append(Cg); dw.append(w)
    di = torch.tensor(di, device=device)
    dR = torch.tensor(np.array(dR), dtype=torch.float32, device=device)
    dC = torch.tensor(np.array(dC), dtype=torch.float32, device=device)
    dw = torch.tensor(dw, dtype=torch.float32, device=device).sqrt()
    if rel:
        ri = torch.tensor([e[0] for e in rel], device=device)
        rj = torch.tensor([e[1] for e in rel], device=device)
        rdR = torch.tensor(np.array([e[2] for e in rel]), dtype=torch.float32, device=device)
        rdC = torch.tensor(np.array([e[3] for e in rel]), dtype=torch.float32, device=device)
        rw = torch.tensor([e[4] for e in rel], dtype=torch.float32, device=device).sqrt()
    anchor = torch.tensor(_so3_log_np(R0[0]), dtype=torch.float32, device=device)
    C0t = torch.tensor(C0[0], dtype=torch.float32, device=device)

    for it in range(iters):
        opt.zero_grad()
        Rall = _so3_exp(om)
        # data: rotation (frobenius) + center
        l_dR = (((Rall[di] - dR) ** 2).sum((-1, -2)) * dw ** 2).mean()
        l_dC = (((C[di] - dC) ** 2).sum(-1) * dw ** 2).mean()
        loss = l_dR + l_dC
        if rel:
            pdR = Rall[rj] @ Rall[ri].transpose(1, 2)
            l_rR = (((pdR - rdR) ** 2).sum((-1, -2)) * rw ** 2).mean()
            l_rC = ((((C[rj] - C[ri]) - rdC) ** 2).sum(-1) * rw ** 2).mean()
            loss = loss + lam_rel * (l_rR + l_rC)
        # gauge anchor: pin camera 0
        loss = loss + ((om[0] - anchor) ** 2).sum() + ((C[0] - C0t) ** 2).sum()
        loss.backward(); opt.step()
        if verbose and (it % 1000 == 0 or it == iters - 1):
            print(f"[pgo]  it {it:5d}  loss {loss.item():.6f}")

    with torch.no_grad():
        Rfin = _so3_exp(om).cpu().numpy()
        Cfin = C.cpu().numpy()
    return names_g, Rfin, Cfin
