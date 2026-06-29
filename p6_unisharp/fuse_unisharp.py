#!/usr/bin/env python3
"""Direction 2: fuse several per-frame UniSHARP plys into ONE world frame to extend
the roam range / fill occlusions.

Each UniSHARP ply is a metric 3DGS in its own pano-view (camera) frame. We place
ply_i into a common frame using the VGGT pose of that pano (from pano_cams json):
  world = R_wp^T @ local + s * C_vggt      (s = VGGT->metric scale, --scale)
and rotate each gaussian's quaternion by R_wp^T. Colors/scales/opacities kept.
Without GPS the absolute scale is unknown -> --scale is approximate; this is a
coverage demo, expect some ghosting where views disagree.

Usage:
  fuse_unisharp.py <pano_cams.json> <out.ply> --scale S --plys idx:ply [idx:ply ...]
"""
import sys, json, argparse, numpy as np
from plyfile import PlyData, PlyElement

def quat_mul(q, r):  # (...,4) wxyz
    w0,x0,y0,z0 = q[...,0],q[...,1],q[...,2],q[...,3]
    w1,x1,y1,z1 = r[...,0],r[...,1],r[...,2],r[...,3]
    return np.stack([w0*w1-x0*x1-y0*y1-z0*z1,
                     w0*x1+x0*w1+y0*z1-z0*y1,
                     w0*y1-x0*z1+y0*w1+z0*x1,
                     w0*z1+x0*y1-y0*x1+z0*w1], -1)

def mat2quat(R):  # 3x3 -> wxyz
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t+1.0)*2; w=0.25*s; x=(R[2,1]-R[1,2])/s; y=(R[0,2]-R[2,0])/s; z=(R[1,0]-R[0,1])/s
    elif R[0,0]>R[1,1] and R[0,0]>R[2,2]:
        s=np.sqrt(1+R[0,0]-R[1,1]-R[2,2])*2; w=(R[2,1]-R[1,2])/s; x=0.25*s; y=(R[0,1]+R[1,0])/s; z=(R[0,2]+R[2,0])/s
    elif R[1,1]>R[2,2]:
        s=np.sqrt(1+R[1,1]-R[0,0]-R[2,2])*2; w=(R[0,2]-R[2,0])/s; x=(R[0,1]+R[1,0])/s; y=0.25*s; z=(R[1,2]+R[2,1])/s
    else:
        s=np.sqrt(1+R[2,2]-R[0,0]-R[1,1])*2; w=(R[1,0]-R[0,1])/s; x=(R[0,2]+R[2,0])/s; y=(R[1,2]+R[2,1])/s; z=0.25*s
    return np.array([w,x,y,z])

ap = argparse.ArgumentParser()
ap.add_argument("cams"); ap.add_argument("out")
ap.add_argument("--scale", type=float, default=30.0)
ap.add_argument("--plys", nargs="+", required=True, help="idx:path entries")
a = ap.parse_args()
cams = {c["idx"]: c for c in json.load(open(a.cams))["cameras"]}

merged = {}
props = None
for ent in a.plys:
    idx_s, path = ent.split(":", 1); idx = int(idx_s)
    c = cams[idx]; Rwp = np.array(c["R_wp"], float); C = np.array(c["C"], float) * a.scale
    v = PlyData.read(path)["vertex"]
    if props is None:
        props = [p.name for p in v.properties]
        merged = {p: [] for p in props}
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    world = xyz @ Rwp + C                      # local->world : R_wp^T @ local => xyz @ R_wp
    Rt_q = mat2quat(Rwp.T)
    q = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], 1).astype(np.float64)
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
    qn = quat_mul(np.broadcast_to(Rt_q, q.shape), q)
    for p in props:
        if p == "x": merged[p].append(world[:, 0].astype(np.float32))
        elif p == "y": merged[p].append(world[:, 1].astype(np.float32))
        elif p == "z": merged[p].append(world[:, 2].astype(np.float32))
        elif p == "rot_0": merged[p].append(qn[:, 0].astype(np.float32))
        elif p == "rot_1": merged[p].append(qn[:, 1].astype(np.float32))
        elif p == "rot_2": merged[p].append(qn[:, 2].astype(np.float32))
        elif p == "rot_3": merged[p].append(qn[:, 3].astype(np.float32))
        else: merged[p].append(np.asarray(v[p]).astype(np.float32))
    print(f"  +{idx}: {len(xyz)} gaussians from {path}")

arrs = {p: np.concatenate(merged[p]) for p in props}
N = len(arrs["x"])
verts = np.empty(N, dtype=[(p, "f4") for p in props])
for p in props: verts[p] = arrs[p]
PlyData([PlyElement.describe(verts, "vertex")]).write(a.out)
print(f"[fuse] {len(a.plys)} plys -> {N} gaussians -> {a.out}")
