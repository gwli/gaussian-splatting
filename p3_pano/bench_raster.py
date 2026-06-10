#!/usr/bin/env python3
"""Micro-benchmark: INRIA diff-gaussian-rasterization vs gsplat.rasterization
forward+backward throughput on identical gaussians + camera (pinhole).

Answers "is gsplat 1.5-2x faster" without densification-API differences.
Usage: bench_raster.py [N=100000] [res=1024] [iters=200]
"""
import sys, math, time, torch
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
RES = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
ITERS = int(sys.argv[3]) if len(sys.argv) > 3 else 200
dev = "cuda"; SH_DEG = 3
torch.manual_seed(0)

# shared gaussian params (scene in a box in front of camera at z~4)
means = (torch.rand(N, 3, device=dev) - 0.5) * 4.0
means[:, 2] += 4.0
quats = torch.randn(N, 4, device=dev); quats = torch.nn.functional.normalize(quats)
scales = torch.exp(torch.full((N, 3), -3.0, device=dev) + 0.2*torch.randn(N,3,device=dev))
opac = torch.sigmoid(torch.randn(N, 1, device=dev))
sh = torch.randn(N, (SH_DEG+1)**2, 3, device=dev) * 0.3
sh[:, 0, :] += 0.5

# camera: identity rotation, looking +z; pinhole fov ~ 60deg
fov = math.radians(60); f = (RES/2) / math.tan(fov/2)
K = torch.tensor([[f,0,RES/2],[0,f,RES/2],[0,0,1]], device=dev, dtype=torch.float32)
viewmat = torch.eye(4, device=dev)                 # world->cam = identity

def bench(name, fn):
    for _ in range(10): fn()           # warmup
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(ITERS): fn()
    torch.cuda.synchronize(); dt = (time.time()-t0)/ITERS
    print(f"{name:10s}: {dt*1000:7.2f} ms/iter   {1/dt:7.1f} iter/s")
    return dt

results = {}

# ---- gsplat ----
try:
    from gsplat import rasterization
    def gs():
        m = means.clone().requires_grad_(True)
        col = sh.clone().requires_grad_(True)
        sc = scales.clone().requires_grad_(True); op = opac.clone().requires_grad_(True)
        q = quats.clone().requires_grad_(True)
        out, _, _ = rasterization(m, q, sc, op.squeeze(-1), col,
            viewmat[None], K[None], RES, RES, sh_degree=SH_DEG, render_mode="RGB")
        out.sum().backward()
    results["gsplat"] = bench("gsplat", gs)
except Exception as e:
    print("gsplat FAILED:", repr(e)[:200])

# ---- INRIA ----
try:
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    bg = torch.zeros(3, device=dev)
    # build proj matrix (OpenGL-style) like the repo
    def proj(fovx, fovy, znear=0.01, zfar=100.0):
        t = math.tan(fovx/2); b=-t; r=math.tan(fovy/2)*0  # placeholder
        P = torch.zeros(4,4, device=dev)
        P[0,0]=1/math.tan(fovx/2); P[1,1]=1/math.tan(fovy/2)
        P[2,2]=zfar/(zfar-znear); P[2,3]=-(zfar*znear)/(zfar-znear); P[3,2]=1.0
        return P
    fovx=fovy=fov
    world_view = viewmat.transpose(0,1).contiguous()
    full_proj = (proj(fovx,fovy) @ viewmat).transpose(0,1).contiguous()
    campos = torch.zeros(3, device=dev)
    def inria():
        m = means.clone().requires_grad_(True); sp = torch.zeros_like(m, requires_grad=True)+0
        col = sh.clone().requires_grad_(True)
        sc = scales.clone().requires_grad_(True); op = opac.clone().requires_grad_(True)
        q = quats.clone().requires_grad_(True)
        st = GaussianRasterizationSettings(RES, RES, math.tan(fovx/2), math.tan(fovy/2),
             bg, 1.0, world_view, full_proj, SH_DEG, campos, False, False, False)
        r = GaussianRasterizer(st)
        out, _, _ = r(means3D=m, means2D=sp, shs=col, opacities=op, scales=sc, rotations=q)
        out.sum().backward()
    results["inria"] = bench("inria", inria)
except Exception as e:
    print("inria FAILED:", repr(e)[:300])

if "gsplat" in results and "inria" in results:
    print(f"\nN={N} res={RES}: gsplat is {results['inria']/results['gsplat']:.2f}x "
          f"{'faster' if results['gsplat']<results['inria'] else 'slower'} than INRIA (fwd+bwd)")
