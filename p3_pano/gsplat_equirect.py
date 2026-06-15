#!/usr/bin/env python3
"""T-F7: a SPHERICAL (equirectangular) rasterizer built on gsplat's fast CUDA
tile compositor.

gsplat is pinhole-only, so direct-pano had to either use the OmniGS LONLAT
kernel (INRIA-class per-pixel speed) or render 6 cube faces (T-F6, 3x pixels ->
slower). This module instead does the equirect PROJECTION in autograd-PyTorch
(analytic Jacobian for the 2D covariance) and then hands the projected 2D
gaussians to gsplat's `rasterize_to_pixels` — gsplat's fast tile alpha-compositor
is projection-agnostic. Result: ONE equirect pass (like LONLAT) but with gsplat's
optimized rasterization kernel. Densification reuses gsplat DefaultStrategy
(the info dict matches what it consumes).

Convention matches OmniGS auxiliary.h point3ToLonlatPixel exactly:
  lon = atan2(x, z),  lat = asin(y/r)
  px = (lon/pi + 1)*W/2,  py = (lat/(pi/2) + 1)*H/2     (view space)

Known v1 limitations (documented, not hidden):
  - seam at lon=+-pi: a gaussian whose footprint crosses x=0/x=W is not wrapped.
  - poles (lat=+-pi/2): the equirect Jacobian diverges; we clamp rho and cull
    degenerate gaussians, so the very top/bottom rows may be slightly soft.
"""
import os, math, torch
from gsplat.cuda._wrapper import (isect_tiles, isect_offset_encode,
                                  rasterize_to_pixels, spherical_harmonics,
                                  fully_fused_projection)


def render_equirect_fused(means, quats, scales, opacities, sh_coeffs, viewmat,
                          cam_center, W, H, sh_degree, tile_size=16):
    """T-F8: native FUSED equirect rasterizer. Uses gsplat's fully_fused_projection
    with camera_model="equirect" (our new CUDA projection + analytic VJP), then
    gsplat's fast tile compositor. All-CUDA fwd+bwd, one equirect pass."""
    dev = means.device
    vm = viewmat[None]                                  # (1,4,4) world->cam
    K = torch.eye(3, device=dev)[None]                  # dummy; equirect ignores K
    radii, means2d, depths, conics, _ = fully_fused_projection(
        means, None, quats, scales, vm, K, W, H,
        eps2d=0.3, near_plane=0.01, far_plane=1e10, packed=False,
        camera_model="equirect")                        # (1,N,2)/(1,N,2)/(1,N)/(1,N,3)
    dirs = torch.nn.functional.normalize(means - cam_center[None], dim=-1)
    colors = (spherical_harmonics(sh_degree, dirs, sh_coeffs) + 0.5).clamp_min(0.0)[None]
    op = opacities[None]
    tw, th = math.ceil(W / tile_size), math.ceil(H / tile_size)
    _, isect_ids, flatten_ids = isect_tiles(means2d, radii, depths, tile_size, tw, th,
                                            packed=False, n_images=1,
                                            conics=conics, opacities=op)
    offs = isect_offset_encode(isect_ids, 1, tw, th)
    img, _ = rasterize_to_pixels(means2d, conics, colors, op, W, H, tile_size, offs, flatten_ids)
    info = {"means2d": means2d, "radii": radii, "width": W, "height": H,
            "n_cameras": 1, "gaussian_ids": None}
    return img[0], info


def quat_to_rotmat(q):                       # (N,4) wxyz -> (N,3,3)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    N = q.shape[0]
    R = torch.empty(N, 3, 3, device=q.device, dtype=q.dtype)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - w * z); R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y); R[:, 2, 1] = 2 * (y * z + w * x); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def _project_equirect(means, quats, scales, viewmat, W, H, eps, blur):
    """Pure-torch equirect projection -> (means2d, conics, radii). torch.compile-able."""
    R = viewmat[:3, :3]; t = viewmat[:3, 3]
    mu = means @ R.T + t                                   # (N,3) view space
    x, y, z = mu[:, 0], mu[:, 1], mu[:, 2]
    r2 = (mu * mu).sum(-1); r = r2.clamp_min(eps).sqrt()
    rho2 = (x * x + z * z).clamp_min(eps); rho = rho2.sqrt()

    lon = torch.atan2(x, z)
    lat = torch.asin((y / r).clamp(-1 + 1e-6, 1 - 1e-6))
    px = (lon / math.pi + 1.0) * (W / 2.0)
    py = (lat / (math.pi / 2.0) + 1.0) * (H / 2.0)
    means2d = torch.stack([px, py], -1)                    # (N,2) pixels

    cwl, chl = W / (2 * math.pi), H / math.pi
    z0 = torch.zeros_like(x)
    J = torch.stack([
        torch.stack([cwl * z / rho2, z0, cwl * (-x / rho2)], -1),
        torch.stack([chl * (-x * y / (rho * r2)), chl * (rho / r2), chl * (-z * y / (rho * r2))], -1),
    ], 1)                                                  # (N,2,3)
    M = R[None] @ quat_to_rotmat(quats)                    # (N,3,3)
    cov_v = M @ (scales[:, :, None] ** 2 * M.transpose(1, 2))
    cov2d = J @ cov_v @ J.transpose(1, 2)
    a = cov2d[:, 0, 0] + blur; b = cov2d[:, 0, 1]; c = cov2d[:, 1, 1] + blur
    det = (a * c - b * b).clamp_min(eps)
    conics = torch.stack([c / det, -b / det, a / det], -1)
    mid = 0.5 * (a + c)
    lam = mid + (mid * mid - det).clamp_min(0).sqrt()
    rad = (3.0 * lam.clamp_min(0).sqrt()).ceil()
    valid = (r > eps) & (rho > 1e-3) & torch.isfinite(rad)
    rad = torch.where(valid, rad, torch.zeros_like(rad))
    radii = torch.stack([rad, rad], -1).to(torch.int32)
    return means2d, conics, radii, r


_proj_fn = _project_equirect
if os.environ.get("GSPLAT_EQUIRECT_COMPILE") == "1":
    _proj_fn = torch.compile(_project_equirect, dynamic=True)


def render_equirect(means, quats, scales, opacities, sh_coeffs, viewmat,
                    cam_center, W, H, sh_degree, tile_size=16, eps=1e-6, blur=0.3):
    """means(N,3) world; quats(N,4) wxyz; scales(N,3); opacities(N,);
    sh_coeffs(N,K,3); viewmat(4,4) world->cam. Returns (img(H,W,3), info)."""
    means2d, conics, radii, r = _proj_fn(means, quats, scales, viewmat, W, H, eps, blur)
    dirs = torch.nn.functional.normalize(means - cam_center[None], dim=-1)
    colors = (spherical_harmonics(sh_degree, dirs, sh_coeffs) + 0.5).clamp_min(0.0)  # (N,3)

    # add camera dim C=1
    m2d, cn, col = means2d[None], conics[None], colors[None]
    op, rd, dp = opacities[None], radii[None], r[None]
    tw, th = math.ceil(W / tile_size), math.ceil(H / tile_size)
    _, isect_ids, flatten_ids = isect_tiles(m2d, rd, dp, tile_size, tw, th,
                                            packed=False, n_images=1,
                                            conics=cn, opacities=op)
    offs = isect_offset_encode(isect_ids, 1, tw, th)
    img, alpha = rasterize_to_pixels(m2d, cn, col, op, W, H, tile_size, offs, flatten_ids)
    info = {"means2d": m2d, "radii": rd, "width": W, "height": H, "n_cameras": 1,
            "gaussian_ids": None}   # non-packed path doesn't use it; key required by DefaultStrategy
    return img[0], info                                    # (H,W,3)
