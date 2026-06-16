#!/usr/bin/env python3
"""task_sr: shared SR method library — model loaders, tiled inference, and
equirect<->cubemap projection (torch grid_sample, distortion-aware).

Methods dispatched by name:
  lanczos        : classical cv2 Lanczos upscale + unsharp (M0 baseline)
  rrdbnet        : Real-ESRGAN RRDBNet on the equirect directly (M1)
  rrdbnet-cube   : RRDBNet on 6 cubemap faces, recombined (M2, 360-aware)
  swinir         : SwinIR transformer SISR on the equirect (M3)
  swinir-cube    : SwinIR on cubemap faces (M3 + 360-aware)
"""
import os, sys, math
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "vendor"))

# ----- RRDBNet (reuse the one in realesrgan_infer.py) -----
sys.path.insert(0, os.path.dirname(HERE))               # p5_sr/
from realesrgan_infer import RRDBNet                      # noqa: E402

def load_rrdbnet(weights, scale, dev):
    nb = 6 if "anime" in os.path.basename(weights) else 23
    m = RRDBNet(scale=scale, nb=nb).to(dev).eval()
    sd = torch.load(weights, map_location="cpu", weights_only=False)
    sd = sd.get("params_ema", sd.get("params", sd))
    m.load_state_dict(sd, strict=True)
    return m

def load_swinir(weights, scale, dev):
    from network_swinir import SwinIR
    # classical-SR DF2K SwinIR-M config (matches 001_classicalSR_*_x{2,4}.pth)
    m = SwinIR(upscale=scale, in_chans=3, img_size=64, window_size=8,
               img_range=1.0, depths=[6, 6, 6, 6, 6, 6], embed_dim=180,
               num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2,
               upsampler="pixelshuffle", resi_connection="1conv").to(dev).eval()
    sd = torch.load(weights, map_location="cpu", weights_only=False)
    sd = sd.get("params", sd.get("params_ema", sd))
    m.load_state_dict(sd, strict=True)
    return m, 8  # window size (for padding)

# ----- tiled SISR forward -----
@torch.no_grad()
def _forward(model, x, fp16, win=0):
    # pad to multiple of `win` (SwinIR needs window-divisible input)
    _, _, h, w = x.shape
    if win:
        ph, pw = (win - h % win) % win, (win - w % win) % win
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    with torch.autocast("cuda", enabled=fp16):
        y = model(x).float()
    return y, (h, w)

@torch.no_grad()
def sisr(model, img, scale, tile, pad, dev, fp16, win=0):
    """img: HxWx3 float[0,1] -> sH x sW x3"""
    x = torch.from_numpy(img.transpose(2, 0, 1))[None].to(dev)
    if fp16: x = x.half()
    _, _, H, W = x.shape
    if tile <= 0:
        y, (h, w) = _forward(model, x, fp16, win)
        return y[:, :, :h * scale, :w * scale].clamp(0, 1)[0].cpu().float().numpy().transpose(1, 2, 0)
    out = torch.zeros(1, 3, H * scale, W * scale, device=dev, dtype=torch.float32)
    for y0 in range(0, H, tile):
        for x0 in range(0, W, tile):
            y1, x1 = min(y0 + tile, H), min(x0 + tile, W)
            ya, xa = max(0, y0 - pad), max(0, x0 - pad)
            yb, xb = min(H, y1 + pad), min(W, x1 + pad)
            patch = x[:, :, ya:yb, xa:xb]
            sp, _ = _forward(model, patch, fp16, win)
            sp = sp[:, :, :(yb - ya) * scale, :(xb - xa) * scale]
            ty0, tx0 = (y0 - ya) * scale, (x0 - xa) * scale
            th, tw = (y1 - y0) * scale, (x1 - x0) * scale
            out[:, :, y0 * scale:y1 * scale, x0 * scale:x1 * scale] = \
                sp[:, :, ty0:ty0 + th, tx0:tx0 + tw]
    return out.clamp(0, 1)[0].cpu().numpy().transpose(1, 2, 0)

def lanczos(img, scale):
    import cv2
    H, W = img.shape[:2]
    up = cv2.resize(img, (W * scale, H * scale), interpolation=cv2.INTER_LANCZOS4)
    blur = cv2.GaussianBlur(up, (0, 0), 1.0)
    return np.clip(up * 1.5 - blur * 0.5, 0, 1)   # unsharp

# ================= equirect <-> cubemap (torch) =================
# faces order: +X(right) -X(left) +Y(up) -Y(down) +Z(front) -Z(back)
def _face_dirs(face, res, dev):
    a = torch.linspace(-1, 1, res, device=dev)
    yy, xx = torch.meshgrid(a, a, indexing="ij")
    o = torch.ones_like(xx)
    if   face == 0: d = torch.stack([o, -yy, -xx], -1)   # +X
    elif face == 1: d = torch.stack([-o, -yy, xx], -1)   # -X
    elif face == 2: d = torch.stack([xx, o, yy], -1)     # +Y
    elif face == 3: d = torch.stack([xx, -o, -yy], -1)   # -Y
    elif face == 4: d = torch.stack([xx, -yy, o], -1)    # +Z
    else:           d = torch.stack([-xx, -yy, -o], -1)  # -Z
    return F.normalize(d, dim=-1)

def equirect_to_cubemap(eq, res):
    """eq: 1x3xHxW -> 1x3x(res)x(6*res) faces side by side"""
    dev = eq.device
    faces = []
    for f in range(6):
        d = _face_dirs(f, res, dev)                       # res,res,3
        lon = torch.atan2(d[..., 0], d[..., 2])           # -pi..pi
        lat = torch.asin(d[..., 1].clamp(-1, 1))          # -pi/2..pi/2
        gx = lon / math.pi                                # -1..1
        gy = -2 * lat / math.pi                           # -1..1 (lat+ -> up -> grid y-)
        grid = torch.stack([gx, gy], -1)[None]
        faces.append(F.grid_sample(eq, grid, mode="bilinear",
                                   padding_mode="border", align_corners=True))
    return torch.cat(faces, dim=3)                        # 1,3,res,6*res

def cubemap_to_equirect(cube, H, W):
    """cube: 1x3xR x6R -> 1x3xHxW equirect"""
    dev = cube.device
    R = cube.shape[2]
    lon = torch.linspace(-math.pi, math.pi, W, device=dev)
    lat = torch.linspace(math.pi / 2, -math.pi / 2, H, device=dev)
    lat, lon = torch.meshgrid(lat, lon, indexing="ij")    # H,W
    x = torch.cos(lat) * torch.sin(lon)
    y = torch.sin(lat)
    z = torch.cos(lat) * torch.cos(lon)
    ax, ay, az = x.abs(), y.abs(), z.abs()
    out = torch.zeros(1, 3, H, W, device=dev)
    def sample_face(f, mask, u, v):
        gx = (u.clamp(-1, 1))
        gy = (v.clamp(-1, 1))
        # map face-local (-1..1) to that face's column block in the cube strip
        col = (f + (gx + 1) / 2) / 6 * 2 - 1               # x within full strip -1..1
        grid = torch.stack([col, gy], -1)[None]
        s = F.grid_sample(cube, grid, mode="bilinear", padding_mode="border", align_corners=True)
        m = mask[None, None].float()
        return s * m
    acc = torch.zeros_like(out)
    # +X / -X (major axis x)
    mx = (ax >= ay) & (ax >= az)
    px = mx & (x > 0); nx = mx & (x <= 0)
    acc += sample_face(0, px, -z / ax.clamp(min=1e-6), -y / ax.clamp(min=1e-6))
    acc += sample_face(1, nx,  z / ax.clamp(min=1e-6), -y / ax.clamp(min=1e-6))
    # +Y / -Y (major axis y)
    my = (ay > ax) & (ay >= az)
    py = my & (y > 0); ny = my & (y <= 0)
    acc += sample_face(2, py,  x / ay.clamp(min=1e-6),  z / ay.clamp(min=1e-6))
    acc += sample_face(3, ny,  x / ay.clamp(min=1e-6), -z / ay.clamp(min=1e-6))
    # +Z / -Z (major axis z)
    mz = (az > ax) & (az > ay)
    pz = mz & (z > 0); nz = mz & (z <= 0)
    acc += sample_face(4, pz,  x / az.clamp(min=1e-6), -y / az.clamp(min=1e-6))
    acc += sample_face(5, nz, -x / az.clamp(min=1e-6), -y / az.clamp(min=1e-6))
    return acc.clamp(0, 1)

@torch.no_grad()
def sisr_cubemap(model, img, scale, tile, pad, dev, fp16, win=0, face_res=None):
    """360-aware: equirect -> cube faces -> SISR each -> cube -> equirect"""
    H, W = img.shape[:2]
    eq = torch.from_numpy(img.transpose(2, 0, 1))[None].to(dev).float()
    R = face_res or (H // 2)
    cube = equirect_to_cubemap(eq, R)                     # 1,3,R,6R
    # SISR per face (process the strip face-by-face to keep model happy)
    faces_sr = []
    for f in range(6):
        face = cube[:, :, :, f * R:(f + 1) * R][0].cpu().numpy().transpose(1, 2, 0)
        sr = sisr(model, np.ascontiguousarray(face), scale, tile, pad, dev, fp16, win)
        faces_sr.append(torch.from_numpy(sr.transpose(2, 0, 1))[None].to(dev))
    cube_sr = torch.cat(faces_sr, dim=3)                  # 1,3,sR,6sR
    eq_sr = cubemap_to_equirect(cube_sr, H * scale, W * scale)
    return eq_sr[0].cpu().numpy().transpose(1, 2, 0)

# ----- unified dispatch -----
def make_model(method, weights, scale, dev):
    if method in ("rrdbnet", "rrdbnet-cube"):
        return load_rrdbnet(weights, scale, dev), 0
    if method in ("swinir", "swinir-cube"):
        return load_swinir(weights, scale, dev)
    return None, 0

def run_method(method, model, win, img, scale, tile, pad, dev, fp16):
    if method == "lanczos":
        return lanczos(img, scale)
    # SwinIR (transformer) is numerically unstable under fp16 autocast -> force fp32
    fp = False if method.startswith("swinir") else fp16
    if method.endswith("-cube"):
        return sisr_cubemap(model, img, scale, tile, pad, dev, fp, win)
    return sisr(model, img, scale, tile, pad, dev, fp, win)
