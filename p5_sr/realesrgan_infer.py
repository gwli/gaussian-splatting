#!/usr/bin/env python3
"""task3 stage-3 (model SR): self-contained Real-ESRGAN (RRDBNet) inference.

Pure-PyTorch reimplementation of BasicSR's RRDBNet so we need NO basicsr /
realesrgan pip packages (which drag in old torchvision APIs). Loads the official
release weights (RealESRGAN_x2plus.pth / x4plus / x4plus_anime). Tiled inference
keeps 8K-class equirect frames within GPU memory.

Usage:
  realesrgan_infer.py <in_frames_dir> <out_frames_dir> <weights.pth> \
      [--scale 2|4] [--tile 512] [--pad 16] [--fp16] [--glob '*.png']
Frames are matched by name and written with the same filename to out dir.
"""
import os, sys, glob, argparse
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

# ---------------- RRDBNet (matches BasicSR key names) ----------------
class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(nf, gc)
        self.rdb2 = ResidualDenseBlock(nf, gc)
        self.rdb3 = ResidualDenseBlock(nf, gc)
    def forward(self, x):
        out = self.rdb3(self.rdb2(self.rdb1(x)))
        return out * 0.2 + x

def pixel_unshuffle(x, s):
    b, c, h, w = x.shape
    x = x.view(b, c, h // s, s, w // s, s)
    return x.permute(0, 1, 3, 5, 2, 4).reshape(b, c * s * s, h // s, w // s)

class RRDBNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, scale=4, nf=64, nb=23, gc=32):
        super().__init__()
        self.scale = scale
        ic = in_ch * (4 if scale == 2 else 16 if scale == 1 else 1)
        self.conv_first = nn.Conv2d(ic, nf, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x):
        feat = pixel_unshuffle(x, 2) if self.scale == 2 else \
               pixel_unshuffle(x, 4) if self.scale == 1 else x
        feat = self.conv_first(feat)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))

# ---------------- tiled inference ----------------
@torch.no_grad()
def sr_image(model, img, scale, tile, pad, dev, fp16):
    # img: HxWx3 float [0,1] -> upscaled HxWx3 float
    x = torch.from_numpy(img.transpose(2, 0, 1))[None].to(dev)
    if fp16: x = x.half()
    _, _, H, W = x.shape
    if tile <= 0:
        return _fwd(model, x, fp16).clamp(0, 1)[0].cpu().float().numpy().transpose(1, 2, 0)
    out = torch.zeros(1, 3, H * scale, W * scale, device=dev, dtype=x.dtype)
    for y0 in range(0, H, tile):
        for x0 in range(0, W, tile):
            y1, x1 = min(y0 + tile, H), min(x0 + tile, W)
            ya, xa = max(0, y0 - pad), max(0, x0 - pad)
            yb, xb = min(H, y1 + pad), min(W, x1 + pad)
            patch = x[:, :, ya:yb, xa:xb]
            sp = _fwd(model, patch, fp16)
            ty0, tx0 = (y0 - ya) * scale, (x0 - xa) * scale
            th, tw = (y1 - y0) * scale, (x1 - x0) * scale
            out[:, :, y0 * scale:y1 * scale, x0 * scale:x1 * scale] = sp[:, :, ty0:ty0 + th, tx0:tx0 + tw]
    return out.clamp(0, 1)[0].cpu().float().numpy().transpose(1, 2, 0)

def _fwd(model, x, fp16):
    with torch.autocast("cuda", enabled=fp16):
        return model(x).float()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("indir"); ap.add_argument("outdir"); ap.add_argument("weights")
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--pad", type=int, default=16)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--glob", default="*.png")
    a = ap.parse_args()
    try:
        import cv2
    except ImportError:
        print("[realesrgan] need opencv (pip install --no-deps opencv-python-headless)", file=sys.stderr)
        raise
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    nb = 6 if "anime" in os.path.basename(a.weights) else 23
    model = RRDBNet(scale=a.scale, nb=nb).to(dev).eval()
    sd = torch.load(a.weights, map_location="cpu", weights_only=False)
    sd = sd.get("params_ema", sd.get("params", sd))
    model.load_state_dict(sd, strict=True)
    if a.fp16: model = model.half()
    os.makedirs(a.outdir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.indir, a.glob)))
    print(f"[realesrgan] {len(files)} frames x{a.scale} tile={a.tile} fp16={a.fp16} dev={dev}")
    for i, f in enumerate(files):
        img = cv2.imread(f, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
        img = img[:, :, ::-1].copy()                       # BGR->RGB
        sr = sr_image(model, img, a.scale, a.tile, a.pad, dev, a.fp16)
        sr = (sr[:, :, ::-1] * 255.0).round().astype(np.uint8)  # RGB->BGR
        cv2.imwrite(os.path.join(a.outdir, os.path.basename(f)), sr)
        if (i + 1) % 10 == 0 or i == len(files) - 1:
            print(f"[realesrgan] {i+1}/{len(files)}")

if __name__ == "__main__":
    main()
