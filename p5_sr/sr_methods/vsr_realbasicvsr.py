#!/usr/bin/env python3
"""task_sr M4: self-contained Real-BasicVSR (x4 video super-resolution) inference.

Faithful re-implementation of mmediting's RealBasicVSRNet (image_cleaning +
BasicVSRNet: SPyNet flow + bidirectional recurrent propagation + pixelshuffle
x4), with module/key names matching the official checkpoint so it loads with
strict=True. NO mmcv/mmagic dependency — flow_warp is just grid_sample.

Checkpoint: RealBasicVSR_x4.pth (OpenMMLab). Uses the `generator.` weights.

Usage:
  vsr_realbasicvsr.py <in_frames_dir> <out_frames_dir> <weights.pth> [--fp16] [--win 0]
A whole frame sequence is processed together (bidirectional needs the clip).
"""
import os, sys, glob, argparse
import torch, torch.nn as nn, torch.nn.functional as F

# ---------- primitives ----------
class ResidualBlockNoBN(nn.Module):
    def __init__(self, nf=64):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))

class ResidualBlocksWithInputConv(nn.Module):
    def __init__(self, in_ch, out_ch=64, num_blocks=30):
        super().__init__()
        main = [nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=True),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Sequential(*[ResidualBlockNoBN(out_ch) for _ in range(num_blocks)])]
        self.main = nn.Sequential(*main)
    def forward(self, x):
        return self.main(x)

class PixelShufflePack(nn.Module):
    def __init__(self, in_ch, out_ch, scale=2, ksize=3):
        super().__init__()
        self.upsample_conv = nn.Conv2d(in_ch, out_ch * scale * scale, ksize, 1, ksize // 2)
        self.scale = scale
    def forward(self, x):
        return F.pixel_shuffle(self.upsample_conv(x), self.scale)

def flow_warp(x, flow):
    # x: N,C,H,W ; flow: N,H,W,2 (dx,dy in pixels)
    N, C, H, W = x.size()
    gy, gx = torch.meshgrid(torch.arange(H, device=x.device),
                            torch.arange(W, device=x.device), indexing="ij")
    grid = torch.stack((gx, gy), 2).float()                  # H,W,2
    vgrid = grid[None] + flow
    vgrid_x = 2.0 * vgrid[..., 0] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[..., 1] / max(H - 1, 1) - 1.0
    g = torch.stack((vgrid_x, vgrid_y), dim=3)
    return F.grid_sample(x, g, mode="bilinear", padding_mode="border", align_corners=True)

# ---------- SPyNet ----------
class SPyNetBasicModule(nn.Module):
    def __init__(self):
        super().__init__()
        ch = [8, 32, 64, 32, 16, 2]
        layers = []
        for i in range(5):
            layers.append(_ConvAct(ch[i], ch[i + 1], 7, last=(i == 4)))
        self.basic_module = nn.Sequential(*layers)
    def forward(self, x):
        return self.basic_module(x)

class _ConvAct(nn.Module):
    def __init__(self, i, o, k, last=False):
        super().__init__()
        self.conv = nn.Conv2d(i, o, k, 1, k // 2)
        self.act = None if last else nn.ReLU(inplace=False)
    def forward(self, x):
        x = self.conv(x)
        return x if self.act is None else self.act(x)

class SPyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.basic_module = nn.ModuleList([SPyNetBasicModule() for _ in range(6)])
        self.register_buffer("mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))
    def compute_flow(self, ref, supp):
        n, _, h, w = ref.size()
        ref = [(ref - self.mean) / self.std]
        supp = [(supp - self.mean) / self.std]
        for _ in range(5):
            ref.append(F.avg_pool2d(ref[-1], 2, 2, count_include_pad=False))
            supp.append(F.avg_pool2d(supp[-1], 2, 2, count_include_pad=False))
        ref = ref[::-1]; supp = supp[::-1]                   # coarse->fine
        flow = ref[0].new_zeros(n, 2, h // 32, w // 32)
        for level in range(len(ref)):
            if level == 0:
                up = flow
            else:
                up = F.interpolate(flow, scale_factor=2, mode="bilinear",
                                   align_corners=True) * 2.0
            warped = flow_warp(supp[level], up.permute(0, 2, 3, 1))
            flow = up + self.basic_module[level](
                torch.cat([ref[level], warped, up], 1))
        return flow
    def forward(self, ref, supp):
        h, w = ref.shape[2:]
        H = (h + 31) // 32 * 32; W = (w + 31) // 32 * 32
        ref = F.interpolate(ref, size=(H, W), mode="bilinear", align_corners=False)
        supp = F.interpolate(supp, size=(H, W), mode="bilinear", align_corners=False)
        flow = self.compute_flow(ref, supp)
        flow = F.interpolate(flow, size=(h, w), mode="bilinear", align_corners=False)
        flow[:, 0] *= w / W; flow[:, 1] *= h / H
        return flow

# ---------- BasicVSR ----------
class BasicVSRNet(nn.Module):
    def __init__(self, mid=64, nb=20):
        super().__init__()
        self.spynet = SPyNet()
        self.backward_resblocks = ResidualBlocksWithInputConv(mid + 3, mid, nb)
        self.forward_resblocks = ResidualBlocksWithInputConv(mid + 3, mid, nb)
        self.fusion = nn.Conv2d(mid * 2, mid, 1, 1, 0)
        self.upsample1 = PixelShufflePack(mid, mid, 2)
        self.upsample2 = PixelShufflePack(mid, mid, 2)
        self.conv_hr = nn.Conv2d(mid, mid, 3, 1, 1)
        self.conv_last = nn.Conv2d(mid, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.mid = mid
    def compute_flow(self, lrs):
        n, t, c, h, w = lrs.size()
        a = lrs[:, :-1].reshape(-1, c, h, w)
        b = lrs[:, 1:].reshape(-1, c, h, w)
        fb = self.spynet(a, b).view(n, t - 1, 2, h, w)       # backward
        ff = self.spynet(b, a).view(n, t - 1, 2, h, w)       # forward
        return ff, fb
    def forward(self, lrs):
        n, t, c, h, w = lrs.size()
        ff, fb = self.compute_flow(lrs)
        outs = []
        feat = lrs.new_zeros(n, self.mid, h, w)
        for i in range(t - 1, -1, -1):
            if i < t - 1:
                feat = flow_warp(feat, fb[:, i].permute(0, 2, 3, 1))
            feat = self.backward_resblocks(torch.cat([lrs[:, i], feat], 1))
            outs.append(feat)
        outs = outs[::-1]
        feat = torch.zeros_like(feat)
        res = []
        for i in range(t):
            lr = lrs[:, i]
            if i > 0:
                feat = flow_warp(feat, ff[:, i - 1].permute(0, 2, 3, 1))
            feat = self.forward_resblocks(torch.cat([lr, feat], 1))
            out = torch.cat([outs[i], feat], 1)
            out = self.lrelu(self.fusion(out))
            out = self.lrelu(self.upsample1(out))
            out = self.lrelu(self.upsample2(out))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            out = out + self.img_upsample(lr)
            res.append(out)
        return torch.stack(res, 1)

class RealBasicVSRNet(nn.Module):
    def __init__(self, mid=64, nb=20, clean_blocks=20):
        super().__init__()
        self.image_cleaning = nn.Sequential(
            ResidualBlocksWithInputConv(3, mid, clean_blocks),
            nn.Conv2d(mid, 3, 3, 1, 1, bias=True))
        self.basicvsr = BasicVSRNet(mid, nb)
    def forward(self, lqs):
        n, t, c, h, w = lqs.size()
        v = lqs.view(-1, c, h, w)
        v = (v + self.image_cleaning(v)).clamp(0, 1)
        return self.basicvsr(v.view(n, t, c, h, w))

def load(weights, dev):
    m = RealBasicVSRNet().to(dev).eval()
    sd = torch.load(weights, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    g = {k[len("generator."):]: v for k, v in sd.items() if k.startswith("generator.")}
    m.load_state_dict(g, strict=True)
    return m

@torch.no_grad()
def run_clip(model, frames, dev, fp16):
    # frames: list of HxWx3 float[0,1] -> list of (4H)x(4W)x3
    import numpy as np
    x = torch.stack([torch.from_numpy(f.transpose(2, 0, 1)) for f in frames])[None].to(dev)
    if fp16: x = x.half(); model = model.half()
    with torch.autocast("cuda", enabled=fp16):
        y = model(x).float()
    y = y.clamp(0, 1)[0].cpu().numpy()
    return [y[i].transpose(1, 2, 0) for i in range(y.shape[0])]

if __name__ == "__main__":
    import numpy as np, cv2
    ap = argparse.ArgumentParser()
    ap.add_argument("indir"); ap.add_argument("outdir"); ap.add_argument("weights")
    ap.add_argument("--fp16", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load(a.weights, dev)
    print("[vsr] loaded RealBasicVSR (strict=True OK)")
    files = sorted(glob.glob(os.path.join(a.indir, "*.png")))
    frames = [cv2.imread(f, cv2.IMREAD_COLOR)[:, :, ::-1].astype("float32") / 255.0 for f in files]
    sr = run_clip(model, frames, dev, a.fp16)
    os.makedirs(a.outdir, exist_ok=True)
    for f, s in zip(files, sr):
        cv2.imwrite(os.path.join(a.outdir, os.path.basename(f)),
                    (s[:, :, ::-1] * 255).round().astype("uint8"))
    print(f"[vsr] {len(sr)} frames x4 -> {a.outdir}")
