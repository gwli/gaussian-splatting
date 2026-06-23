#!/usr/bin/env python3
"""task3 extra: virtual flat (rectilinear) camera over a 360 equirect frame seq.
Per-frame yaw/pitch/h_fov animation via torch grid_sample (ffmpeg v360 can't be
animated — its params aren't runtime commands). Output is normal flat video.

Equirect convention: u->lon in [-pi,pi] (lon=0 at center=+Z), v->lat in
[+pi/2 (top=zenith) .. -pi/2 (bottom=nadir)]. Camera looks +Z; pitch<0 looks DOWN.

  flat_camera.py <in_frames_dir> <out_frames_dir> --res 1920x1080 \
     --yaw0 -80 --yaw1 -30 --pitch0 -42 --pitch1 -34 --hfov0 88 --hfov1 78 [--fp16]
  flat_camera.py --still <eq.png> <out.png> --res 1920x1080 --yaw -55 --pitch -40 --hfov 84
"""
import os, sys, glob, math, argparse
import numpy as np, torch, torch.nn.functional as F, cv2

def ray_grid(Wf, Hf, hfov_deg, dev):
    f = (Wf / 2) / math.tan(math.radians(hfov_deg) / 2)
    xs = torch.arange(Wf, device=dev) - (Wf - 1) / 2
    ys = torch.arange(Hf, device=dev) - (Hf - 1) / 2
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    d = torch.stack([gx, -gy, torch.full_like(gx, f)], -1)      # X right, Y up, Z fwd
    return F.normalize(d, dim=-1)                                # Hf,Wf,3

def rot_x(a):  # pitch
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=torch.float32)
def rot_y(a):  # yaw
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float32)

def sample(eq, dirs, yaw_deg, pitch_deg, dev, fp16):
    # eq: 1x3xHxW ; dirs: Hf,Wf,3 -> rectilinear view as 1x3xHfxWf
    # pitch<0 = look DOWN (negate so sign matches ffmpeg v360 convention)
    R = (rot_y(math.radians(yaw_deg)) @ rot_x(math.radians(-pitch_deg))).to(dev)
    wd = dirs.reshape(-1, 3) @ R.T                              # world dirs
    lon = torch.atan2(wd[:, 0], wd[:, 2])                       # -pi..pi
    lat = torch.asin(wd[:, 1].clamp(-1, 1))                     # -pi/2..pi/2
    u = lon / math.pi                                           # -1..1
    v = -lat / (math.pi / 2)                                    # top(+lat)->-1
    grid = torch.stack([u, v], -1).reshape(1, dirs.shape[0], dirs.shape[1], 2)
    if fp16: eq = eq.half(); grid = grid.half()
    out = F.grid_sample(eq, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return out.float()

def smooth(a): a = max(0.0, min(1.0, a)); return a * a * (3 - 2 * a)

def load_eq(path, dev):
    img = cv2.imread(path, cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32) / 255.0
    return torch.from_numpy(img.transpose(2, 0, 1))[None].to(dev)

def save(t, path):
    a = (t.clamp(0, 1)[0].cpu().numpy().transpose(1, 2, 0)[:, :, ::-1] * 255).round().astype(np.uint8)
    cv2.imwrite(path, a)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--res", default="1920x1080")
    ap.add_argument("--still", action="store_true")
    ap.add_argument("--yaw", type=float, default=-55); ap.add_argument("--pitch", type=float, default=-40)
    ap.add_argument("--hfov", type=float, default=84)
    ap.add_argument("--yaw0", type=float, default=-80); ap.add_argument("--yaw1", type=float, default=-30)
    ap.add_argument("--pitch0", type=float, default=-42); ap.add_argument("--pitch1", type=float, default=-34)
    ap.add_argument("--hfov0", type=float, default=88); ap.add_argument("--hfov1", type=float, default=78)
    ap.add_argument("--fp16", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Wf, Hf = map(int, a.res.split("x"))
    if a.still:
        eq = load_eq(a.a, dev)
        save(sample(eq, ray_grid(Wf, Hf, a.hfov, dev), a.yaw, a.pitch, dev, a.fp16), a.b)
        print(f"[flat] still yaw={a.yaw} pitch={a.pitch} hfov={a.hfov} -> {a.b}"); return
    files = sorted(glob.glob(os.path.join(a.a, "*.png")))
    os.makedirs(a.b, exist_ok=True)
    n = len(files)
    print(f"[flat] {n} frames -> {Wf}x{Hf}  yaw {a.yaw0}->{a.yaw1} pitch {a.pitch0}->{a.pitch1} hfov {a.hfov0}->{a.hfov1}")
    for i, fp in enumerate(files):
        k = smooth(i / max(1, n - 1))
        yaw = a.yaw0 + (a.yaw1 - a.yaw0) * k
        pit = a.pitch0 + (a.pitch1 - a.pitch0) * k
        hf = a.hfov0 + (a.hfov1 - a.hfov0) * k
        eq = load_eq(fp, dev)
        save(sample(eq, ray_grid(Wf, Hf, hf, dev), yaw, pit, dev, a.fp16),
             os.path.join(a.b, os.path.basename(fp)))
        if (i + 1) % 50 == 0 or i == n - 1: print(f"[flat] {i+1}/{n}")

if __name__ == "__main__":
    main()
