#!/usr/bin/env python3
"""task_sr: apply a single-image SR method to a directory of frames.
Backs the swinir/realesrgan engines in enhance.sh (reuses sr_lib).

  run_frames.py <in_dir> <out_dir> <method> <weights> [--scale 2] [--tile 512]
                [--pad 16] [--fp16] [--glob '*.png']
  method: rrdbnet | rrdbnet-cube | swinir | swinir-cube | lanczos
"""
import os, sys, glob, argparse
import numpy as np, torch, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sr_lib

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("indir"); ap.add_argument("outdir"); ap.add_argument("method")
    ap.add_argument("weights", nargs="?", default="")
    ap.add_argument("--scale", type=int, default=2); ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--pad", type=int, default=16); ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--glob", default="*.png")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, win = sr_lib.make_model(a.method, a.weights, a.scale, dev)
    os.makedirs(a.outdir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.indir, a.glob)))
    print(f"[run_frames] {len(files)} frames, method={a.method} x{a.scale} tile={a.tile} dev={dev}")
    for i, f in enumerate(files):
        img = cv2.imread(f, cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32) / 255.0
        sr = sr_lib.run_method(a.method, model, win, np.ascontiguousarray(img),
                               a.scale, a.tile, a.pad, dev, a.fp16)
        sr = (np.clip(sr, 0, 1)[:, :, ::-1] * 255.0).round().astype(np.uint8)
        cv2.imwrite(os.path.join(a.outdir, os.path.basename(f)), sr)
        if (i + 1) % 10 == 0 or i == len(files) - 1:
            print(f"[run_frames] {i+1}/{len(files)}")

if __name__ == "__main__":
    main()
