#!/usr/bin/env python3
"""task_sr: apply Real-BasicVSR (x4 video SR) to a directory of frames, chunked
into overlapping windows so arbitrary-length / large clips fit in VRAM.
Backs the `vsr` engine in enhance.sh.

  vsr_run_frames.py <in_dir> <out_dir> <weights> [--win 12] [--overlap 2] [--fp16]

Bidirectional propagation needs a temporal window; we process WIN frames at a
time, keep the middle (WIN-2*overlap) outputs, and slide. Overlap frames give
each kept frame valid past+future context.
"""
import os, sys, glob, argparse
import numpy as np, torch, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vsr_realbasicvsr as V

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("indir"); ap.add_argument("outdir"); ap.add_argument("weights")
    ap.add_argument("--win", type=int, default=12); ap.add_argument("--overlap", type=int, default=2)
    ap.add_argument("--fp16", action="store_true"); ap.add_argument("--glob", default="*.png")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = V.load(a.weights, dev)
    if a.fp16: model = model.half()
    files = sorted(glob.glob(os.path.join(a.indir, a.glob)))
    os.makedirs(a.outdir, exist_ok=True)
    n = len(files)
    print(f"[vsr_run] {n} frames x4, win={a.win} overlap={a.overlap} fp16={a.fp16} dev={dev}")
    ov, win = a.overlap, max(a.win, 2 * a.overlap + 1)
    step = win - 2 * ov
    done = 0; i = 0
    while i < n:
        s = max(0, i - ov)                       # window start with left context
        e = min(n, i + step + ov)                # window end with right context
        batch = files[s:e]
        frames = [cv2.imread(f, cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32) / 255.0 for f in batch]
        with torch.no_grad():
            sr = V.run_clip(model, frames, dev, a.fp16)
        # keep outputs corresponding to files[i : i+step]
        for k, f in enumerate(batch):
            gi = s + k
            if i <= gi < min(n, i + step):
                out = (np.clip(sr[k], 0, 1)[:, :, ::-1] * 255.0).round().astype(np.uint8)
                cv2.imwrite(os.path.join(a.outdir, os.path.basename(f)), out)
                done += 1
        print(f"[vsr_run] {done}/{n}")
        i += step
    print(f"[vsr_run] wrote {done} frames -> {a.outdir}")

if __name__ == "__main__":
    main()
