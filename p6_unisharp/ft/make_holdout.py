#!/usr/bin/env python3
"""Split a built scene into train/val SimPanorama scenes via symlinks (no copy).
Tail `--val-frac` frames -> *_val (held out); the rest -> *_train.

  python make_holdout.py --scene scene_023hf --val-frac 0.2

Creates:
  ft/data/<scene>_train/{NNNNN.jpg, depth/NNNNN.npy, sky/NNNNN.png}  (symlinks)
  ft/data/<scene>_val/...
  ft/poses/<scene>_train.csv , <scene>_val.csv   (split pose rows)
  ft/manifests/<scene>_train.txt , <scene>_val.txt
"""
import argparse, csv, os
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--scene", required=True)
ap.add_argument("--val-frac", type=float, default=0.2)
ap.add_argument("--ft", default="/raid/git/gaussian-splatting/p6_unisharp/ft")
a = ap.parse_args()

FT = Path(a.ft)
src_dir = FT / "data" / a.scene
pose_csv = FT / "poses" / f"{a.scene}.csv"
assert src_dir.is_dir(), src_dir
assert pose_csv.exists(), pose_csv

rows = list(csv.DictReader(open(pose_csv)))
frames = sorted(int(r["frame"]) for r in rows)
n = len(frames)
cut = int(round(n * (1.0 - a.val_frac)))
train_fr = set(frames[:cut]); val_fr = set(frames[cut:])
print(f"{a.scene}: {n} frames -> train={len(train_fr)} val={len(val_fr)} (cut at frame {frames[cut]})")

def link(sub_fr, suffix):
    dst = FT / "data" / f"{a.scene}_{suffix}"
    (dst / "depth").mkdir(parents=True, exist_ok=True)
    (dst / "sky").mkdir(parents=True, exist_ok=True)
    for fr in sub_fr:
        stem = f"{fr:05d}"
        for rel in (f"{stem}.jpg", f"depth/{stem}.npy", f"sky/{stem}.png"):
            s = src_dir / rel; d = dst / rel
            if s.exists():
                if d.is_symlink() or d.exists():
                    d.unlink()
                d.symlink_to(os.path.relpath(s, d.parent))
    # split pose csv
    pc = FT / "poses" / f"{a.scene}_{suffix}.csv"
    with open(pc, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["frame", "x", "y", "z"])
        for r in rows:
            if int(r["frame"]) in sub_fr:
                w.writerow([r["frame"], r["x"], r["y"], r["z"]])
    man = FT / "manifests" / f"{a.scene}_{suffix}.txt"
    man.parent.mkdir(parents=True, exist_ok=True)
    man.write_text(f"{a.scene}_{suffix}\n")
    print(f"  {a.scene}_{suffix}: {len(sub_fr)} frames -> {dst}  | manifest {man.name}")

link(train_fr, "train")
link(val_fr, "val")
