#!/usr/bin/env python3
"""Gate + tool: text-prompted pixel segmentation via CLIPSeg (CLIP-based, more
robust to aerial/ERP domain shift than a box detector). Produces per-query dense
masks (water / sky / ...) for the water-plane depth prior.

Modes:
  --viz  : overlay masks on one frame for the feasibility gate
  else   : write binary masks <out>/<query>/<stem>.png for a whole frame dir
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

ap = argparse.ArgumentParser()
ap.add_argument("--image")                       # single image (viz mode)
ap.add_argument("--frames-dir")                  # or a dir (batch mode)
ap.add_argument("--out", required=True)
ap.add_argument("--queries", nargs="+", default=["water", "sky"])
ap.add_argument("--thresh", type=float, default=0.45)
ap.add_argument("--viz", action="store_true")
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
proc = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(dev).eval()

@torch.no_grad()
def masks_for(img):
    W, H = img.size
    inp = proc(text=a.queries, images=[img] * len(a.queries), padding=True, return_tensors="pt").to(dev)
    logits = model(**inp).logits                 # (Q,352,352)
    if logits.ndim == 2:
        logits = logits[None]
    prob = torch.sigmoid(logits).float().cpu()
    prob = torch.nn.functional.interpolate(prob[None], size=(H, W), mode="bilinear", align_corners=False)[0]
    return prob.numpy()                          # (Q,H,W) in [0,1]

if a.viz:
    img = Image.open(a.image).convert("RGB"); W, H = img.size
    prob = masks_for(img)
    over = np.asarray(img, np.float32).copy()
    cols = [np.array([0, 120, 255]), np.array([255, 60, 60]), np.array([0, 220, 0]), np.array([255, 200, 0])]
    for i, q in enumerate(a.queries):
        m = prob[i] > a.thresh
        over[m] = 0.5 * over[m] + 0.5 * cols[i % len(cols)]
        print(f"[{q}] coverage={m.mean():.2f} meanprob={prob[i].mean():.2f}")
    Image.fromarray(over.clip(0, 255).astype(np.uint8)).save(a.out)
    print(f"saved {a.out}")
else:
    fdir = Path(a.frames_dir); outp = Path(a.out); outp.mkdir(parents=True, exist_ok=True)
    for q in a.queries:
        (outp / q.replace(" ", "_")).mkdir(exist_ok=True)
    jpgs = sorted(fdir.glob("*.jpg"))
    for j in jpgs:
        prob = masks_for(Image.open(j).convert("RGB"))
        for i, q in enumerate(a.queries):
            m = (prob[i] > a.thresh).astype(np.uint8) * 255
            Image.fromarray(m).save(outp / q.replace(" ", "_") / f"{j.stem}.png")
    print(f"[clipseg] {len(jpgs)} frames x {len(a.queries)} masks -> {outp}")
