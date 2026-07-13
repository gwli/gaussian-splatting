#!/usr/bin/env python3
"""Dynamic-content weight masks for ERP panos via CLIPSeg: union of vehicle /
person / boat probabilities -> weight = 1 - mask (dynamic pixels ~0).
Writes 8-bit grayscale 1024x512 PNGs <pano_dir>/../dyn_mask/pano_XXXX.png
usage: mask_dyn.py <pano_dir> [n=1140] [thresh=0.40]
"""
import sys, os
import numpy as np, torch
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

PANO = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 1140
TH = float(sys.argv[3]) if len(sys.argv) > 3 else 0.40
OUT = os.path.join(os.path.dirname(PANO.rstrip("/")), "dyn_mask")
os.makedirs(OUT, exist_ok=True)
Q = ["car", "truck or bus", "boat or ship", "person"]
W, H = 1024, 512

dev = "cuda"
proc = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(dev).eval()

import torch.nn.functional as F
S = int(model.config.vision_config.image_size)  # cached snapshot may be 224 or 352
txt = proc.tokenizer(Q, padding=True, return_tensors="pt").to(dev)
# small objects (cars/people) vanish at S^2 full-ERP input -> tile the lower 3/4
# (rows H/4..H, where ground content lives) into a 4x2 grid at ~8x eff. resolution.
COLS, ROWS = 4, 2
Y0 = H // 4; TW, THh = W // COLS, (H - Y0) // ROWS
with torch.no_grad():
    for k in range(1, N + 1):
        im = Image.open(f"{PANO}/pano_{k:04d}.jpg").convert("RGB").resize((W, H), Image.BILINEAR)
        tiles = [im.crop((c*TW, Y0+r*THh, (c+1)*TW, Y0+(r+1)*THh))
                 for r in range(ROWS) for c in range(COLS)]
        pix = proc.image_processor(tiles, size={"height": S, "width": S},
                                   return_tensors="pt")["pixel_values"].to(dev)  # (8,3,S,S)
        probs = []
        for q in range(len(Q)):  # per-query pass over all tiles (batch=8)
            t = {"input_ids": txt["input_ids"][q:q+1].repeat(len(tiles), 1),
                 "attention_mask": txt["attention_mask"][q:q+1].repeat(len(tiles), 1)}
            p = torch.sigmoid(model(pixel_values=pix, **t).logits)   # (8,S,S)
            if p.ndim == 2: p = p[None]
            probs.append(p)
        m8 = torch.stack(probs).max(0).values                        # union over queries, (8,S,S)
        canvas = torch.zeros(H, W, device=dev)
        for i in range(len(tiles)):
            r, c = divmod(i, COLS)
            canvas[Y0+r*THh:Y0+(r+1)*THh, c*TW:(c+1)*TW] = F.interpolate(
                m8[i][None, None], size=(THh, TW), mode="bilinear")[0, 0]
        m = (canvas > TH).float()
        m = F.max_pool2d(m[None, None], 11, stride=1, padding=5)[0, 0]  # dilate 5px
        w = ((1.0 - m) * 255).byte().cpu().numpy()
        Image.fromarray(w).save(f"{OUT}/pano_{k:04d}.png")
        if k % 200 == 0: print(k, flush=True)
print(f"[mask_dyn] {N} weight masks -> {OUT}")
