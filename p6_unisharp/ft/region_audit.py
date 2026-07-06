#!/usr/bin/env python3
"""Region audit: does fog concentrate on non-rigid content (water/trees) vs
rigid (buildings/roads)? Split render|GT stack, CLIPSeg the GT half, report
per-class mean|render-GT| and sharpness retention (grad(render)/grad(GT))."""
import numpy as np, torch
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

img = np.asarray(Image.open("/w/p6_unisharp/ft/r_dense_hr.png").convert("RGB"), np.float32)
H = img.shape[0] // 2
ren, gt = img[:H], img[H:2*H]
gtim = Image.fromarray(gt.astype(np.uint8))
dev = "cuda" if torch.cuda.is_available() else "cpu"
proc = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(dev).eval()
Q = ["water", "trees and vegetation", "buildings", "roads and paved paths"]
with torch.no_grad():
    inp = proc(text=Q, images=[gtim]*len(Q), padding=True, return_tensors="pt").to(dev)
    pr = torch.sigmoid(model(**inp).logits).float().cpu()
pr = torch.nn.functional.interpolate(pr[None], size=gt.shape[:2], mode="bilinear")[0].numpy()
lab = pr.argmax(0); conf = pr.max(0)
def grad(x):
    g = np.abs(np.diff(x.mean(2), axis=0)).mean() + 0  # placeholder
    gx = np.abs(np.diff(x.mean(2), axis=1)); gy = np.abs(np.diff(x.mean(2), axis=0))
    return gx[:-1,:]+gy[:,:-1]
gr, gg = grad(ren), grad(gt)
err = np.abs(ren - gt).mean(2)
print(f"{'class':>22} {'area%':>6} {'|err|':>6} {'sharp_ret%':>10}  (grad_render/grad_GT)")
for i, q in enumerate(Q):
    m = (lab == i) & (conf > 0.4)
    mg = m[:-1, :-1]
    if m.sum() < 500: print(f"{q:>22}  (too few px)"); continue
    ret = 100 * gr[mg].mean() / (gg[mg].mean() + 1e-6)
    print(f"{q:>22} {100*m.mean():6.1f} {err[m].mean():6.1f} {ret:10.1f}")
