#!/usr/bin/env python3
"""Feasibility gate: run LocateAnything-3B on a de-rotated ERP frame and draw the
detected boxes for water/building/sky, to check whether text grounding works on
our aerial equirectangular imagery before building the full plane-prior pipeline.
"""
import argparse, re
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModel, AutoTokenizer, AutoProcessor

ap = argparse.ArgumentParser()
ap.add_argument("--image", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="nvidia/LocateAnything-3B")
ap.add_argument("--queries", nargs="+",
                default=["water", "river or canal", "building rooftop", "sky", "boat"])
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
proc = AutoProcessor.from_pretrained(a.model, trust_remote_code=True)
model = AutoModel.from_pretrained(a.model, torch_dtype=torch.bfloat16,
                                 trust_remote_code=True).to("cuda").eval()

img = Image.open(a.image).convert("RGB")
W, H = img.size
draw = ImageDraw.Draw(img)
colors = ["#00e5ff", "#00ff5e", "#ffd400", "#ff4dd2", "#ff7a00", "#ffffff"]

def detect(query):
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": Image.open(a.image).convert("RGB")},
        {"type": "text", "text": f"Locate all the instances that match: {query}"}]}]
    text = proc.py_apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    images, videos = proc.process_vision_info(msgs)
    inp = proc(text=[text], images=images, videos=videos, return_tensors="pt").to("cuda")
    out = model.generate(pixel_values=inp["pixel_values"].to(torch.bfloat16),
                         input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                         image_grid_hws=inp.get("image_grid_hws", None), tokenizer=tok,
                         max_new_tokens=4096, generation_mode="hybrid", use_cache=True)
    ans = out[0] if isinstance(out, (list, tuple)) else out
    if not isinstance(ans, str):
        ans = tok.decode(ans, skip_special_tokens=True) if hasattr(ans, "__len__") else str(ans)
    boxes = [[int(g) / 1000 for g in m.groups()]
             for m in re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", ans)]
    return boxes, ans

import numpy as np
for i, q in enumerate(a.queries):
    try:
        boxes, ans = detect(q)
    except Exception as e:
        print(f"[{q}] ERROR: {e}"); continue
    cen = np.array([[(x1+x2)/2, (y1+y2)/2] for (x1,y1,x2,y2) in boxes]) if boxes else np.zeros((0,2))
    areas = np.array([abs(x2-x1)*abs(y2-y1) for (x1,y1,x2,y2) in boxes]) if boxes else np.zeros(0)
    print(f"[{q}] {len(boxes)} boxes | centroid_y mean={cen[:,1].mean():.2f} std={cen[:,1].std():.2f} "
          f"| median_area={np.median(areas) if len(areas) else 0:.4f}", flush=True)
    print(f"      raw[:200]={ans[:200]!r}", flush=True)
    c = colors[i % len(colors)]
    for (x1, y1, x2, y2) in boxes:
        xa, xb = sorted([x1 * W, x2 * W]); ya, yb = sorted([y1 * H, y2 * H])
        draw.rectangle([xa, ya, xb, yb], outline=c, width=2)
    if boxes:
        draw.text((5, 5 + i * 14), f"{q}: {len(boxes)}", fill=c)

img.save(a.out)
print(f"saved {a.out}")
