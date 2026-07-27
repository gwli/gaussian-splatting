#!/usr/bin/env python3
"""Export the DOWN-lens poses of a rig reconstruction as a plain COLMAP TXT model.

The rig model contains two images per frame (down/ + up/). Everything downstream
(rig_solve_scene.py -> pano_cams json) expects one image per frame named
f_XXXX.jpg in the down-camera frame, which is also the frame our ERP is stitched
in. So: keep down/ only, strip the prefix, drop the up/ entries.

usage: export_rig_model.py <rec_dir> <out_txt_dir> [--prefix down/]
"""
import sys, os, argparse
import numpy as np
import pycolmap

ap = argparse.ArgumentParser()
ap.add_argument("rec"); ap.add_argument("out")
ap.add_argument("--prefix", default="down/")
a = ap.parse_args()

rec = pycolmap.Reconstruction(a.rec)
os.makedirs(a.out, exist_ok=True)
print(f"[export] input: {rec.num_reg_images()} images, {rec.num_points3D()} points")

def cam_from_world(img):
    c = getattr(img, "cam_from_world", None)
    return c() if callable(c) else c

lines, cam_id_used = [], None
for iid, img in rec.images.items():
    if not img.name.startswith(a.prefix):
        continue
    p = cam_from_world(img)
    if p is None:                      # unregistered
        continue
    q = p.rotation.quat               # xyzw
    t = p.translation
    name = img.name[len(a.prefix):]
    cam_id_used = img.camera_id
    lines.append((name, f"{iid} {q[3]} {q[0]} {q[1]} {q[2]} {t[0]} {t[1]} {t[2]} {img.camera_id} {name}\n\n"))

with open(os.path.join(a.out, "images.txt"), "w") as f:
    f.write("# Image list\n#\n")
    for _, l in sorted(lines):
        f.write(l)

cam = rec.cameras[cam_id_used]
with open(os.path.join(a.out, "cameras.txt"), "w") as f:
    f.write("# Camera list\n#\n")
    f.write(f"{cam.camera_id} {cam.model.name} {cam.width} {cam.height} "
            + " ".join(str(x) for x in cam.params) + "\n")
open(os.path.join(a.out, "points3D.txt"), "w").close()
print(f"[export] wrote {len(lines)} down-lens poses -> {a.out}")
print(f"[export] camera: {cam.model.name} f={cam.params[0]:.3f} k1={cam.params[4]:.6f}")
