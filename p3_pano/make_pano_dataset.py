#!/usr/bin/env python3
"""P1.3b: derive one equirectangular camera per panorama from the VGGT
per-crop reconstruction, and emit pano_cams.json for direct-pano training.

Each crop filename encodes its (yaw,pitch) offset within the panorama
(pano_XXXX_y±YYY_p±PP). All crops of a pano share a camera CENTER. The crop
camera→pano-frame rotation is exactly pano_to_perspective.rotation_matrix(yaw,
pitch) = R_off. With VGGT giving world→crop_cam (R_v, t_v):

    X_cam  = R_v (X - C)                       # VGGT extrinsic, C = center
    d_pano = R_off · d_cam                      # crop→pano (pano_to_perspective)
  ⇒ X_pano = R_off · R_v · (X - C) = R_wp (X - C),  R_wp = R_off · R_v

R_wp is the world→pano-view rotation the LONLAT rasterizer needs.

Usage: make_pano_dataset.py <scene_pano_dir> <out_json>
"""
import sys, os, re, json, importlib.util, numpy as np
# Load colmap_loader.py directly (avoid scene/__init__.py -> simple_knn import)
_spec = importlib.util.spec_from_file_location(
    "colmap_loader", "/workspace/gaussian-splatting/scene/colmap_loader.py")
_cl = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_cl)
read_extrinsics_binary, qvec2rotmat = _cl.read_extrinsics_binary, _cl.qvec2rotmat
sys.path.insert(0, "/workspace/gaussian-splatting/pano_pipeline")
from pano_to_perspective import rotation_matrix  # Ry(yaw)·Rx(pitch), cam→pano

scene_dir, out_json = sys.argv[1], sys.argv[2]
ext = read_extrinsics_binary(os.path.join(scene_dir, "sparse/0/images.bin"))

name_re = re.compile(r"pano_(\d+)_y([+-]\d+)_p([+-]\d+)")
panos = {}  # idx -> list of (off_mag, yaw, pitch, R_v, C)
for img in ext.values():
    m = name_re.search(img.name)
    if not m:
        continue
    idx, yaw, pitch = int(m.group(1)), float(m.group(2)), float(m.group(3))
    R_v = qvec2rotmat(img.qvec)               # world->cam
    t_v = np.array(img.tvec)
    C = -R_v.T @ t_v                           # camera center (world)
    panos.setdefault(idx, []).append((abs(yaw) + abs(pitch), yaw, pitch, R_v, C))

cams = []
for idx in sorted(panos):
    crops = panos[idx]
    C = np.mean([c[4] for c in crops], axis=0)           # shared center
    _, yaw, pitch, R_v, _ = min(crops, key=lambda c: c[0])  # front-most crop
    R_off = rotation_matrix(yaw, pitch)                   # cam->pano
    R_wp = R_off @ R_v                                    # world->pano-view
    T = -R_wp @ C
    img_path = os.path.join(scene_dir, "panoramas", f"pano_{idx:04d}.jpg")
    if not os.path.exists(img_path):
        continue
    cams.append({"idx": idx, "image": img_path,
                 "R_wp": R_wp.tolist(), "T": T.tolist(), "C": C.tolist(),
                 "n_crops": len(crops), "ref_yaw": yaw, "ref_pitch": pitch})

centers = np.array([c["C"] for c in cams])
extent = float(np.linalg.norm(centers - centers.mean(0), axis=1).max() * 1.1)
out = {"scene_dir": scene_dir,
       "point_cloud": os.path.join(scene_dir, "sparse/0/points3D.ply"),
       "cameras_extent": extent, "cameras": cams}
with open(out_json, "w") as f:
    json.dump(out, f, indent=1)
print(f"OK {len(cams)} pano cameras | extent={extent:.3f} | "
      f"centers span {centers.ptp(0).round(2).tolist()}")
