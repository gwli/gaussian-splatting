# P2.1 — VGGT feed-forward SfM (replaces COLMAP)

[VGGT](https://github.com/facebookresearch/vggt) (Visual Geometry Grounded
Transformer, Meta + Oxford) predicts camera poses, depth, and a 3D point cloud
in **one forward pass**. We use it as a drop-in replacement for COLMAP's
Stage-3 SfM in the panorama→3DGS pipeline.

## Why

Measured on scene_023 (140 frames):

| | COLMAP (original) | COLMAP v2g (exhaustive) | **VGGT** |
|---|---|---|---|
| Stage 3 (SfM) | ~15 min | ~2.6 h | **96 s** |
| init points | ~100k | 119k | 100k |
| PSNR @ 7k iter | ~20 | 18 | **23.0** |
| Stage 4 (15k iter) | ~17 min | 18 min | 9 min |
| **total** | ~32 min | ~3 h | **~10.5 min** |

VGGT is the first config that is **both faster and comparable/better quality** —
the "fast AND good" that the P0/P1 knobs (which only traded quality for speed)
could not reach.

## Setup (one time)

These artifacts are git-ignored (large / third-party). Recreate with:

```bash
cd p2_vggt

# 1. Clone VGGT
git clone https://github.com/facebookresearch/vggt.git

# 2. Download the 1B checkpoint (~4.7 GB) to the weights cache
mkdir -p weights
curl -L "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt" \
     -o weights/model.pt
```

`vggt_sfm.sh` auto-patches `demo_colmap.py` (makes the lightglue/pyceres import
lazy so the feed-forward path needs no bundle-adjustment deps) and seeds the
weights into the container's `TORCH_HOME` cache so no HuggingFace access is
needed at run time.

## Usage

```bash
# scene_name | max_frames | iterations | conf_thres
bash ../pano_pipeline/vggt_sfm.sh scene_023 150 15000 1.5
```

Output goes to `data/8kpano/scenes/<scene>/vggt/` (kept separate from the COLMAP
`output/`), with `output/point_cloud/iteration_<N>/point_cloud.{ply,ksplat}`.

## Knobs & caveats

- **max_frames (150)**: VGGT attends across ALL frames at once → GPU memory is
  ~O(N²). 150 frames fit comfortably in 80 GB. For more frames, chunk the
  sequence or use the sliding-window variant. The script subsamples the scene's
  `input/` evenly down to this count.
- **conf_thres (1.5)**: depth-confidence cutoff for which pixels become 3D
  points. The demo default of **5.0 filters out ALL points on low-texture drone
  footage** → 0-point reconstruction. 1.5 keeps a dense ~100k cloud. Tune per
  scene if the cloud looks sparse/noisy.
- **--use_ba**: VGGT also has a bundle-adjustment mode (LightGlue + pycolmap)
  for higher accuracy. Not wired into `vggt_sfm.sh` yet; needs `lightglue` and
  `pyceres` installed. Feed-forward mode was sufficient here.
- **pycolmap==3.10.0**: VGGT's `np_to_pycolmap.py` uses the 3.10 Image API;
  newer pycolmap breaks with `Image has no attribute 'id'`.
