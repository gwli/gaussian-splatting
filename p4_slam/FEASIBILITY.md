# MASt3R-SLAM (P2.2 streaming reconstruction) — feasibility assessment

Goal: "边飞边重建" — real-time/streaming reconstruction so the drone has a
coarse model on landing. Evaluated [MASt3R-SLAM](https://github.com/rmurai0610/MASt3R-SLAM).

## What's confirmed working / available
- ✅ Repo clones (core). The only failed submodule is `thirdparty/in3d`
  (an imgui GUI viewer needing git-lfs) — **not needed headless**.
- ✅ Headless supported: `main.py --no-viz --dataset <path> [--calib f.yaml]`.
- ✅ Input formats (`mast3r_slam/dataloader.py`): **MP4** (`MP4Dataset`) or a
  **folder of PNGs** (`RGBFiles`), plus TUM/EuRoC/ETH3D/7-Scenes/webcam/realsense.
- ✅ Checkpoints downloadable via curl (sandbox off) from naverlabs — 2.9 GB
  fetched: metric `.pth` (2.6G) + retrieval `.pth` (8M) + codebook `.pkl` (257M).

## Blockers / risks
1. **Build chain vs torch version.** Repo pins **torch 2.5.1**; our container is
   **torch 2.6 (nv24.12) / CUDA 12.6**. The build needs:
   - `lietorch` (git, princeton-vl) — a CUDA extension notoriously sensitive to
     the torch version. A time-boxed build attempt against torch 2.6 did **not**
     produce a success marker (inconclusive / classic version-mismatch trouble).
   - `thirdparty/mast3r` (+ `asmk` C++ build, `dust3r`)
   - the SLAM package's own CUDA ext (`gn_kernels.cu`, `matching_kernels.cu`,
     DROID-SLAM-derived).
   Getting all of these to compile together against torch 2.6 is a multi-hour,
   high-risk effort. The reliable path is the repo's intended env: a dedicated
   **conda env with torch==2.5.1 + matching pytorch-cuda**, or their Docker.
2. **Domain mismatch — monocular *perspective* SLAM vs our 360° data.**
   MASt3R-SLAM expects a normal pinhole video. Our capture is equirectangular.
   Feeding equirect/fisheye directly won't work; we'd feed a **forward-crop
   sequence** (e.g. `pano_XXXX_y+000_p+00`). At our sparse pano sampling
   (~90 frames/flight) consecutive-frame overlap may be too low for stable
   tracking — a denser perspective render from the equirect video would be
   needed.

## Recommendation
- **Not worth forcing into the torch-2.6 container.** If pursued, do it in a
  **separate torch-2.5.1 conda env** (or the official MASt3R-SLAM Docker),
  isolated from this repo's build.
- For our actual need, **VGGT already covers feed-forward SfM** (10–100× faster
  than COLMAP, validated on all 7 scenes). True *streaming* (while-flying)
  reconstruction is a distinct capability that would require:
  (a) a torch-2.5.1 MASt3R-SLAM env, and
  (b) a dense forward-facing perspective stream from the drone (not 360° pano).
- Assets are cached under `p4_slam/MASt3R-SLAM/checkpoints/` (git-ignored) so a
  future torch-2.5.1 attempt can skip the 2.9 GB re-download.

## Status: assessed, not run to completion (research-scale, deferred).
