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

## Build (CORRECTED 2026-06-11)
Earlier I reported the build as "inconclusive / likely torch-version trouble".
That was **wrong**. The first attempt failed only because pip **build isolation**
pulled a fresh `torch` compiled with **CUDA 13.0** into the build env, which
mismatched the system CUDA 12.6 — *not* a torch-2.5.1-vs-2.6 problem.

- ✅ **`lietorch` builds fine** against the container's torch 2.6 / CUDA 12.6
  when installed with **`--no-build-isolation`** (which the repo README
  specifies): wheel `lietorch-0.3` built, `import lietorch` OK.
- Remaining build pieces (`thirdparty/mast3r` + `roma`, the SLAM package's
  `gn_kernels.cu`/`matching_kernels.cu` CUDA ext) are being built the same way;
  status tracked in `data/8kpano/slam_fullbuild.log`.

So the build is **viable in our torch-2.6 container** with `--no-build-isolation`
— no separate torch-2.5.1 env required after all.

## Remaining real consideration (not a build blocker)
- **Domain mismatch — monocular *perspective* SLAM vs our 360° data.**
   MASt3R-SLAM expects a normal pinhole video. Our capture is equirectangular.
   Feeding equirect/fisheye directly won't work; we'd feed a **forward-crop
   sequence** (e.g. `pano_XXXX_y+000_p+00`). At our sparse pano sampling
   (~90 frames/flight) consecutive-frame overlap may be too low for stable
   tracking — a denser perspective render from the equirect video would be
   needed.

## Recommendation
- **Build is viable here** (torch-2.6 container, `--no-build-isolation`) — no
  separate torch-2.5.1 env needed. lietorch confirmed; mast3r + SLAM CUDA ext
  build status in `data/8kpano/slam_fullbuild.log`.
- The one genuine gap is **input**: MASt3R-SLAM is monocular *perspective* SLAM,
  so a streaming demo needs a **dense forward-facing perspective stream** from
  the drone (or a perspective re-render from the equirect video at high fps),
  not the 360° panoramas.
- For our current SfM need, **VGGT already covers it** (10–100× faster than
  COLMAP, validated on all 7 scenes). MASt3R-SLAM adds *streaming / while-flying*
  reconstruction, which is the next step if that capability is wanted.
- Assets cached under `p4_slam/MASt3R-SLAM/checkpoints/` (git-ignored, 2.9 GB).

## Build CONFIRMED (2026-06-11)
Full build succeeds in the torch-2.6 / CUDA-12.6 container with
`--no-build-isolation`:
- `lietorch` 0.3 — built + imports
- `thirdparty/mast3r` (+ `roma`) — editable install OK
- MASt3R-SLAM package incl. the `gn_kernels.cu`/`matching_kernels.cu` CUDA ext —
  `import mast3r_slam.backend` → **"backend ok"**.
(An `IMPORTS_OK` line was missing only due to a typo in my test string — a bad
ternary inside an `import` — not a real import failure; the `backend` import,
which exercises the compiled ext, succeeded.)

## RUN CONFIRMED — end-to-end (2026-06-11)
MASt3R-SLAM now **runs headless end-to-end** in our torch-2.6/CUDA-12.6
container and produces a trajectory + dense point cloud. Reproduce with
`p4_slam/run_slam_full.sh` (builds a persistent `mast3r-slam:built` image once,
then runs `main.py --no-viz --dataset p4_slam/seq_023_front --config
config/base.yaml`).

Getting from "build confirmed" to "runs" took four concrete fixes (all captured
in `p4_slam/mast3r_slam_patches.diff` + the build scripts):
1. **Retrieval stack** — `main.py` imports `mast3r.retrieval.processor`
   unconditionally → needs **faiss-cpu** + **asmk** (asmk's `hamming` cython ext
   built in place; the `.c` is pre-generated so no cython toolchain needed).
2. **`pyrealsense2`** import in `dataloader.py` pulled `libusb` (RealSense camera
   driver we don't use) → made the import optional (`try/except → rs=None`).
3. **`imgui`/`moderngl`/`in3d` GUI stack** imported at `main.py` top via
   `visualization` → moved `WindowMsg` to a local dataclass and made
   `run_visualization` a lazy import (only under non-`--no-viz`).
4. **CUDA arch** — this machine is **sm_90 (Hopper, H100/H200)** but `setup.py`
   compiled the `gn`/`matching` kernels only up to sm_86 → `no kernel image is
   available for execution on the device`. Added `-gencode …compute_90` and
   built lietorch with `TORCH_CUDA_ARCH_LIST=9.0`.
   - Side fix: lietorch's pip install clones `eigen` from **gitlab.com**, which
     was down (502/503). Vendored eigen 3.4.90 from `thirdparty/eigen` into a
     local `p4_slam/lietorch_src/eigen` and install lietorch `-e` from there —
     removes the gitlab dependency entirely.

### Result on our data (`seq_023_front`, 90 forward crops from scene_023 panos)
- **Tracked + reconstructed the first 16 keyframes** → TUM trajectory
  (`slam_output_seq023/trajectory_tum.txt`, 16 poses, plausible forward motion)
  + a 10.8 MB dense point cloud (`reconstruction.ply`, git-ignored — regenerable).
  Ran at ~3.4–3.8 FPS on one H100.
- Then **lost tracking at frame 16** and repeatedly "Failed to relocalize"
  against the 8-keyframe map → "Skipped frame 16", finished cleanly.
- This is exactly the **predicted domain caveat**: 90 sparse forward-crops from
  360° panoramas have too little frame-to-frame overlap for monocular
  perspective SLAM to keep lock. The fix for a real streaming demo is a **denser
  forward-perspective stream** (re-render the equirect video to perspective at
  higher fps, or feed the drone's native forward camera), not the sparse panos.

## Status: build + run BOTH CONFIRMED in-container. The streaming-reconstruction
capability works; demonstrating it across a *full* flight needs dense
forward-perspective input (our 360° pano sampling is too sparse to track past
the initial overlapping segment).
