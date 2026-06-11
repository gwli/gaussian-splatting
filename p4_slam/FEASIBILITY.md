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

## Run status (2026-06-11)
Build is confirmed; the headless **run** then surfaces MASt3R-SLAM's
retrieval/loop-closure stack, which `main.py` imports unconditionally
(`mast3r_utils -> retrieval_database -> mast3r.retrieval.processor`):
needs **faiss** + **asmk** (the latter a cython build under
`thirdparty/mast3r/asmk`). Also a benign warning: RoPE2D CUDA kernel not built
→ slow pytorch fallback (fine).

These are finite, known deps (`faiss-cpu` + building asmk), but each attempt is
a ~12-min full rebuild and the deps surface one-by-one — a dependency rabbit
hole. Plus the input caveat stands: 90 sparse forward crops may not track at
video-rate expectations.

Decision: stopped at **build confirmed + run blocked on faiss/asmk retrieval
stack**. Finishing the run = install faiss-cpu, build asmk, then
`main.py --no-viz`; optionally feed a denser perspective stream.

## Status: build CONFIRMED working in-container. Remaining to demonstrate
streaming: feed a dense forward-**perspective** sequence (PNG folder / mp4) —
our 360° panoramas need a perspective re-render — and run
`main.py --no-viz --dataset <seq>`.
