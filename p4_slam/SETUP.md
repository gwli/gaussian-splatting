# MASt3R-SLAM — one-time setup (reproducible run)

Everything heavy here is git-ignored (the `MASt3R-SLAM/` clone, 2.9 GB
checkpoints, `lietorch_src/`, derived sequences). To reproduce on a fresh
machine:

```bash
cd p4_slam

# 1. Clone the upstream repo (recursive for thirdparty/mast3r etc.)
git clone https://github.com/rmurai0610/MASt3R-SLAM.git --recursive

# 2. Download the 3 checkpoints (2.9 GB) into MASt3R-SLAM/checkpoints/
#    (see MASt3R-SLAM/README.md "Setup the checkpoints" — metric .pth,
#     retrieval .pth, codebook .pkl)

# 3. Vendor lietorch locally so its build doesn't depend on gitlab.com
#    (its eigen submodule URL is gitlab, which is flaky / was down).
git clone --depth 1 https://github.com/princeton-vl/lietorch.git lietorch_src
cp -r MASt3R-SLAM/thirdparty/eigen/. lietorch_src/eigen/

# 4. Build + run (one command). Builds a persistent mast3r-slam:built image,
#    auto-applies our headless patches, then runs on the default sequence.
bash run_slam_full.sh
```

Notes:
- Our patches to the upstream clone (faiss/asmk import path, optional
  `pyrealsense2`, lazy GUI import, **sm_90** gencode) live in
  `mast3r_slam_patches.diff` and are auto-applied by `_slam_build.sh` (idempotent
  `git apply --check`).
- `bash run_slam_full.sh <dataset_dir> <tag>` runs on a custom sequence.
- `bash run_slam_full.sh --clean` removes the multi-GB `mast3r-slam:built` image.
- For a dense forward-perspective stream from a raw `.insv` (best tracking), use
  `make_dense_perspective.sh <insv> <out_dir> [fps] [start] [dur]` — see T-F1 in
  `../tasks.md` and `FEASIBILITY.md`.
