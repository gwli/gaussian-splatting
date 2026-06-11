#!/bin/bash
# Runs INSIDE nvcr.io/nvidia/pytorch:24.12-py3. Repo at /w. Build deps only.
set -e
cd /w/p4_slam/MASt3R-SLAM
export TORCH_CUDA_ARCH_LIST="9.0"   # Hopper (H100/H200) — lietorch + backend CUDA arch
PIPQ="pip install -q --no-build-isolation"

# [0/5] Apply our headless/run patches if not already applied, so a FRESH clone
# of MASt3R-SLAM is one-command runnable (faiss/asmk import path, optional
# pyrealsense2, lazy GUI import, sm_90 gencode). Idempotent.
PATCH=/w/p4_slam/mast3r_slam_patches.diff
if [ -f "$PATCH" ] && git apply --check "$PATCH" 2>/dev/null; then
  echo ">> [0/5] applying mast3r_slam_patches.diff"; git apply "$PATCH"
else
  echo ">> [0/5] patches already applied (or not needed)"
fi
echo ">> [1/5] lietorch (local clone; eigen vendored to avoid gitlab submodule)"
$PIPQ -e /w/p4_slam/lietorch_src
echo ">> [2/5] mast3r (editable, +roma)"
pip install -q -e thirdparty/mast3r
echo ">> [3/5] MASt3R-SLAM package (gn/matching CUDA ext)"
$PIPQ -e .
echo ">> [4/5] retrieval stack: faiss-cpu + asmk"
pip install -q faiss-cpu pyaml
( cd thirdparty/mast3r/asmk && python setup.py build_ext --inplace && pip install -q --no-build-isolation . )
echo ">> [5/5] import sanity"
python - <<'PY'
import lietorch, mast3r_slam, faiss, asmk
import mast3r_slam_backends as _b
print("IMPORTS_OK: lietorch + mast3r_slam + backend + faiss + asmk")
PY
echo ">> BUILD_DONE"
