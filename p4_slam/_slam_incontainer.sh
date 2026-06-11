#!/bin/bash
# Runs INSIDE nvcr.io/nvidia/pytorch:24.12-py3. Repo mounted at /w.
# Single invocation: build CUDA deps + faiss/asmk retrieval stack, then run headless.
set -e
cd /w/p4_slam/MASt3R-SLAM
export QT_QPA_PLATFORM=offscreen
PIP="pip install -q --no-build-isolation"

echo ">> [1/5] lietorch (CUDA ext, ~slow)"
pip install -q --no-build-isolation "lietorch @ git+https://github.com/princeton-vl/lietorch.git"

echo ">> [2/5] mast3r (editable, +roma)"
pip install -q -e thirdparty/mast3r

echo ">> [3/5] MASt3R-SLAM package (gn/matching CUDA ext)"
pip install -q --no-build-isolation -e .

echo ">> [4/5] retrieval stack: faiss-cpu + asmk (hamming cython ext)"
pip install -q faiss-cpu pyaml
cd thirdparty/mast3r/asmk
python setup.py build_ext --inplace
pip install -q --no-build-isolation .
cd /w/p4_slam/MASt3R-SLAM

echo ">> [5/5] import sanity"
python - <<'PY'
import lietorch, mast3r_slam, faiss, asmk
import mast3r_slam_backends as _b
print("IMPORTS_OK: lietorch + mast3r_slam + backend + faiss + asmk")
PY

echo ">> RUN MASt3R-SLAM (headless, no-calib) on 90-frame forward sequence"
python main.py --no-viz \
    --dataset /w/p4_slam/seq_023_front \
    --config config/base.yaml \
    --save-as seq023front

echo ">> outputs:"
ls -la logs/ 2>/dev/null || true
find . -newermt '-30 minutes' -name '*.txt' -o -newermt '-30 minutes' -name '*.ply' 2>/dev/null | head
echo ">> DONE"
