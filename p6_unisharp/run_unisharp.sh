#!/bin/bash
# Build a persistent venv (torch 2.8 + UniSHARP reqs + UniK3D) once, then run
# single-panorama inference. Panorama path doesn't need 3dgeer (lazy import).
#   run_unisharp.sh <input_image> [out_subdir]
set -e
ROOT=/raid/git/gaussian-splatting
US=$ROOT/p6_unisharp/UniSHARP
PT=nvcr.io/nvidia/pytorch:24.12-py3
IMG="${1:-inputs/watertown_027.jpg}"; OUT="${2:-outputs/watertown_027}"
mkdir -p "$ROOT/p6_unisharp/.pipcache"; chmod -R 777 "$ROOT/p6_unisharp" 2>/dev/null || true

docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --user 0:0 \
  -e PIP_CACHE_DIR=/w/p6_unisharp/.pipcache -e HF_HOME=/w/p6_unisharp/.hf \
  -e TORCH_HOME=/w/p6_unisharp/.torch -e TORCH_CUDA_ARCH_LIST=9.0 \
  -v "$ROOT":/w -w /w/p6_unisharp/UniSHARP $PT bash -c '
  set -e
  VENV=/w/p6_unisharp/venv
  if [ ! -f $VENV/.ready ]; then
    echo ">> building venv (one-time, ~10 min)"
    python -m venv $VENV
    . $VENV/bin/activate
    pip install -q --upgrade pip
    pip install -q torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126 2>&1 | tail -2
    pip install -q -r requirements.txt 2>&1 | tail -3
    pip uninstall -y opencv-python opencv-contrib-python 2>/dev/null | tail -0
    pip install -q opencv-python-headless 2>&1 | tail -1   # GUI build needs libxcb (absent)
    pip install -q -e UniK3D 2>&1 | tail -3
    touch $VENV/.ready
    echo ">> venv ready"
  fi
  . $VENV/bin/activate
  echo ">> torch: $(python -c "import torch;print(torch.__version__, torch.cuda.is_available())")"
  export PYTHONPATH=/w/p6_unisharp/UniSHARP:/w/p6_unisharp/UniSHARP/UniK3D
  python scripts/infer_unisharp.py \
    --checkpoint weights/pretained_model.pt \
    --image "'"$IMG"'" --camera panorama --save-ply \
    --out-dir "'"$OUT"'" 2>&1 | tail -30
' 2>&1 | grep -avE "Copyright|NVIDIA Release|Idiap|Caffe|Google|NEC|Deepmind|Facebook|reserved|NYU|This container|By pulling|^==|PyTorch Version|SHMEM|recommend|insufficient|docs.nvidia|governed|Various|developer|^$"
echo ">> outputs in p6_unisharp/UniSHARP/$OUT"
ls -la "$US/$OUT" 2>/dev/null
