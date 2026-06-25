#!/bin/bash
# T-F3 real-data validation: run sliding-window VGGT on a >300-crop scene and
# compare sequential vs global Sim3 alignment. Repo mounted at the path
# vggt_window.py hardcodes (/workspace/gaussian-splatting); weights at /wcache.
#   $1 = scene dir (has images/ with the crops)   $2 = win   $3 = overlap   $4 = conf
set -e
ROOT=/raid/git/gaussian-splatting
PT=nvcr.io/nvidia/pytorch:24.12-py3
SCENE_HOST="$1"; WIN="${2:-250}"; OVL="${3:-50}"; CONF="${4:-1.5}"
SCENE_REL=$(realpath --relative-to="$ROOT" "$SCENE_HOST")

docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --user 0:0 \
  -v "$ROOT":/workspace/gaussian-splatting \
  -v "$ROOT/p2_vggt/weights":/wcache:ro \
  -e VGGT_PGO="${VGGT_PGO:-0}" \
  -w /workspace/gaussian-splatting/p2_vggt $PT bash -c "
  pip install -q --no-deps einops safetensors huggingface_hub 2>&1 | tail -1
  python vggt_window.py /workspace/gaussian-splatting/$SCENE_REL $WIN $OVL $CONF
" 2>&1 | grep -vaE "DEPRECATION|notice|satisfied|Copyright|Various|governed|developer|terms|^==|PyTorch Version|NVIDIA Release|Idiap|Caffe|Google|NEC|Deepmind|Facebook|reserved|NYU|This container|By pulling|^$|SHMEM|recommend|insufficient|FutureWarning|weights_only|model.load_state"
