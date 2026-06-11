#!/bin/bash
# T-F4: VGGT SfM WITH bundle adjustment (--use_ba), using locally-cached weights
# to bypass the api.github.com 403 rate-limit that blocked T-D5.
#   $1 = scene dir (has images/; gets its own sparse/0 written)   $2 = conf (1.5)
set -e
ROOT=/raid/git/gaussian-splatting
PT=nvcr.io/nvidia/pytorch:24.12-py3
SCENE_HOST="$1"; CONF="${2:-1.5}"
SCENE_REL=$(realpath --relative-to="$ROOT" "$SCENE_HOST")

docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --user 0:0 \
  -e TORCH_HOME=/w/p2_vggt/hub_cache \
  -e VGGSFM_TRACKER_PT=/w/p2_vggt/hub_cache/checkpoints/vggsfm_v2_tracker.pt \
  -e DINOV2_LOCAL_DIR=/w/p2_vggt/hub_cache/dinov2_repo \
  -v "$ROOT":/w -v "$ROOT/p2_vggt/weights":/wsrc:ro -v "$ROOT/p2_vggt/weights":/wcache:ro -w /w $PT bash -c "
  mkdir -p /w/p2_vggt/hub_cache/hub/checkpoints
  ln -sf /wsrc/model.pt /w/p2_vggt/hub_cache/hub/checkpoints/model.pt
  pip install -q --no-deps einops safetensors trimesh huggingface_hub 'pycolmap==3.10.0' \
      'numpy<2' pyceres kornia kornia_rs 'hydra-core==1.3.2' 'omegaconf==2.3.0' 'antlr4-python3-runtime==4.9.3' 2>&1 | tail -2
  # lightglue is not on PyPI — install from git (clone works; only the github API rate-limits)
  pip install -q --no-deps 'git+https://github.com/cvg/LightGlue.git' 2>&1 | tail -2
  cd /w/p2_vggt/vggt && export PYTHONPATH=\$PWD
  python demo_colmap.py --scene_dir /w/$SCENE_REL --use_ba --conf_thres_value $CONF \
    --max_query_pts ${MAXQ:-1024} --query_frame_num ${QFN:-4} 2>&1 | tail -45
" 2>&1 | grep -vaE "DEPRECATION|notice|satisfied|Copyright|Various|governed|developer|terms|^==|PyTorch Version|NVIDIA Release|Idiap|Caffe|Google|NEC|Deepmind|Facebook|reserved|NYU|This container|By pulling|^$|SHMEM|recommend|insufficient|docker run --gpus|docs.nvidia"
