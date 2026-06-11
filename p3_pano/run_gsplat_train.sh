#!/bin/bash
# T-F2 runner: train a scene on the gsplat backend in the nvcr container.
# gsplat JIT-compiles its CUDA backend on first import (~16 min) into a
# persistent TORCH_EXTENSIONS_DIR cache, so later runs are fast.
#   $1 = scene dir (host path)   $2 = iters (default 7000)   $3 = extra (e.g. --no-eval)
set -e
ROOT=/raid/git/gaussian-splatting
PT=nvcr.io/nvidia/pytorch:24.12-py3
SCENE_HOST="$1"; ITERS="${2:-7000}"; EXTRA="${3:-}"
SCENE_REL=$(realpath --relative-to="$ROOT" "$SCENE_HOST")
mkdir -p "$ROOT/p3_pano/.torch_ext_cache"

# gsplat's CUDA JIT needs glm headers (its third_party/glm submodule is empty in
# our gitignored clone). Vendor them from the diff-gaussian-rasterization submodule.
GLM_DST="$ROOT/p3_pano/gsplat/gsplat/cuda/csrc/third_party/glm"
if [ ! -f "$GLM_DST/glm/gtc/type_ptr.hpp" ]; then
  echo ">> vendoring glm headers into gsplat third_party"
  cp -r "$ROOT/submodules/diff-gaussian-rasterization/third_party/glm/glm" "$GLM_DST/" 2>/dev/null || true
fi

docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --user 0:0 \
  -e TORCH_EXTENSIONS_DIR=/w/p3_pano/.torch_ext_cache \
  -e TORCH_CUDA_ARCH_LIST=9.0 \
  -e PYTHONPATH=/w/p3_pano/gsplat \
  -v "$ROOT":/w -w /w $PT bash -c "
  pip install -q --no-deps ninja rich jaxtyping 2>&1 | tail -1
  python /w/p3_pano/train_gsplat.py /w/$SCENE_REL $ITERS $EXTRA
" 2>&1 | grep -vE "DEPRECATION|notice|satisfied|Copyright|Various|governed|developer|terms|^==|PyTorch Version|NVIDIA Release|Idiap|Caffe|Google|NEC|Deepmind|Facebook|reserved|NYU|This container|By pulling|^$|SHMEM|recommend|insufficient"
