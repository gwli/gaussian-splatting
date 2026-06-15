#!/bin/bash
# Runner for select_compare.py (frame-selection 3-way) on the fused gsplat backend.
#   $1 = pool_cams.json   $2 = n_train   $3 = n_test   $4 = iters   $5 = sel_iters
set -e
ROOT=/raid/git/gaussian-splatting
PT=nvcr.io/nvidia/pytorch:24.12-py3
CAMS="$1"; NTR="${2:-78}"; NTE="${3:-12}"; ITERS="${4:-7000}"; SEL="${5:-2000}"
mkdir -p "$ROOT/p3_pano/.torch_ext_cache"
GLM_DST="$ROOT/p3_pano/gsplat/gsplat/cuda/csrc/third_party/glm"
[ -f "$GLM_DST/glm/gtc/type_ptr.hpp" ] || cp -r "$ROOT/submodules/diff-gaussian-rasterization/third_party/glm/glm" "$GLM_DST/" 2>/dev/null || true
PATCH="$ROOT/p3_pano/gsplat_equirect_kernel.patch"
if [ -f "$PATCH" ] && ! grep -q 'EQUIRECT' "$ROOT/p3_pano/gsplat/gsplat/cuda/include/Common.h" 2>/dev/null; then
  ( cd "$ROOT/p3_pano/gsplat" && git apply "$PATCH" ) || true
fi
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --user 0:0 \
  -e TORCH_EXTENSIONS_DIR=/w/p3_pano/.torch_ext_cache -e TORCH_CUDA_ARCH_LIST=9.0 \
  -e PYTHONPATH=/w/p3_pano/gsplat -e TORCH_HOME=/wcache \
  -v "$ROOT":/w -v "$ROOT/p2_vggt/weights":/wcache -w /w $PT bash -c "
  pip install -q --no-deps ninja rich jaxtyping plyfile 2>&1 | tail -1
  python /w/p3_pano/select_compare.py /w/$CAMS $NTR $NTE $ITERS $SEL
" 2>&1 | grep -vaE "DEPRECATION|notice|satisfied|Copyright|Various|governed|developer|terms|^==|PyTorch Version|NVIDIA Release|Idiap|Caffe|Google|NEC|Deepmind|Facebook|reserved|NYU|This container|By pulling|^$|SHMEM|recommend|insufficient"
