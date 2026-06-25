#!/bin/bash
# T-F6 runner: direct-pano training on the gsplat cubemap backend, in the nvcr
# container. Persistent TORCH_EXTENSIONS_DIR so gsplat's CUDA JIT compiles once;
# auto-vendors glm headers (gsplat third_party/glm submodule is empty here).
#   $1 = pano_cams.json (repo-relative ok)   $2 = out_dir   $3 = iters   $4 = width   $5 = face
set -e
ROOT=/raid/git/gaussian-splatting
PT=nvcr.io/nvidia/pytorch:24.12-py3
CAMS="$1"; OUT="$2"; ITERS="${3:-7000}"; WIDTH="${4:-1024}"; FACE="${5:-512}"; SPH="${6:-}"
mkdir -p "$ROOT/p3_pano/.torch_ext_cache"
GLM_DST="$ROOT/p3_pano/gsplat/gsplat/cuda/csrc/third_party/glm"
if [ ! -f "$GLM_DST/glm/gtc/type_ptr.hpp" ]; then
  cp -r "$ROOT/submodules/diff-gaussian-rasterization/third_party/glm/glm" "$GLM_DST/" 2>/dev/null || true
fi
# Apply the fused equirect CUDA kernel patch (T-F8) if not already applied, so a
# fresh gsplat clone gets the "equirect" camera_model. Idempotent.
PATCH="$ROOT/p3_pano/gsplat_equirect_kernel.patch"
if [ -f "$PATCH" ] && ! grep -q 'EQUIRECT' "$ROOT/p3_pano/gsplat/gsplat/cuda/include/Common.h" 2>/dev/null; then
  echo ">> applying gsplat_equirect_kernel.patch"; ( cd "$ROOT/p3_pano/gsplat" && git apply "$PATCH" ) || true
fi

docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --user 0:0 \
  -e TORCH_EXTENSIONS_DIR=/w/p3_pano/.torch_ext_cache \
  -e TORCH_CUDA_ARCH_LIST=9.0 \
  -e PYTHONPATH=/w/p3_pano/gsplat \
  -e GSPLAT_EQUIRECT_COMPILE="${GSPLAT_EQUIRECT_COMPILE:-0}" \
  -e GSPLAT_EQUIRECT_FUSED="${GSPLAT_EQUIRECT_FUSED:-1}" \
  -e GROW_GRAD2D="${GROW_GRAD2D:-0.0002}" \
  -e REFINE_STOP_FRAC="${REFINE_STOP_FRAC:-0.5}" \
  -v "$ROOT":/w -w /w $PT bash -c "
  pip install -q --no-deps ninja rich jaxtyping plyfile 2>&1 | tail -1
  if [ \"$SPH\" = \"sph\" ]; then
    python /w/p3_pano/train_pano_gsplat_sph.py /w/$CAMS /w/$OUT $ITERS $WIDTH
  else
    python /w/p3_pano/train_pano_gsplat.py /w/$CAMS /w/$OUT $ITERS $WIDTH $FACE
  fi
" 2>&1 | grep -vaE "DEPRECATION|notice|satisfied|Copyright|Various|governed|developer|terms|^==|PyTorch Version|NVIDIA Release|Idiap|Caffe|Google|NEC|Deepmind|Facebook|reserved|NYU|This container|By pulling|^$|SHMEM|recommend|insufficient"
