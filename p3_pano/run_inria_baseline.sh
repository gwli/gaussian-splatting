#!/bin/bash
# T-F2 baseline: INRIA backend on the SAME scene/split as train_gsplat.py, for a
# matched held-out PSNR + wall-clock comparison. Builds the 3 CUDA extensions,
# then trains train.py with --eval (every-8th holdout, llffhold=8) for N iters.
#   $1 = scene dir (host path)   $2 = iters (default 7000)
set -e
ROOT=/raid/git/gaussian-splatting
PT=nvcr.io/nvidia/pytorch:24.12-py3
SCENE_HOST="$1"; ITERS="${2:-7000}"
SCENE_REL=$(realpath --relative-to="$ROOT" "$SCENE_HOST")

docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --user 0:0 \
  -e TORCH_EXTENSIONS_DIR=/w/p3_pano/.torch_ext_cache \
  -v "$ROOT":/w -w /w $PT bash -c "
  export TORCH_CUDA_ARCH_LIST=9.0
  pip install -q --no-build-isolation --no-deps /w/submodules/diff-gaussian-rasterization /w/submodules/simple-knn /w/submodules/fused-ssim 2>&1 | tail -2
  pip install -q --no-deps plyfile 2>&1 | tail -1
  cd /w
  /usr/bin/time -v python train.py -s /w/$SCENE_REL -m /w/$SCENE_REL/output_inria_eval \
    --eval --iterations $ITERS --test_iterations $ITERS --save_iterations $ITERS \
    --disable_viewer 2>&1 | grep -aE 'ITER|PSNR|it/s|Elapsed|Training|wall|Maximum resident'
" 2>&1 | grep -vE "DEPRECATION|notice|satisfied|Copyright|Various|governed|developer|terms|^==|PyTorch Version|NVIDIA Release|Idiap|Caffe|Google|NEC|Deepmind|Facebook|reserved|NYU|This container|By pulling|^$|SHMEM|recommend|insufficient"
