#!/bin/bash
# Eval a checkpoint on a held-out sim val scene -> PSNR/SSIM/LPIPS.
#   bash eval_a.sh <val_scene> <checkpoint_relpath_or_abs> <out_tag>
# e.g. bash eval_a.sh scene_023hf_val weights/pretained_model.pt pretrained
set -u
ROOT=/raid/git/gaussian-splatting; FT=$ROOT/p6_unisharp/ft
PT=nvcr.io/nvidia/pytorch:24.12-py3
VAL="${1:?val scene}"; CKPT="${2:?checkpoint}"; TAG="${3:-eval}"
FAST="${FAST:-0}"   # 1 = skip LPIPS (faster)
SIM_PAIR_MAX_TR="${SIM_PAIR_MAX_TR:-6.0}"; SIM_PAIR_MIN_OVERLAP="${SIM_PAIR_MIN_OVERLAP:-0.1}"
FAR="${FAR:-300}"
# resolve checkpoint to a /w path
case "$CKPT" in /*) CKW="/w/${CKPT#/raid/git/gaussian-splatting/}";; *) CKW="/w/p6_unisharp/UniSHARP/$CKPT";; esac
OUT=/w/p6_unisharp/ft/runs/eval_${VAL}_${TAG}
mkdir -p "$FT/runs"; chmod -R 777 "$FT" 2>/dev/null || true
FASTFLAG=""; [ "$FAST" = "1" ] && FASTFLAG="--fast-metrics"

docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --user 0:0 \
  -e HF_HOME=/w/p6_unisharp/.hf -e TORCH_HOME=/w/p6_unisharp/.torch -e HF_HUB_OFFLINE=1 \
  -e TORCH_CUDA_ARCH_LIST=9.0 -e CUDA_LAUNCH_BLOCKING=1 \
  -e SIM_PAIR_MAX_TR="$SIM_PAIR_MAX_TR" -e SIM_PAIR_MIN_OVERLAP="$SIM_PAIR_MIN_OVERLAP" \
  -v "$ROOT":/w -w /w/p6_unisharp/UniSHARP "$PT" bash -c "
  set -e
  . /w/p6_unisharp/venv/bin/activate
  export PYTHONPATH=/w/p6_unisharp/ft:/w/p6_unisharp/UniSHARP:/w/p6_unisharp/UniSHARP/UniK3D
  python /w/p6_unisharp/ft/eval_a.py \
    --checkpoint $CKW --dataset sim \
    --data-root /w/p6_unisharp/ft/data --sim-pose-root /w/p6_unisharp/ft/poses \
    --manifest-file /w/p6_unisharp/ft/manifests/${VAL}.txt \
    --pair-max-translation-m $SIM_PAIR_MAX_TR --pair-min-overlap $SIM_PAIR_MIN_OVERLAP \
    --max-index-gap 16 --sim-far-depth-invalid-m $FAR --validation-batch-size 4 \
    --out-dir $OUT $FASTFLAG
" 2>&1 | grep -avE "Copyright|NVIDIA Release|reserved|This container|By pulling|^==|PyTorch Version|docs.nvidia|governed|recommend|insufficient|SHMEM|^$|gpus all|memlock|Driver"
chmod -R 777 "$FT" 2>/dev/null || true
echo ">> metrics:"; cat ${FT}/runs/eval_${VAL}_${TAG}/validation_metrics_sim.csv 2>/dev/null