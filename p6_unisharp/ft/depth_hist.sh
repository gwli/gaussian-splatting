#!/bin/bash
# Task 4: compare predicted gaussian-depth distribution before/after fine-tune.
# Runs UniSHARP inference on one held-out 023 frame with the pretrained vs the
# fine-tuned checkpoint, then histograms the gaussians' radial distance ||xyz||.
#   bash depth_hist.sh
set -u
ROOT=/raid/git/gaussian-splatting; US=$ROOT/p6_unisharp/UniSHARP
PT=nvcr.io/nvidia/pytorch:24.12-py3
FRAME="${FRAME:-00200}"                       # a 023 val frame (193-240)
IMG_SRC=$ROOT/p6_unisharp/ft/data/scene_023hf/${FRAME}.jpg
LONG="${LONG:-$ROOT/p6_unisharp/ft/runs/a_scene_023hf_train/unified_feature_20260630_070556}"
FT_CKPT="${FT_CKPT:-$LONG/step_0003000.pt}"
[ -f "$IMG_SRC" ] || { echo "no frame $IMG_SRC"; exit 2; }
# stage the input where infer expects it
cp "$IMG_SRC" "$US/inputs/dh_${FRAME}.jpg"; chmod -R 777 "$ROOT/p6_unisharp" 2>/dev/null || true

run_infer(){  # $1=tag $2=ckpt_abs
  docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --user 0:0 \
    -e HF_HOME=/w/p6_unisharp/.hf -e TORCH_HOME=/w/p6_unisharp/.torch -e TORCH_CUDA_ARCH_LIST=9.0 \
    -v "$ROOT":/w -w /w/p6_unisharp/UniSHARP "$PT" bash -c "
    . /w/p6_unisharp/venv/bin/activate
    export PYTHONPATH=/w/p6_unisharp/UniSHARP:/w/p6_unisharp/UniSHARP/UniK3D
    python scripts/infer_unisharp.py --checkpoint '$2' \
      --image inputs/dh_${FRAME}.jpg --camera panorama --save-ply --out-dir outputs/dh_$1" \
    2>&1 | grep -aiE "saved|ply|gaussian|error|traceback" | tail -3
}
echo ">> infer PRETRAINED"; run_infer pre /w/p6_unisharp/UniSHARP/weights/pretained_model.pt
echo ">> infer FT step3000"; run_infer ft "/w/${FT_CKPT#$ROOT/}"
chmod -R 777 "$ROOT/p6_unisharp" 2>/dev/null || true

# histogram radial distance of gaussians from each ply
docker run --rm --user 0:0 -v "$ROOT":/w -w /w "$PT" bash -c "
  . /w/p6_unisharp/venv/bin/activate 2>/dev/null
  python /w/p6_unisharp/ft/depth_hist.py --frame ${FRAME}" 2>&1 | grep -aiE "saved|median|p90|frac|error|Traceback"
echo ">> hist -> p6_unisharp/ft/runs/depth_hist_${FRAME}.png"
