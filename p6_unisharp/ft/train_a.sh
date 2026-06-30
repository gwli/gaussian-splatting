#!/bin/bash
# A-tier fine-tune: load pretrained UniSHARP, freeze UniK3D encoder, gently adapt
# the depth prior + heads on our de-rotated watertown panos (photometric + weak
# pseudo-depth anchor). Drives the stock CLI via train_a.py (see its docstring).
#   bash p6_unisharp/ft/train_a.sh 027
# Env: STEPS(=30 smoke) BATCH(=2) LR0(=1e-4) UNIK3D_LR0(=1e-5 depth-prior nudge)
#      SIM_PAIR_MAX_TR(=6) SIM_PAIR_MIN_OVERLAP(=0.1) FAR(=300)
set -u
ROOT=/raid/git/gaussian-splatting
FT=$ROOT/p6_unisharp/ft; US=$ROOT/p6_unisharp/UniSHARP
PT=nvcr.io/nvidia/pytorch:24.12-py3
S="${1:-027}"; SC="${SC_NAME:-scene_${S}hf}"   # SC_NAME overrides (e.g. scene_023hf_train)
STEPS="${STEPS:-30}"; BATCH="${BATCH:-2}"; LR0="${LR0:-1e-4}"; LR1="${LR1:-1e-5}"
UNIK3D_LR0="${UNIK3D_LR0:-1e-5}"; UNIK3D_LR1="${UNIK3D_LR1:-1e-6}"
SIM_PAIR_MAX_TR="${SIM_PAIR_MAX_TR:-6.0}"; SIM_PAIR_MIN_OVERLAP="${SIM_PAIR_MIN_OVERLAP:-0.1}"
FAR="${FAR:-300}"; CKPT="${CKPT:-weights/pretained_model.pt}"
ENC_LR0="${ENC_LR0:-0}"; ENC_LR1="${ENC_LR1:-0}"   # >0 unfreezes UniK3D encoder (B-tier)
# MANIFEST_SCENES (newline/space-separated) trains on several scenes; SC names the run dir.
MANIFEST_SCENES="${MANIFEST_SCENES:-$SC}"

[ -d "$FT/data/$SC" ] || { echo "no dataset $FT/data/$SC — run build_ft_dataset.sh $S first"; exit 2; }
mkdir -p "$FT/manifests" "$FT/runs"
printf '%s\n' $MANIFEST_SCENES > "$FT/manifests/sim_train_scenes.txt"
echo ">> train scenes: $(tr '\n' ' ' < $FT/manifests/sim_train_scenes.txt)"
: > "$FT/manifests/_wild_roots_dummy.txt"      # dummy file for --wild-roots-file (exists=True)
chmod -R 777 "$FT" 2>/dev/null || true

# all exists=True roots we don't use are pointed at an existing dir to pass click validation;
# their dataset weights are 0 so they're never actually constructed.
DUM=/w/p6_unisharp/ft/data

docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --user 0:0 \
  -e HF_HOME=/w/p6_unisharp/.hf -e TORCH_HOME=/w/p6_unisharp/.torch -e HF_HUB_OFFLINE=1 \
  -e TORCH_CUDA_ARCH_LIST=9.0 \
  -e INIT_CKPT="/w/p6_unisharp/UniSHARP/$CKPT" \
  -e SIM_PAIR_MAX_TR="$SIM_PAIR_MAX_TR" -e SIM_PAIR_MIN_OVERLAP="$SIM_PAIR_MIN_OVERLAP" \
  -v "$ROOT":/w -w /w/p6_unisharp/UniSHARP "$PT" bash -c "
  set -e
  . /w/p6_unisharp/venv/bin/activate
  export PYTHONPATH=/w/p6_unisharp/ft:/w/p6_unisharp/UniSHARP:/w/p6_unisharp/UniSHARP/UniK3D
  python /w/p6_unisharp/ft/train_a.py train-feature \
    --out-root /w/p6_unisharp/ft/runs/a_${SC} \
    --steps $STEPS --warmup 5 --batch-size $BATCH --num-workers 1 \
    --log-every ${LOG_EVERY:-1} --vis-every ${VIS_EVERY:-0} --save-every ${SAVE_EVERY:-0} \
    --lr0 $LR0 --lr1 $LR1 \
    --unik3d-lr0 $UNIK3D_LR0 --unik3d-lr1 $UNIK3D_LR1 \
    --unik3d-encoder-lr0 $ENC_LR0 --unik3d-encoder-lr1 $ENC_LR1 \
    --lambda-depth 0.1 \
    --dataset-weight-sim 1 --dataset-weight-re10k 0 --dataset-weight-hm3d 0 \
    --dataset-weight-wildrgbd 0 --dataset-weight-dl3dv 0 --dataset-weight-scanetpp 0 \
    --data-root-sim /w/p6_unisharp/ft/data --sim-pose-root /w/p6_unisharp/ft/poses \
    --dataset-manifest-dir /w/p6_unisharp/ft/manifests \
    --sim-far-depth-invalid-m $FAR --sim-max-long-edge 512 --max-index-gap 16 \
    --data-root-hm3d $DUM --data-root-dl3dv $DUM --data-root-dl3dv-depth $DUM \
    --data-root-scanetpp $DUM --wild-roots-file /w/p6_unisharp/ft/manifests/_wild_roots_dummy.txt
" 2>&1 | grep -avE "Copyright|NVIDIA Release|reserved|This container|By pulling|^==|PyTorch Version|docs.nvidia|governed|recommend|insufficient|SHMEM|^$|gpus all|memlock"
chmod -R 777 "$FT" 2>/dev/null || true
echo ">> run dir: p6_unisharp/ft/runs/a_${SC}"
ls -la "$FT/runs/a_${SC}" 2>/dev/null | head