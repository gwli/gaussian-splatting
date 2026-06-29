#!/bin/bash
# A-tier dataset builder: de-rotate -> pseudo-depth -> pairs, into SimPanorama layout.
#   bash p6_unisharp/ft/build_ft_dataset.sh 027 023
# Env: POS_SCALE (VGGT->metre, default 1.0; tune so adjacent steps ~0.1-1m, see task_ft.md)
#      FAR_INVALID_M (default 150)  MIN_TR/MAX_TR (default 0.3/8.0)
set -u
ROOT=/raid/git/gaussian-splatting
FT=$ROOT/p6_unisharp/ft
PT=nvcr.io/nvidia/pytorch:24.12-py3
VENV=/w/p6_unisharp/venv
POS_SCALE="${POS_SCALE:-1.0}"; FAR_INVALID_M="${FAR_INVALID_M:-300}"  # 300: aerial OOD rails high, see task_ft.md §5.1
MIN_TR="${MIN_TR:-0.3}"; MAX_TR="${MAX_TR:-8.0}"
SCENES="${*:-027}"

mkdir -p "$FT/data" "$FT/poses" "$FT/pairs"; chmod -R 777 "$FT" 2>/dev/null || true

DRUN(){ docker run --rm --gpus all --ipc=host --user 0:0 \
  -e HF_HOME=/w/p6_unisharp/.hf -e TORCH_HOME=/w/p6_unisharp/.torch \
  -e TORCH_CUDA_ARCH_LIST=9.0 -v "$ROOT":/w -w /w "$PT" bash -c "$1" 2>&1 \
  | grep -avE "Copyright|NVIDIA Release|reserved|This container|By pulling|^==|PyTorch Version|^$|docs.nvidia|governed|recommend|insufficient"; }

for S in $SCENES; do
  SC=scene_${S}hf
  CAMS=$ROOT/p3_pano/pano_cams_${SC}.json
  PANO=$ROOT/data/8kpano/scenes/${SC}_pano/panoramas
  [ -f "$CAMS" ] || { echo "[$S] no cams json: $CAMS — skip"; continue; }
  [ -d "$PANO" ] || { echo "[$S] no panoramas: $PANO — skip"; continue; }
  echo "######################## $SC (pos_scale=$POS_SCALE) ########################"

  echo ">> [1/3] de-rotate + pose csv"
  GL=""; [ "${GRAVITY:-0}" = "1" ] && GL="--gravity-level"
  DRUN "python /w/p6_unisharp/ft/derotate_and_pose.py \
    --cams /w/p3_pano/pano_cams_${SC}.json \
    --panodir /w/data/8kpano/scenes/${SC}_pano/panoramas \
    --out-rgb /w/p6_unisharp/ft/data/${SC} \
    --out-pose /w/p6_unisharp/ft/poses/${SC}.csv \
    --pos-scale $POS_SCALE $GL"

  echo ">> [2/3] pseudo depth (UniK3D vitl, needs venv from run_unisharp.sh)"
  DRUN "if [ ! -f $VENV/.ready ]; then echo '[ERR] venv missing — run p6_unisharp/run_unisharp.sh once first'; exit 3; fi
    . $VENV/bin/activate
    export PYTHONPATH=/w/p6_unisharp/UniSHARP:/w/p6_unisharp/UniSHARP/UniK3D
    python /w/p6_unisharp/ft/pseudo_depth.py \
      --rgb-dir /w/p6_unisharp/ft/data/${SC} --far-invalid-m $FAR_INVALID_M"

  echo ">> [3/3] pairs"
  python3 "$FT/make_pairs.py" --pose "$FT/poses/${SC}.csv" --scene "$SC" \
    --out "$FT/pairs/${SC}.jsonl" --min-tr "$MIN_TR" --max-tr "$MAX_TR"

  chmod -R 777 "$FT" 2>/dev/null || true
  echo ">> $SC summary:"
  echo "   rgb:   $(ls $FT/data/${SC}/*.jpg 2>/dev/null | wc -l)"
  echo "   depth: $(ls $FT/data/${SC}/depth/*.npy 2>/dev/null | wc -l)"
  echo "   pairs: $(wc -l < $FT/pairs/${SC}.jsonl 2>/dev/null || echo 0)"
done
echo "######################## DONE ########################"
