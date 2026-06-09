#!/bin/bash
# One-click direct-pano: .insv -> stitch+crop+VGGT -> per-pano poses ->
# direct equirect training (PSNR/SSIM/LPIPS) -> KSPLAT. Emits timings.json.
#
# Usage: run_pano_e2e.sh <scene_name> <insv_filename> [n_panos=90] [iters=7000] [width=1024]
set -u
S="$1"; INSV="$2"; NP="${3:-90}"; ITERS="${4:-7000}"; WID="${5:-1024}"
ROOT=/raid/git/gaussian-splatting; PP=$ROOT/p3_pano; PT=nvcr.io/nvidia/pytorch:24.12-py3
D=$ROOT/data/8kpano/scenes/${S}_pano
t() { date +%s; }
declare -A T

echo "=== run_pano_e2e: $S ($INSV) np=$NP iters=$ITERS width=$WID ==="
t0=$(t)
if [ -f "$D/sparse/0/cameras.bin" ]; then
  echo "[prep] reusing existing $D"
else
  bash $PP/prep_pano.sh "$S" "$INSV" "$NP" || { echo "PREP FAILED"; exit 1; }
fi
T[prep]=$(( $(t) - t0 ))

t0=$(t)
docker run --rm --user 0:0 -v $ROOT:/workspace/gaussian-splatting $PT bash -c "
  pip install -q --no-deps opencv-python plyfile 2>/dev/null
  cd /workspace/gaussian-splatting
  python p3_pano/make_pano_dataset.py data/8kpano/scenes/${S}_pano p3_pano/pano_cams_${S}.json" \
  2>&1 | grep -E "OK|Error" | tail -2
sed -i 's#sparse/0/points3D.ply#sparse/0/points.ply#' $PP/pano_cams_${S}.json 2>/dev/null
[ -f "$PP/pano_cams_${S}.json" ] || { echo "POSE FAILED"; exit 1; }
T[pose]=$(( $(t) - t0 ))

t0=$(t)
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --user 0:0 \
  -e TORCH_HOME=/wcache -v $ROOT:/workspace/gaussian-splatting -v $ROOT/p2_vggt/weights:/wcache $PT bash -c "
    set -e; mkdir -p /wcache/hub/checkpoints; cd /workspace/gaussian-splatting
    pip install -q --no-deps plyfile opencv-python joblib \
      submodules/diff-gaussian-rasterization submodules/simple-knn submodules/fused-ssim >/dev/null 2>&1
    pip install -q --no-deps -e p3_pano/diff_gaussian_rasterization_pano >/dev/null 2>&1
    python p3_pano/train_pano.py p3_pano/pano_cams_${S}.json \
      data/8kpano/scenes/${S}_pano/output_pano $ITERS $WID 2>&1 | tail -20" \
  2>&1 | grep -E "iter |EVAL|DONE|Error" | tail -8
T[train]=$(( $(t) - t0 ))

t0=$(t)
PLY=$D/output_pano/point_cloud/iteration_${ITERS}/point_cloud.ply
[ -f "$PLY" ] && bash $ROOT/pano_pipeline/ply_to_ksplat.sh "$PLY" "${PLY%.ply}.ksplat" 1 2>&1 | tail -1
T[ksplat]=$(( $(t) - t0 ))

python3 - "$D/output_pano/timings.json" "${T[prep]}" "${T[pose]}" "${T[train]}" "${T[ksplat]}" <<'PY'
import json,sys
p,*v=sys.argv[1:]; k=["prep","pose","train","ksplat"]
d={k[i]:int(v[i]) for i in range(4)}; d["total"]=sum(d.values())
json.dump(d,open(p,"w"),indent=1); print("timings(s):",d)
PY
echo "=== DONE $S -> $D/output_pano (view: ?scene=$S&source=pano) ==="
