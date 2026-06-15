#!/bin/bash
# Direct-pano end-to-end over all scenes: prep (skip if done) -> per-pano poses
# -> direct equirect training with PSNR/SSIM/LPIPS. Writes results.json/scene.
set -u
ROOT=/raid/git/gaussian-splatting
PP=$ROOT/p3_pano
PT=nvcr.io/nvidia/pytorch:24.12-py3
LOG=$ROOT/data/8kpano
ITERS=7000; WID=1024

declare -A INSV=(
  [scene_021]=VID_20260321_171111_021.insv
  [scene_022]=VID_20260321_171737_022.insv
  [scene_023]=VID_20260326_073432_023.insv
  [scene_025]=VID_20260406_115153_025.insv
  [scene_026]=VID_20260417_162702_026.insv
  [scene_027]=VID_20260502_053911_027.insv
  [scene_028]=VID_20260502_164639_028.insv
)

for S in scene_021 scene_022 scene_023 scene_025 scene_026 scene_027 scene_028; do
  D=$ROOT/data/8kpano/scenes/${S}_pano
  echo "########## $(date) :: $S ##########"

  # 1) prep (stitch+crop+curate+VGGT) — skip if sparse already exists
  if [ -f "$D/sparse/0/cameras.bin" ]; then
    echo "[$S] prep already done, skipping stitch/VGGT"
  else
    bash $PP/prep_pano.sh $S ${INSV[$S]} 90 > $LOG/${S}_pano_prep.log 2>&1 \
      || { echo "[$S] PREP FAILED"; continue; }
  fi

  # 2) per-pano poses
  docker run --rm --user 0:0 -v $ROOT:/workspace/gaussian-splatting $PT bash -c "
    pip install -q --no-deps opencv-python plyfile 2>/dev/null
    cd /workspace/gaussian-splatting
    python p3_pano/make_pano_dataset.py data/8kpano/scenes/${S}_pano p3_pano/pano_cams_${S}.json" \
    > $LOG/${S}_pano_pose.log 2>&1
  sed -i 's#sparse/0/points3D.ply#sparse/0/points.ply#' $PP/pano_cams_${S}.json 2>/dev/null
  [ -f "$PP/pano_cams_${S}.json" ] || { echo "[$S] POSE FAILED"; continue; }

  # 3) direct-pano training + eval (PSNR/SSIM/LPIPS).
  # Default backend: fused equirect-gsplat CUDA kernel (T-F8) — ~1.28x faster than
  # LONLAT at parity quality (see OPTIMIZATION_PLAN.md). BACKEND=lonlat for the
  # legacy OmniGS rasterizer.
  BACKEND=${BACKEND:-fused}
  if [ "$BACKEND" = "lonlat" ]; then
    docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --user 0:0 \
      -e TORCH_HOME=/wcache -v $ROOT:/workspace/gaussian-splatting \
      -v $ROOT/p2_vggt/weights:/wcache nvcr.io/nvidia/pytorch:24.12-py3 bash -c "
        set -e
        mkdir -p /wcache/hub/checkpoints
        cd /workspace/gaussian-splatting
        pip install -q --no-deps plyfile opencv-python joblib \
          submodules/diff-gaussian-rasterization submodules/simple-knn submodules/fused-ssim >/dev/null 2>&1
        pip install -q --no-deps -e p3_pano/diff_gaussian_rasterization_pano >/dev/null 2>&1
        python p3_pano/train_pano.py p3_pano/pano_cams_${S}.json \
          data/8kpano/scenes/${S}_pano/output_pano $ITERS $WID 2>&1 | tail -25" \
      > $LOG/${S}_pano_train.log 2>&1
  else
    GSPLAT_EQUIRECT_FUSED=1 bash $PP/run_pano_gsplat_train.sh \
      p3_pano/pano_cams_${S}.json data/8kpano/scenes/${S}_pano/output_pano $ITERS $WID 512 sph \
      > $LOG/${S}_pano_train.log 2>&1
  fi
  echo "[$S] $(grep -E '\[EVAL\]|\[DONE\]|FAILED|Error' $LOG/${S}_pano_train.log | tail -2)"
  # ksplat for viewer
  PLY=$D/output_pano/point_cloud/iteration_${ITERS}/point_cloud.ply
  [ -f "$PLY" ] && bash $ROOT/pano_pipeline/ply_to_ksplat.sh "$PLY" "${PLY%.ply}.ksplat" 1 >/dev/null 2>&1
done

echo "########## DIRECT-PANO BATCH COMPLETE $(date) ##########"
for S in scene_021 scene_022 scene_023 scene_025 scene_026 scene_027 scene_028; do
  J=$ROOT/data/8kpano/scenes/${S}_pano/output_pano/results.json
  echo -n "$S: "; [ -f "$J" ] && cat "$J" | tr -d '\n ' && echo || echo "no results"
done
