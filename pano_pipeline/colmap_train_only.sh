#!/bin/bash
# Run COLMAP (GPU) + 3DGS training for a scene that already has input/ images
# Usage: colmap_train_only.sh <scene_name>

set -e

SCENE_NAME="$1"
[ -z "$SCENE_NAME" ] && { echo "Usage: $0 <scene_name>"; exit 1; }

ROOT=/raid/git/gaussian-splatting
SCENE_DIR=$ROOT/data/8kpano/scenes/$SCENE_NAME
PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:24.12-py3
COLMAP_IMAGE=colmap/colmap:latest

[ -d "$SCENE_DIR/input" ] || { echo "ERROR: $SCENE_DIR/input does not exist"; exit 1; }

# Clean stale COLMAP/training artifacts
rm -rf $SCENE_DIR/{distorted,sparse,images,stereo,output,run-colmap-*.sh}

PERSP_COUNT=$(ls $SCENE_DIR/input/*.jpg 2>/dev/null | wc -l)
echo "=================================================="
echo "Scene: $SCENE_NAME ($PERSP_COUNT perspective views)"
echo "=================================================="

# === Stage 3: COLMAP with GPU SIFT ===
echo "[Stage 3] COLMAP SfM (GPU SIFT + GPU matcher)..."
docker rm -f gs-colmap-$SCENE_NAME 2>/dev/null || true

docker run --rm --gpus all --user 0:0 \
    -v $SCENE_DIR:/scene \
    --name gs-colmap-$SCENE_NAME \
    $COLMAP_IMAGE \
    bash -c "
        set -e
        mkdir -p /scene/distorted/sparse

        echo '>> Feature extraction (GPU)...'
        colmap feature_extractor \
            --database_path /scene/distorted/database.db \
            --image_path /scene/input \
            --ImageReader.single_camera 1 \
            --ImageReader.camera_model PINHOLE \
            --FeatureExtraction.use_gpu 1 2>&1 | tail -15

        echo '>> Feature matching (GPU, sequential window=56 + quadratic)...'
        colmap sequential_matcher \
            --database_path /scene/distorted/database.db \
            --SequentialMatching.overlap 56 \
            --SequentialMatching.quadratic_overlap 1 \
            --FeatureMatching.use_gpu 1 2>&1 | tail -15

        echo '>> Sparse reconstruction (BA)...'
        colmap mapper \
            --database_path /scene/distorted/database.db \
            --image_path /scene/input \
            --output_path /scene/distorted/sparse \
            --Mapper.ba_global_function_tolerance 0.000001 \
            --Mapper.init_min_tri_angle 4.0 \
            --Mapper.init_min_num_inliers 30 \
            --Mapper.abs_pose_min_num_inliers 15 \
            --Mapper.min_num_matches 15 2>&1 | tail -25

        echo '>> Image undistortion...'
        colmap image_undistorter \
            --image_path /scene/input \
            --input_path /scene/distorted/sparse/0 \
            --output_path /scene \
            --output_type COLMAP 2>&1 | tail -5

        mkdir -p /scene/sparse/0
        for f in /scene/sparse/*; do
            [ \"\$(basename \$f)\" = '0' ] && continue
            [ -e \$f ] && mv \$f /scene/sparse/0/ 2>/dev/null || true
        done
        echo '>> COLMAP DONE'
    " 2>&1 | grep -E ">> |Processed|Features|reconstruction|registered|images|Mean|3D|Loading|Error|ERROR|FAILED|fail|Unknown|Elapsed|points"

NUM_IMG=$(ls $SCENE_DIR/images/*.jpg 2>/dev/null | wc -l)
echo "  → COLMAP reconstructed $NUM_IMG images"

if [ "$NUM_IMG" -lt 20 ]; then
    echo "[$SCENE_NAME] FAILED: Too few images reconstructed ($NUM_IMG < 20)"
    exit 1
fi

# === Stage 4: 3DGS training ===
echo "[Stage 4] Training 3DGS..."
docker rm -f gs-train-$SCENE_NAME 2>/dev/null || true

docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    --user 0:0 \
    -v $ROOT:/workspace/gaussian-splatting \
    --name gs-train-$SCENE_NAME \
    $PYTORCH_IMAGE \
    bash -c "
        set -e
        cd /workspace/gaussian-splatting
        pip install -q --no-deps plyfile opencv-python joblib \
            submodules/diff-gaussian-rasterization \
            submodules/simple-knn \
            submodules/fused-ssim > /dev/null 2>&1

        SCENE=/workspace/gaussian-splatting/data/8kpano/scenes/$SCENE_NAME
        python train.py -s \$SCENE -m \$SCENE/output --iterations 30000 2>&1 | \
            grep -E 'ITER|PSNR|Saving|Training complete|Number of|Output folder' | tail -10
    " 2>&1 | tail -15

# Cleanup intermediate
rm -rf $SCENE_DIR/{distorted,stereo,run-colmap-*.sh}

PLY=$SCENE_DIR/output/point_cloud/iteration_30000/point_cloud.ply
if [ -f "$PLY" ]; then
    echo "[SUCCESS] $SCENE_NAME: $PLY ($(du -h $PLY | cut -f1)), output total $(du -sh $SCENE_DIR/output | cut -f1)"
else
    echo "[FAILED] $SCENE_NAME: no final ply produced"
fi
