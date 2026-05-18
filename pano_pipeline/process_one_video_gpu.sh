#!/bin/bash
# Process one .insv video end-to-end through 3DGS pipeline (GPU-accelerated COLMAP)
#
# Usage: process_one_video_gpu.sh <insv_path> <scene_name> [target_panos]

set -e

INSV="$1"
SCENE_NAME="$2"
TARGET_PANOS="${3:-80}"

if [ -z "$INSV" ] || [ -z "$SCENE_NAME" ]; then
    echo "Usage: $0 <insv_path> <scene_name> [target_panos=80]"
    exit 1
fi

ROOT=/raid/git/gaussian-splatting
SCENE_DIR=$ROOT/data/8kpano/scenes/$SCENE_NAME
PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:24.12-py3
FFMPEG_IMAGE=linuxserver/ffmpeg
COLMAP_IMAGE=colmap/colmap:latest

mkdir -p $SCENE_DIR/{panoramas,input}
chmod 777 $SCENE_DIR $SCENE_DIR/panoramas $SCENE_DIR/input

DURATION_SEC=$(docker run --rm --user 0:0 -v $ROOT/data/8kpano:/data $FFMPEG_IMAGE \
    -i /data/$(basename $INSV) 2>&1 | grep "Duration" | awk '{print $2}' | tr -d ',' | \
    awk -F: '{print ($1*3600 + $2*60 + $3)}')
FPS=$(awk -v t=$TARGET_PANOS -v d=$DURATION_SEC 'BEGIN{ printf "%.4f", t/d }')

echo "=================================================="
echo "Scene: $SCENE_NAME"
echo "Source: $INSV  ($(du -h $INSV | cut -f1))"
echo "Duration: ${DURATION_SEC}s, fps=$FPS for $TARGET_PANOS panos"
echo "=================================================="

# === Stage 1: Extract equirect panoramas ===
echo "[Stage 1] Extracting panoramas from .insv ..."
docker run --rm --gpus all --user 0:0 \
    -v $ROOT/data/8kpano:/data \
    $FFMPEG_IMAGE \
    -hwaccel cuda \
    -i /data/$(basename $INSV) \
    -filter_complex "\
[0:0]v360=input=fisheye:output=equirect:ih_fov=200:iv_fov=200:pitch=90[a]; \
[0:1]v360=input=fisheye:output=equirect:ih_fov=200:iv_fov=200:pitch=-90[b]; \
[a][b]blend=all_mode=average,fps=$FPS,scale=8192:4096" \
    -q:v 2 \
    /data/scenes/$SCENE_NAME/panoramas/pano_%04d.jpg 2>&1 | tail -2

PANO_COUNT=$(ls $SCENE_DIR/panoramas/*.jpg 2>/dev/null | wc -l)
echo "  → $PANO_COUNT panoramas extracted"

# === Stage 2: Split panoramas to perspective views ===
echo "[Stage 2] Splitting to perspective views..."
docker run --rm --user 0:0 \
    -v $ROOT:/workspace/gaussian-splatting \
    $PYTORCH_IMAGE \
    bash -c "pip install -q --no-deps opencv-python 2>/dev/null && \
        cd /workspace/gaussian-splatting && \
        python pano_pipeline/pano_to_perspective.py \
            -i /workspace/gaussian-splatting/data/8kpano/scenes/$SCENE_NAME/panoramas \
            -o /workspace/gaussian-splatting/data/8kpano/scenes/$SCENE_NAME/input \
            --fov 120 --size 1280 --preset standard --quality 90" 2>&1 | tail -3

PERSP_COUNT=$(ls $SCENE_DIR/input/*.jpg 2>/dev/null | wc -l)
echo "  → $PERSP_COUNT perspective views"

# === Stage 3: COLMAP with GPU SIFT ===
echo "[Stage 3] COLMAP SfM (GPU SIFT + GPU matcher)..."
docker rm -f gs-colmap-$SCENE_NAME 2>/dev/null || true

docker run --rm --gpus all --user 0:0 \
    -v $ROOT/data/8kpano/scenes/$SCENE_NAME:/scene \
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

        echo '>> Feature matching (GPU)...'
        colmap exhaustive_matcher \
            --database_path /scene/distorted/database.db \
            --FeatureMatching.use_gpu 1 2>&1 | tail -15

        echo '>> Sparse reconstruction (BA)...'
        colmap mapper \
            --database_path /scene/distorted/database.db \
            --image_path /scene/input \
            --output_path /scene/distorted/sparse \
            --Mapper.ba_global_function_tolerance 0.000001 2>&1 | tail -25

        echo '>> Image undistortion...'
        colmap image_undistorter \
            --image_path /scene/input \
            --input_path /scene/distorted/sparse/0 \
            --output_path /scene \
            --output_type COLMAP 2>&1 | tail -5

        # Reorganize sparse/0
        mkdir -p /scene/sparse/0
        for f in /scene/sparse/*; do
            [ \"\$(basename \$f)\" = '0' ] && continue
            [ -e \$f ] && mv \$f /scene/sparse/0/ 2>/dev/null || true
        done
        echo '>> COLMAP DONE'
    " 2>&1 | grep -E ">> |Processed|Features|reconstruction|registered|images|Mean|3D|Loading|Error|ERROR|FAILED|fail|Unknown|Elapsed|points" | tail -50

NUM_IMG=$(ls $SCENE_DIR/images/*.jpg 2>/dev/null | wc -l)
echo "  → COLMAP reconstructed $NUM_IMG images"

if [ "$NUM_IMG" -lt 20 ]; then
    echo "[$SCENE_NAME] FAILED: Too few images reconstructed ($NUM_IMG < 20)"
    rm -rf $SCENE_DIR/panoramas $SCENE_DIR/distorted $SCENE_DIR/stereo $SCENE_DIR/run-colmap-*.sh
    exit 0
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

# === Cleanup ===
rm -rf $SCENE_DIR/panoramas $SCENE_DIR/distorted $SCENE_DIR/stereo $SCENE_DIR/run-colmap-*.sh

PLY=$SCENE_DIR/output/point_cloud/iteration_30000/point_cloud.ply
if [ -f "$PLY" ]; then
    echo "[SUCCESS] $SCENE_NAME: $PLY ($(du -h $PLY | cut -f1)), output total $(du -sh $SCENE_DIR/output | cut -f1)"
else
    echo "[FAILED] $SCENE_NAME: no final ply produced"
fi
