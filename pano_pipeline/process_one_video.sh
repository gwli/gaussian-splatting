#!/bin/bash
# Process one .insv video end-to-end through 3DGS pipeline
#
# Usage: process_one_video.sh <insv_path> <scene_name> [target_panos]

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
DOCKER_IMAGE=nvcr.io/nvidia/pytorch:24.12-py3
FFMPEG_IMAGE=linuxserver/ffmpeg

mkdir -p $SCENE_DIR/{panoramas,input}
chmod 777 $SCENE_DIR $SCENE_DIR/panoramas $SCENE_DIR/input

# Compute target fps to hit TARGET_PANOS frames
DURATION_SEC=$(docker run --rm --user 0:0 -v $ROOT/data/8kpano:/data $FFMPEG_IMAGE \
    -i /data/$(basename $INSV) 2>&1 | grep "Duration" | awk '{print $2}' | tr -d ',' | \
    awk -F: '{print ($1*3600 + $2*60 + $3)}')

FPS=$(awk -v t=$TARGET_PANOS -v d=$DURATION_SEC 'BEGIN{ printf "%.4f", t/d }')

echo "=================================================="
echo "Scene: $SCENE_NAME"
echo "Source: $INSV  ($(du -h $INSV | cut -f1))"
echo "Duration: ${DURATION_SEC}s"
echo "Target panos: $TARGET_PANOS, fps: $FPS"
echo "=================================================="

# === Stage 1: Extract equirect panoramas ===
echo ""
echo "[Stage 1] Extracting $TARGET_PANOS panoramas from .insv ..."
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
echo ""
echo "[Stage 2] Splitting panoramas to perspective views (FOV=120)..."
docker run --rm --user 0:0 \
    -v $ROOT:/workspace/gaussian-splatting \
    $DOCKER_IMAGE \
    bash -c "pip install -q --no-deps opencv-python 2>/dev/null && \
        cd /workspace/gaussian-splatting && \
        python pano_pipeline/pano_to_perspective.py \
            -i /workspace/gaussian-splatting/data/8kpano/scenes/$SCENE_NAME/panoramas \
            -o /workspace/gaussian-splatting/data/8kpano/scenes/$SCENE_NAME/input \
            --fov 120 --size 1280 --preset standard --quality 90" 2>&1 | tail -3

PERSP_COUNT=$(ls $SCENE_DIR/input/*.jpg 2>/dev/null | wc -l)
echo "  → $PERSP_COUNT perspective views"

# === Stage 3 + 4: COLMAP + 3DGS Training ===
echo ""
echo "[Stage 3+4] COLMAP SfM + 3DGS Training..."
docker rm -f gs-$SCENE_NAME 2>/dev/null || true

docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    --user 0:0 \
    -e QT_QPA_PLATFORM=offscreen \
    -v $ROOT:/workspace/gaussian-splatting \
    --name gs-$SCENE_NAME \
    $DOCKER_IMAGE \
    bash -c "
        set -e
        apt-get update -qq > /dev/null 2>&1
        apt-get install -y -qq colmap > /dev/null 2>&1
        cd /workspace/gaussian-splatting
        pip install -q --no-deps plyfile opencv-python joblib \
            submodules/diff-gaussian-rasterization \
            submodules/simple-knn \
            submodules/fused-ssim > /dev/null 2>&1
        echo '=== INSTALL_DONE ==='

        SCENE=/workspace/gaussian-splatting/data/8kpano/scenes/$SCENE_NAME
        echo '=== Running COLMAP (CPU SIFT) ==='
        python convert.py -s \$SCENE --camera PINHOLE --no_gpu 2>&1 | grep -E 'extraction|matching|Mapper|Bundle|Reconstruction|undistortion|images|Image|Done|Error|ERROR|FAILED|points|Elapsed' | tail -30
        echo '=== COLMAP_DONE ==='

        # Check if reconstruction has reasonable number of images
        NUM_IMG=\$(ls \$SCENE/images/*.jpg 2>/dev/null | wc -l)
        echo \"COLMAP reconstructed \$NUM_IMG images\"
        if [ \"\$NUM_IMG\" -lt 20 ]; then
            echo 'TOO FEW IMAGES (<20), SKIPPING TRAINING'
            exit 0
        fi

        echo '=== Training 3DGS ==='
        python train.py -s \$SCENE -m \$SCENE/output --iterations 30000 2>&1 | grep -E 'ITER|PSNR|Saving|Training complete|Number of|Output folder' | tail -10
        echo '=== TRAINING_DONE ==='
    " 2>&1 | grep -E 'INSTALL_DONE|COLMAP_DONE|TRAINING_DONE|reconstructed|TOO FEW|PSNR|Saving|Training complete|Number of|points|Elapsed|Output|images|FAILED|Error'

# === Cleanup intermediate ===
echo ""
echo "[Cleanup] Removing intermediate files (keeping output and metadata)..."
rm -rf $SCENE_DIR/panoramas
rm -rf $SCENE_DIR/distorted
rm -rf $SCENE_DIR/stereo
rm -rf $SCENE_DIR/run-colmap-*.sh

if [ -d "$SCENE_DIR/output/point_cloud" ]; then
    PLY_PATH=$(ls $SCENE_DIR/output/point_cloud/iteration_*/point_cloud.ply | tail -1)
    if [ -n "$PLY_PATH" ]; then
        PLY_SIZE=$(du -h $PLY_PATH | cut -f1)
        OUTPUT_SIZE=$(du -sh $SCENE_DIR/output | cut -f1)
        echo "[SUCCESS] $SCENE_NAME: $PLY_PATH ($PLY_SIZE), total output $OUTPUT_SIZE"
    else
        echo "[FAILED] $SCENE_NAME: no point cloud generated"
    fi
else
    echo "[FAILED] $SCENE_NAME: training did not produce output"
fi
