#!/bin/bash
# Regenerate a scene's input/ perspective views from its source .insv video.
# Used for scenes whose input/ was deleted (e.g. the COLMAP-failed 026/027/028)
# so they can be re-attempted with VGGT.
#
# Usage: regen_input.sh <scene_name> <insv_filename> [target_panos=100]

set -e
SCENE_NAME="$1"
INSV_NAME="$2"
TARGET_PANOS="${3:-100}"
[ -z "$INSV_NAME" ] && { echo "Usage: $0 <scene_name> <insv_filename> [target_panos=100]"; exit 1; }

ROOT=/raid/git/gaussian-splatting
SCENE_DIR=$ROOT/data/8kpano/scenes/$SCENE_NAME
INSV=$ROOT/data/8kpano/$INSV_NAME
FFMPEG_IMAGE=linuxserver/ffmpeg
PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:24.12-py3
[ -f "$INSV" ] || { echo "ERROR: $INSV not found"; exit 1; }

mkdir -p $SCENE_DIR/panoramas $SCENE_DIR/input
chmod -R 777 $SCENE_DIR

DUR=$(docker run --rm --user 0:0 -v $ROOT/data/8kpano:/d $FFMPEG_IMAGE \
    -i /d/$INSV_NAME 2>&1 | grep Duration | awk '{print $2}' | tr -d ',' | \
    awk -F: '{print ($1*3600+$2*60+$3)}')
FPS=$(awk -v t=$TARGET_PANOS -v d=$DUR 'BEGIN{printf "%.4f", t/d}')
echo "[regen $SCENE_NAME] ${DUR}s, fps=$FPS for $TARGET_PANOS panos"

echo "[1/2] stitch equirect panoramas..."
docker run --rm --gpus all --user 0:0 -v $ROOT/data/8kpano:/data $FFMPEG_IMAGE \
    -hwaccel cuda -i /data/$INSV_NAME \
    -filter_complex "[0:0]v360=input=fisheye:output=equirect:ih_fov=200:iv_fov=200:pitch=90[a];[0:1]v360=input=fisheye:output=equirect:ih_fov=200:iv_fov=200:pitch=-90[b];[a][b]blend=all_mode=average,fps=$FPS,scale=8192:4096" \
    -q:v 2 /data/scenes/$SCENE_NAME/panoramas/pano_%04d.jpg 2>&1 | tail -1
echo "  → $(ls $SCENE_DIR/panoramas/*.jpg | wc -l) panoramas"

echo "[2/2] split to perspective views (parallel)..."
docker run --rm --user 0:0 -v $ROOT:/w $PYTORCH_IMAGE bash -c "
    pip install -q --no-deps opencv-python 2>/dev/null
    cd /w && python pano_pipeline/pano_to_perspective.py \
        -i /w/data/8kpano/scenes/$SCENE_NAME/panoramas \
        -o /w/data/8kpano/scenes/$SCENE_NAME/input \
        --fov 120 --size 1280 --preset standard --quality 90" 2>&1 | tail -2
echo "  → $(ls $SCENE_DIR/input/*.jpg | wc -l) perspective views"
rm -rf $SCENE_DIR/panoramas
