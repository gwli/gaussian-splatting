#!/bin/bash
# Optimized pipeline (P0+P1):
#   - GPU SIFT (colmap)
#   - GPU sequential matcher (colmap)
#   - GLOMAP global mapper (replaces colmap incremental mapper, 5-10x faster)
#   - 3DGS training default 15000 iter (P0.3, half the time, PSNR ~23.5)
#
# Usage: colmap_train_v2.sh <scene_name> [iterations=15000]

set -e

SCENE_NAME="$1"
ITERATIONS="${2:-15000}"
MAPPER="${3:-colmap}"     # colmap (quality, slow) | glomap (fast, lower quality on drone footage)
MATCHER="${4:-exhaustive}" # exhaustive (quality, slow) | sequential (fast, fewer pairs)
[ -z "$SCENE_NAME" ] && { echo "Usage: $0 <scene_name> [iter=15000] [mapper=colmap|glomap] [matcher=exhaustive|sequential]"; exit 1; }

ROOT=/raid/git/gaussian-splatting
SCENE_DIR=$ROOT/data/8kpano/scenes/$SCENE_NAME
PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:24.12-py3
# Mapper=colmap: official image (proven, slower); use for both feat+match+mapper
COLMAP_IMAGE=colmap/colmap:latest
# Mapper=glomap: needs DB-compatible colmap (podral3 has both colmap+glomap, but only podral3
# colmap is used; mapper goes through arhanjain since podral3 glomap SIGILLs)
GLOMAP_COLMAP_IMAGE=podral3/glomap:latest
GLOMAP_IMAGE=arhanjain/glomap:latest

[ -d "$SCENE_DIR/input" ] || { echo "ERROR: $SCENE_DIR/input does not exist"; exit 1; }

rm -rf $SCENE_DIR/{distorted,sparse,images,stereo,output,run-colmap-*.sh}

PERSP_COUNT=$(ls $SCENE_DIR/input/*.jpg 2>/dev/null | wc -l)
echo "=================================================="
echo "Scene: $SCENE_NAME ($PERSP_COUNT perspective views)"
echo "Optimized pipeline: GPU SIFT + GLOMAP + ${ITERATIONS} iter"
echo "=================================================="

# === Stage 3: feature_extractor + sequential_matcher + mapper + undistort ===
if [ "$MAPPER" = "glomap" ]; then
    CM_IMG=$GLOMAP_COLMAP_IMAGE
else
    CM_IMG=$COLMAP_IMAGE
fi
echo "[Stage 3] GPU SIFT + GPU matching + $MAPPER mapper..."
docker rm -f gs-colmap-$SCENE_NAME gs-glomap-$SCENE_NAME gs-colmap-mapper-$SCENE_NAME 2>/dev/null || true

START_T3=$(date +%s)

# 3.1 + 3.2: feature_extractor + sequential_matcher (colmap)
docker run --rm --gpus all --user 0:0 --entrypoint /bin/bash \
    -v $SCENE_DIR:/scene \
    --name gs-colmap-$SCENE_NAME \
    $CM_IMG \
    -c "
        set -e
        mkdir -p /scene/distorted/sparse

        echo '>> [3.1] Feature extraction (GPU SIFT)...'
        T0=\$(date +%s)
        colmap feature_extractor \
            --database_path /scene/distorted/database.db \
            --image_path /scene/input \
            --ImageReader.single_camera 1 \
            --ImageReader.camera_model PINHOLE \
            --FeatureExtraction.use_gpu 1 2>&1 | tail -3
        echo \"     elapsed \$((\$(date +%s)-T0))s\"

        echo '>> [3.2] Feature matching (GPU, $MATCHER)...'
        T0=\$(date +%s)
        if [ \"$MATCHER\" = \"sequential\" ]; then
            colmap sequential_matcher \
                --database_path /scene/distorted/database.db \
                --SequentialMatching.overlap 56 \
                --SequentialMatching.quadratic_overlap 1 \
                --FeatureMatching.use_gpu 1 2>&1 | tail -3
        else
            colmap exhaustive_matcher \
                --database_path /scene/distorted/database.db \
                --FeatureMatching.use_gpu 1 2>&1 | tail -3
        fi
        echo \"     elapsed \$((\$(date +%s)-T0))s\"
    " 2>&1 | grep -E ">> |elapsed|Features|Processed|Error|ERROR"

# 3.3: Mapper - either GLOMAP (fast) or COLMAP incremental (quality)
if [ "$MAPPER" = "glomap" ]; then
    echo "  Mapper: GLOMAP (fast, lower quality on forward-flight drone data)"
    docker run --rm --gpus all --user 0:0 --entrypoint /bin/bash \
        -v $SCENE_DIR:/scene \
        --name gs-glomap-$SCENE_NAME \
        $GLOMAP_IMAGE \
        -c "
            set -e
            echo '>> [3.3] Global SfM (GLOMAP)...'
            T0=\$(date +%s)
            glomap mapper \
                --database_path /scene/distorted/database.db \
                --image_path /scene/input \
                --output_path /scene/distorted/sparse 2>&1 | tail -10
            echo \"     elapsed \$((\$(date +%s)-T0))s\"
        " 2>&1 | grep -E ">> |elapsed|Reconstruction|Registered|registered|points|tracks|Error|ERROR|FAILED|fail"
else
    echo "  Mapper: COLMAP incremental (quality)"
    docker run --rm --gpus all --user 0:0 --entrypoint /bin/bash \
        -v $SCENE_DIR:/scene \
        --name gs-colmap-mapper-$SCENE_NAME \
        $CM_IMG \
        -c "
            set -e
            echo '>> [3.3] Incremental SfM (COLMAP mapper)...'
            T0=\$(date +%s)
            colmap mapper \
                --database_path /scene/distorted/database.db \
                --image_path /scene/input \
                --output_path /scene/distorted/sparse \
                --Mapper.ba_global_function_tolerance 0.000001 2>&1 | tail -15
            echo \"     elapsed \$((\$(date +%s)-T0))s\"
        " 2>&1 | grep -E ">> |elapsed|Registered|points|images|Error|ERROR|FAILED|fail"
fi

# 3.4: image undistortion
docker run --rm --gpus all --user 0:0 --entrypoint /bin/bash \
    -v $SCENE_DIR:/scene \
    $CM_IMG \
    -c "
        set -e
        echo '>> [3.4] Image undistortion...'
        colmap image_undistorter \
            --image_path /scene/input \
            --input_path /scene/distorted/sparse/0 \
            --output_path /scene \
            --output_type COLMAP 2>&1 | tail -3

        mkdir -p /scene/sparse/0
        for f in /scene/sparse/*; do
            [ \"\$(basename \$f)\" = '0' ] && continue
            [ -e \$f ] && mv \$f /scene/sparse/0/ 2>/dev/null || true
        done
        echo '>> COLMAP+GLOMAP DONE'
    " 2>&1 | grep -E ">> |Error|ERROR"

END_T3=$(date +%s)
echo "  Stage 3 total: $((END_T3 - START_T3))s"

NUM_IMG=$(ls $SCENE_DIR/images/*.jpg 2>/dev/null | wc -l)
echo "  → reconstructed $NUM_IMG images"

if [ "$NUM_IMG" -lt 20 ]; then
    echo "[$SCENE_NAME] FAILED: Too few images ($NUM_IMG < 20)"
    exit 1
fi

# === Stage 4: 3DGS training ===
echo "[Stage 4] 3DGS training (${ITERATIONS} iter)..."
docker rm -f gs-train-$SCENE_NAME 2>/dev/null || true

START_T4=$(date +%s)
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
        python train.py -s \$SCENE -m \$SCENE/output \
            --iterations ${ITERATIONS} \
            --save_iterations ${ITERATIONS} 2>&1 | \
            grep -E 'ITER|PSNR|Saving|Training complete|Number of|Output folder' | tail -8
    " 2>&1 | tail -12
END_T4=$(date +%s)
echo "  Stage 4 total: $((END_T4 - START_T4))s"

rm -rf $SCENE_DIR/{distorted,stereo,run-colmap-*.sh}

PLY=$SCENE_DIR/output/point_cloud/iteration_${ITERATIONS}/point_cloud.ply
if [ -f "$PLY" ]; then
    echo "[Stage 5] Convert PLY → KSPLAT for WebXR..."
    KSPLAT=${PLY%.ply}.ksplat
    bash $ROOT/pano_pipeline/ply_to_ksplat.sh "$PLY" "$KSPLAT" 1 2>&1 | tail -1
    echo "[SUCCESS] $SCENE_NAME: $PLY ($(du -h $PLY | cut -f1))"
    echo "  Total Stage 3+4: $((END_T4 - START_T3))s"
else
    echo "[FAILED] $SCENE_NAME: no final ply"
    exit 1
fi
