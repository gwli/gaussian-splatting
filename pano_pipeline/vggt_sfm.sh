#!/bin/bash
# P2.1: VGGT feed-forward SfM as a drop-in replacement for COLMAP Stage 3.
#
# VGGT (Visual Geometry Grounded Transformer, Meta/Oxford) predicts camera
# poses + depth + a 3D point cloud in a SINGLE forward pass — seconds instead
# of COLMAP's minutes/hours. Output is written in COLMAP sparse format so the
# existing 3DGS train.py consumes it unchanged.
#
# Caveats:
#  * VGGT attends across ALL frames at once → memory ~O(N^2). We subsample
#    to --max-frames (default 150) to fit 80GB. More frames need chunking.
#  * Feed-forward mode (default) writes PINHOLE cameras + up to 100k points,
#    NO feature tracks. Use --ba for LightGlue+pycolmap bundle adjustment
#    (slower, more accurate, needs extra deps).
#
# Usage: vggt_sfm.sh <scene_name> [max_frames=150] [iterations=15000]

set -e

SCENE_NAME="$1"
MAX_FRAMES="${2:-150}"
ITERATIONS="${3:-15000}"
CONF_THRES="${4:-1.5}"   # VGGT depth-confidence threshold; 5.0 (demo default) filters all
                          # points on low-texture drone footage. 1.5 keeps a dense cloud.
[ -z "$SCENE_NAME" ] && { echo "Usage: $0 <scene_name> [max_frames=150] [iterations=15000] [conf_thres=1.5]"; exit 1; }

ROOT=/raid/git/gaussian-splatting
SCENE_DIR=$ROOT/data/8kpano/scenes/$SCENE_NAME
SRC_INPUT=$SCENE_DIR/input
VGGT_DIR=$ROOT/p2_vggt/vggt
WEIGHTS=$ROOT/p2_vggt/weights/model.pt
WORK=$SCENE_DIR/vggt          # VGGT-based reconstruction lives here, separate from COLMAP output/
PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:24.12-py3

[ -d "$SRC_INPUT" ] || { echo "ERROR: $SRC_INPUT does not exist"; exit 1; }
[ -f "$WEIGHTS" ]   || { echo "ERROR: weights not found at $WEIGHTS (see p2_vggt/README.md)"; exit 1; }
[ -d "$VGGT_DIR" ]  || { echo "ERROR: VGGT repo not at $VGGT_DIR (see p2_vggt/README.md)"; exit 1; }

# Idempotent patch: make the lightglue/pyceres import lazy so the feed-forward
# path needs no BA deps. (Re-applying is a no-op once the marker line exists.)
DC=$VGGT_DIR/demo_colmap.py
if grep -q "^from vggt.dependency.track_predict import predict_tracks" "$DC"; then
    sed -i 's|^from vggt.dependency.track_predict import predict_tracks|# (moved lazy into --use_ba branch by vggt_sfm.sh)|' "$DC"
    sed -i 's|^    if args.use_ba:|    if args.use_ba:\n        from vggt.dependency.track_predict import predict_tracks|' "$DC"
    echo "Patched demo_colmap.py for lazy BA import"
fi

rm -rf "$WORK"
mkdir -p "$WORK/images"

# --- Subsample input/ down to MAX_FRAMES, evenly spaced ---
TOTAL=$(ls "$SRC_INPUT"/*.jpg 2>/dev/null | wc -l)
if [ "$TOTAL" -le "$MAX_FRAMES" ]; then
    STRIDE=1
else
    STRIDE=$(( (TOTAL + MAX_FRAMES - 1) / MAX_FRAMES ))
fi
echo "=================================================="
echo "Scene: $SCENE_NAME"
echo "VGGT SfM: $TOTAL input views → every ${STRIDE}th (≈$((TOTAL/STRIDE))), max $MAX_FRAMES"
echo "=================================================="
i=0
for f in $(ls "$SRC_INPUT"/*.jpg | sort); do
    if [ $((i % STRIDE)) -eq 0 ]; then
        cp "$f" "$WORK/images/"
    fi
    i=$((i+1))
done
NSEL=$(ls "$WORK/images"/*.jpg | wc -l)
echo "Selected $NSEL frames"
chmod -R 777 "$WORK"

# --- Stage 3 (VGGT): pose + depth + point cloud in one forward pass ---
echo "[Stage 3-VGGT] Running VGGT feed-forward SfM..."
docker rm -f gs-vggt-$SCENE_NAME 2>/dev/null || true

START_T3=$(date +%s)
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    --user 0:0 \
    -e TORCH_HOME=/wcache \
    -v $ROOT:/workspace/gaussian-splatting \
    -v $ROOT/p2_vggt/weights:/wcache_src:ro \
    --name gs-vggt-$SCENE_NAME \
    $PYTORCH_IMAGE \
    bash -c "
        set -e
        # torch.hub caches url downloads at \$TORCH_HOME/hub/checkpoints/<basename>.
        # Pre-seed it with our local model.pt so VGGT skips the HF download.
        mkdir -p /wcache/hub/checkpoints
        ln -sf /wcache_src/model.pt /wcache/hub/checkpoints/model.pt

        # pycolmap pinned to 3.10.0 — VGGT's np_to_pycolmap.py uses the 3.10 Image API
        pip install -q --no-deps einops safetensors trimesh huggingface_hub 'pycolmap==3.10.0' 2>&1 | tail -2

        cd /workspace/gaussian-splatting/p2_vggt/vggt
        export PYTHONPATH=/workspace/gaussian-splatting/p2_vggt/vggt:\$PYTHONPATH
        T0=\$(date +%s)
        python demo_colmap.py \
            --scene_dir /workspace/gaussian-splatting/data/8kpano/scenes/$SCENE_NAME/vggt \
            --conf_thres_value $CONF_THRES 2>&1 | tail -25
        echo \"     VGGT elapsed \$((\$(date +%s)-T0))s\"
    " 2>&1 | grep -vE "DEPRECATION|notice|already satisfied"
END_T3=$(date +%s)
echo "  Stage 3-VGGT total: $((END_T3 - START_T3))s"

# --- Restructure: VGGT writes sparse/ ; 3DGS wants sparse/0/ ---
if [ ! -f "$WORK/sparse/cameras.bin" ] && [ ! -f "$WORK/sparse/0/cameras.bin" ]; then
    echo "[FAILED] VGGT produced no sparse model"
    exit 1
fi
if [ -f "$WORK/sparse/cameras.bin" ]; then
    mkdir -p "$WORK/sparse/0"
    mv "$WORK/sparse"/*.bin "$WORK/sparse/0/" 2>/dev/null || true
    [ -f "$WORK/sparse/points.ply" ] && mv "$WORK/sparse/points.ply" "$WORK/sparse/0/" 2>/dev/null || true
fi
echo "  Sparse model: $(ls $WORK/sparse/0/*.bin 2>/dev/null | wc -l) .bin files"

# --- Stage 4: 3DGS training on VGGT reconstruction ---
echo "[Stage 4] 3DGS training (${ITERATIONS} iter)..."
docker rm -f gs-vggt-train-$SCENE_NAME 2>/dev/null || true

START_T4=$(date +%s)
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    --user 0:0 \
    -v $ROOT:/workspace/gaussian-splatting \
    --name gs-vggt-train-$SCENE_NAME \
    $PYTORCH_IMAGE \
    bash -c "
        set -e
        cd /workspace/gaussian-splatting
        pip install -q --no-deps plyfile opencv-python joblib \
            submodules/diff-gaussian-rasterization \
            submodules/simple-knn \
            submodules/fused-ssim > /dev/null 2>&1
        S=/workspace/gaussian-splatting/data/8kpano/scenes/$SCENE_NAME/vggt
        # --eval holds out every 8th camera as a test set for honest metrics
        python train.py -s \$S -m \$S/output --eval \
            --iterations ${ITERATIONS} --save_iterations ${ITERATIONS} 2>&1 | \
            grep -E 'ITER|PSNR|Saving|Training complete|Number of|Output' | tail -8
    " 2>&1 | tail -12
END_T4=$(date +%s)
echo "  Stage 4 total: $((END_T4 - START_T4))s"

# --- Stage 4b: render held-out test views + compute PSNR/SSIM/LPIPS ---
echo "[Stage 4b] Eval on held-out test views (render + metrics)..."
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    --user 0:0 \
    -e TORCH_HOME=/wcache \
    -v $ROOT:/workspace/gaussian-splatting \
    -v $ROOT/p2_vggt/weights:/wcache_src:ro \
    $PYTORCH_IMAGE \
    bash -c "
        set -e
        # seed lpips VGG weights cache dir (downloaded from download.pytorch.org if absent)
        mkdir -p /wcache/hub/checkpoints
        cd /workspace/gaussian-splatting
        pip install -q --no-deps plyfile opencv-python joblib \
            submodules/diff-gaussian-rasterization submodules/simple-knn submodules/fused-ssim > /dev/null 2>&1
        S=/workspace/gaussian-splatting/data/8kpano/scenes/$SCENE_NAME/vggt
        python render.py -m \$S --skip_train 2>&1 | grep -E 'Rendering|Found|test' | tail -5
        python metrics.py -m \$S 2>&1 | grep -E 'SSIM|PSNR|LPIPS|Scene' | tail -8
    " 2>&1 | grep -vE "DEPRECATION|notice|already satisfied" | tail -12

PLY=$WORK/output/point_cloud/iteration_${ITERATIONS}/point_cloud.ply
if [ -f "$PLY" ]; then
    echo "[Stage 5] PLY → KSPLAT..."
    bash $ROOT/pano_pipeline/ply_to_ksplat.sh "$PLY" "${PLY%.ply}.ksplat" 1 2>&1 | tail -1
    echo "[SUCCESS] $SCENE_NAME (VGGT): $PLY ($(du -h $PLY | cut -f1))"
    echo "  Stage3-VGGT: $((END_T3-START_T3))s | Stage4: $((END_T4-START_T4))s | frames: $NSEL"
else
    echo "[FAILED] $SCENE_NAME: no final ply"
    exit 1
fi
