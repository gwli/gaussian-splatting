#!/bin/bash
# Convert PLY to KSPLAT (mkkellogg's compact format) for faster WebXR loading.
# KSPLAT is ~3x smaller and ~5x faster to decode than PLY.
#
# Usage: ply_to_ksplat.sh <ply_file> [output.ksplat] [compression=1]
#
# Compression levels:
#   0 - uncompressed (largest, fastest decode)
#   1 - 16-bit quantization (recommended, ~2x smaller)
#   2 - 8-bit quantization (smallest, slight quality loss)

set -e

PLY_FILE="$1"
[ -z "$PLY_FILE" ] && { echo "Usage: $0 <ply_file> [output.ksplat] [compression=1]"; exit 1; }
[ -f "$PLY_FILE" ] || { echo "ERROR: $PLY_FILE not found"; exit 1; }

OUTPUT="${2:-${PLY_FILE%.ply}.ksplat}"
COMPRESSION="${3:-1}"

GS3D_ROOT=/raid/git/gaussian-splatting/webxr_viewer/GaussianSplats3D
[ -d "$GS3D_ROOT/build" ] || { echo "ERROR: GaussianSplats3D not built. See webxr_viewer/README.md"; exit 1; }

PLY_DIR=$(dirname "$(realpath "$PLY_FILE")")
PLY_NAME=$(basename "$PLY_FILE")
OUT_DIR=$(dirname "$(realpath -m "$OUTPUT")")
OUT_NAME=$(basename "$OUTPUT")

mkdir -p "$OUT_DIR"

echo "Converting $PLY_FILE → $OUTPUT (compression=$COMPRESSION)..."

docker run --rm --user 0:0 \
    -v "$GS3D_ROOT":/gs3d \
    -v "$PLY_DIR":/in:ro \
    -v "$OUT_DIR":/out \
    -w /gs3d \
    node:20 \
    node util/create-ksplat.js "/in/$PLY_NAME" "/out/$OUT_NAME" "$COMPRESSION" 2>&1 | tail -5

if [ -f "$OUTPUT" ]; then
    PLY_SIZE=$(stat -c %s "$PLY_FILE")
    KSPLAT_SIZE=$(stat -c %s "$OUTPUT")
    RATIO=$(awk -v p=$PLY_SIZE -v k=$KSPLAT_SIZE 'BEGIN{printf "%.1fx", p/k}')
    echo "[OK] $(du -h "$OUTPUT" | cut -f1) (PLY was $(du -h "$PLY_FILE" | cut -f1), $RATIO smaller)"
else
    echo "[FAILED] No output produced"
    exit 1
fi
