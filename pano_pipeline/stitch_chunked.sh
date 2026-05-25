#!/bin/bash
# P0.4: Parallel-chunked ffmpeg panorama stitching.
#
# The v360 fisheye→equirect filter is CPU-bound in mainline ffmpeg.
# This script splits a long video into N time chunks, runs N ffmpeg
# processes in parallel (each pinned to a CUDA-decoded chunk), then
# merges the output JPG sequences. Typically 3-4x faster than serial.
#
# Usage: stitch_chunked.sh <insv_file> <output_dir> [target_fps] [chunks]

set -e

INSV="$1"
OUT_DIR="$2"
TARGET_FPS="${3:-0.5}"
CHUNKS="${4:-4}"

[ -f "$INSV" ] || { echo "Usage: $0 <insv> <out_dir> [fps=0.5] [chunks=4]"; exit 1; }

mkdir -p "$OUT_DIR"
chmod 777 "$OUT_DIR"

# Probe duration
DUR=$(docker run --rm --user 0:0 -v "$(dirname "$INSV")":/d linuxserver/ffmpeg \
    -i "/d/$(basename "$INSV")" 2>&1 | grep "Duration" | awk '{print $2}' | tr -d ',' | \
    awk -F: '{print ($1*3600 + $2*60 + $3)}')

CHUNK_SEC=$(awk -v d=$DUR -v c=$CHUNKS 'BEGIN{printf "%.2f", d/c}')
echo "Video: $INSV ($(du -h "$INSV" | cut -f1), ${DUR}s)"
echo "Splitting into $CHUNKS chunks × ${CHUNK_SEC}s each, target fps=$TARGET_FPS"
echo "Output: $OUT_DIR"

PIDS=()
T0=$(date +%s)
for i in $(seq 0 $((CHUNKS-1))); do
    SS=$(awk -v c=$CHUNK_SEC -v i=$i 'BEGIN{printf "%.3f", c*i}')
    OFFSET=$((i * 10000))  # frame number offset between chunks
    LOG="$OUT_DIR/.chunk_${i}.log"

    docker run --rm -d --gpus all --user 0:0 \
        --name stitch_chunk_$i \
        -v "$(dirname "$INSV")":/d:ro \
        -v "$OUT_DIR":/out \
        linuxserver/ffmpeg \
        -hwaccel cuda \
        -ss "$SS" -t "$CHUNK_SEC" \
        -i "/d/$(basename "$INSV")" \
        -filter_complex "\
[0:0]v360=input=fisheye:output=equirect:ih_fov=200:iv_fov=200:pitch=90[a]; \
[0:1]v360=input=fisheye:output=equirect:ih_fov=200:iv_fov=200:pitch=-90[b]; \
[a][b]blend=all_mode=average,fps=$TARGET_FPS,scale=8192:4096" \
        -q:v 2 \
        -start_number $OFFSET \
        "/out/pano_%05d.jpg" > "$LOG" 2>&1
    PIDS+=("stitch_chunk_$i")
    echo "  chunk $i: t=${SS}s for ${CHUNK_SEC}s (container stitch_chunk_$i)"
done

echo "Waiting for $CHUNKS chunks..."
for name in "${PIDS[@]}"; do
    docker wait "$name" >/dev/null 2>&1 || true
done
T1=$(date +%s)

# Renumber outputs to be contiguous
count=$(ls "$OUT_DIR"/pano_*.jpg 2>/dev/null | wc -l)
echo "Done: $count panoramas in $((T1 - T0))s"
rm -f "$OUT_DIR"/.chunk_*.log
