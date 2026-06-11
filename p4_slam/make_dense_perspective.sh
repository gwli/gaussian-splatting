#!/bin/bash
# T-F1: extract a DENSE forward-perspective PNG stream from the raw .insv for
# MASt3R-SLAM. The 90-pano dataset samples only ~0.24 fps (1 frame / 4.2 s over
# a 380 s flight) — far too sparse for monocular SLAM to keep lock. Here we
# stitch dual-fisheye -> equirect and re-project the forward (yaw0/pitch0) view
# in ONE ffmpeg decode pass, at a chosen fps over a chosen segment.
#
# Usage: make_dense_perspective.sh <insv> <out_dir> [fps=4] [start=0] [dur=90] [size=512] [hfov=90]
set -e
ROOT=/raid/git/gaussian-splatting
INSV="$1"; OUT="$2"
FPS="${3:-4}"; SS="${4:-0}"; T="${5:-90}"; SZ="${6:-512}"; HFOV="${7:-90}"
FF=linuxserver/ffmpeg
rm -rf "$OUT"; mkdir -p "$OUT"; chmod 777 "$OUT"

echo "[T-F1] $INSV -> $OUT  | fps=$FPS start=${SS}s dur=${T}s out=${SZ}x${SZ} hfov=$HFOV"
# Two chained v360: (1) each fisheye->equirect (vertical lens, pitch=+/-90) +
# average blend; (2) equirect->flat forward perspective. Intermediate equirect
# kept at 2048x1024 (SLAM downsamples to 512 anyway) to keep the remap cheap.
OREL=$(realpath --relative-to="$ROOT" "$OUT")
docker run --rm --gpus all --user 0:0 -v "$ROOT":/w $FF \
  -hwaccel cuda -ss "$SS" -t "$T" -i "/w/data/8kpano/$INSV" \
  -filter_complex "\
[0:0]v360=input=fisheye:output=equirect:ih_fov=200:iv_fov=200:pitch=90[a];\
[0:1]v360=input=fisheye:output=equirect:ih_fov=200:iv_fov=200:pitch=-90[b];\
[a][b]blend=all_mode=average,fps=${FPS},scale=2048:1024,\
v360=input=equirect:output=flat:h_fov=${HFOV}:v_fov=${HFOV}:yaw=0:pitch=0:w=${SZ}:h=${SZ}" \
  -q:v 2 "/w/$OREL/frame_%04d.png" 2>&1 | tail -2

N=$(ls "$OUT"/frame_*.png 2>/dev/null | wc -l)
echo "[T-F1] extracted $N forward-perspective frames"
