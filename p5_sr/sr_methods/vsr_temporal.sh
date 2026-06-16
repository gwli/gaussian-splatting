#!/bin/bash
# task_sr M4: temporal SISR-vs-VSR comparison on a consecutive real clip.
#   vsr_temporal.sh [SRC_MP4] [N_FRAMES] [CROP] [SS]
# Extracts N consecutive native crops, then temporal_eval.py (x4 downscale-restore,
# rrdbnet-x4 per-frame vs RealBasicVSR-x4, + temporal warp-error).
set -e
ROOT=/raid/git/gaussian-splatting
FF=linuxserver/ffmpeg; PT=nvcr.io/nvidia/pytorch:24.12-py3
SRC="${1:-p5_sr/out/scene_027_8k_highlight.mp4}"
N="${2:-16}"; CROP="${3:-1024}"; SS="${4:-10}"
WK="$ROOT/p5_sr/_work/vsrtmp"; rm -rf "$WK"; mkdir -p "$WK/gt"; chmod -R 777 "$ROOT/p5_sr/_work"

echo ">> [1/3] extract $N consecutive ${CROP}px crops from t=$SS"
docker run --rm --user 0:0 -v "$ROOT":/w $FF -y -ss "$SS" -i "/w/${SRC#$ROOT/}" \
  -frames:v "$N" -vf "crop=$CROP:$CROP:(iw-$CROP)/2:(ih-$CROP)/2" \
  "/w/p5_sr/_work/vsrtmp/gt/f_%03d.png" >/dev/null 2>&1
echo "   $(ls "$WK/gt" | wc -l) frames"

echo ">> [2/3] fetch x4 weights"
WDIR="$ROOT/p5_sr/weights"; chmod 777 "$WDIR"
RRDB="$WDIR/RealESRGAN_x4plus.pth"
[ -f "$RRDB" ] || curl -fsSL "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" -o "$RRDB"
VSR="$WDIR/RealBasicVSR_x4.pth"
[ -f "$VSR" ] || curl -fsSL "https://download.openmmlab.com/mmediting/restorers/real_basicvsr/realbasicvsr_c64b20_1x30x8_lr5e-5_150k_reds_20211104-52f77c2c.pth" -o "$VSR"
ls -la "$RRDB" "$VSR"

echo ">> [3/3] temporal eval (x4)"
docker run --rm --gpus all --ipc=host --user 0:0 -v "$ROOT":/w -w /w $PT bash -c "
  pip install -q --no-deps opencv-python-headless piq 2>&1 | tail -1
  python /w/p5_sr/sr_methods/temporal_eval.py /w/p5_sr/_work/vsrtmp/gt \
    /w/p5_sr/_work/vsrtmp/results.json --rrdb /w/${RRDB#$ROOT/} --vsr /w/${VSR#$ROOT/} --fp16
" 2>&1 | grep -aE "temporal\]|psnr=|results|warp|rrdbnet|realbasicvsr|Error|Traceback" | tail -20
echo ">> results -> $WK/results.json"
