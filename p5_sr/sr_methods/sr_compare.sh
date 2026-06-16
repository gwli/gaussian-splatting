#!/bin/bash
# task_sr: benchmark SR methods on real 360 frames via downscale-restore.
#   sr_compare.sh [SRC_MP4] [N_FRAMES] [GT_W] [METHODS]
#   env GT_MODE=crop|scale  (default crop)
#     crop  : native-resolution GTxGT crop from the 8K equator (REAL detail -> fair
#             SISR fidelity test; use for lanczos/rrdbnet/swinir, not cube).
#     scale : full equirect downscaled to GTxGT/2 (use for cube/360 WS-PSNR; note GT
#             is smooth so it flatters interpolation).
# LR = bicubic down x`scale`; each method upscales back; scored vs GT.
set -e
ROOT=/raid/git/gaussian-splatting
FF=linuxserver/ffmpeg; PT=nvcr.io/nvidia/pytorch:24.12-py3
SRC="${1:-p5_sr/out/scene_027_8k_highlight.mp4}"
N="${2:-8}"; GTW="${3:-1024}"; METHODS="${4:-lanczos,rrdbnet,swinir}"
SCALE="${SCALE:-2}"; GT_MODE="${GT_MODE:-crop}"
SM="$ROOT/p5_sr/sr_methods"; WK="$ROOT/p5_sr/_work/srcmp"
rm -rf "$WK"; mkdir -p "$WK/gt" "$WK/montage"; chmod -R 777 "$ROOT/p5_sr/_work"

echo ">> [1/4] extract $N GT frames (mode=$GT_MODE, ${GTW}px) from $(basename "$SRC")"
DUR=$(docker run --rm --user 0:0 -v "$ROOT":/w --entrypoint ffprobe $FF -v error \
  -show_entries format=duration -of csv=p=0 "/w/${SRC#$ROOT/}" | cut -d. -f1)
SRCH=$(docker run --rm --user 0:0 -v "$ROOT":/w --entrypoint ffprobe $FF -v error \
  -select_streams v:0 -show_entries stream=width,height -of csv=p=0 "/w/${SRC#$ROOT/}")
STEP=$(awk -v d="$DUR" -v n="$N" 'BEGIN{print d/(n+1)}')
if [ "$GT_MODE" = "crop" ]; then            # native-detail equator crop
  VF="crop=$GTW:$GTW:(iw-$GTW)/2:(ih-$GTW)/2"
else                                         # smooth full-equirect downscale
  VF="scale=$GTW:$((GTW/2))"
fi
for i in $(seq 1 "$N"); do
  T=$(awk -v s="$STEP" -v i="$i" 'BEGIN{print s*i}')
  docker run --rm --user 0:0 -v "$ROOT":/w $FF -y -ss "$T" -i "/w/${SRC#$ROOT/}" \
    -frames:v 1 -update 1 -vf "$VF" "/w/p5_sr/_work/srcmp/gt/g$(printf %02d $i).png" >/dev/null 2>&1
done
echo "   src=$SRCH  $(ls "$WK/gt" | wc -l) GT frames"

echo ">> [2/4] fetch weights"
WDIR="$ROOT/p5_sr/weights"; mkdir -p "$WDIR"; chmod 777 "$WDIR"
RRDB="$WDIR/RealESRGAN_x${SCALE}plus.pth"
[ -f "$RRDB" ] || curl -fsSL "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x${SCALE}plus.pth" -o "$RRDB"
SWIN="$WDIR/SwinIR_classicalSR_x${SCALE}.pth"
[ -f "$SWIN" ] || curl -fsSL "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x${SCALE}.pth" -o "$SWIN"
ls -la "$RRDB" "$SWIN"

echo ">> [3/4] run benchmark (methods=$METHODS)"
docker run --rm --gpus all --ipc=host --user 0:0 -v "$ROOT":/w -w /w $PT bash -c "
  pip install -q --no-deps opencv-python-headless piq timm 2>&1 | tail -1
  python /w/p5_sr/sr_methods/sr_eval.py /w/p5_sr/_work/srcmp/gt \
    /w/p5_sr/_work/srcmp/results.json '$METHODS' \
    --scale $SCALE --rrdb /w/${RRDB#$ROOT/} --swinir /w/${SWIN#$ROOT/} \
    --tile 0 --fp16 --montage /w/p5_sr/_work/srcmp/montage
" 2>&1 | grep -aE "eval\]|psnr=|SUMMARY|method|----|lanczos|rrdbnet|swinir|Error|Traceback" | tail -40

echo ">> [4/4] results -> $WK/results.json , montages -> $WK/montage/"
