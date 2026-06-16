#!/bin/bash
# task3 stage-3: enhance / super-resolve a standardized equirect mp4.
#
#   enhance.sh <in_equirect.mp4> <out.mp4> [MODE] [ENGINE]
#     MODE   : 2x | enhance        (2x = super-resolution; enhance = same-res quality boost)
#     ENGINE : ffmpeg | realesrgan (default ffmpeg; realesrgan only for MODE=2x/4x)
#   env: SCALE(2|4) TILE FP16(1) CRF GOP NVENC(1) MODEL(weights url/path)
#
# ffmpeg engine  : nlmeans/hqdn3d denoise + deband (kill compression artifacts) +
#                  unsharp detail; 2x adds lanczos upscale. GPU (nvenc) encode.
# realesrgan engine: extract frames -> RRDBNet x2/x4 (tiled) -> re-encode.
set -e
ROOT=/raid/git/gaussian-splatting
FF=linuxserver/ffmpeg; PT=nvcr.io/nvidia/pytorch:24.12-py3
IN="$1"; OUT="$2"; MODE="${3:-enhance}"; ENGINE="${4:-ffmpeg}"
SCALE="${SCALE:-2}"; TILE="${TILE:-512}"; CRF="${CRF:-18}"; GOP="${GOP:-60}"
# NVENC default OFF: compute GPUs (H100/A100) have no NVENC encoder block.
# Set NVENC=1 on machines with a consumer/pro GPU that has NVENC.
NVENC="${NVENC:-0}"; FP16="${FP16:-1}"
MODEL="${MODEL:-https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth}"
TMP="${TMPDIR:-/tmp/claude}"; mkdir -p "$TMP"
abspath() { case "$1" in /*) echo "$1";; *) echo "$ROOT/$1";; esac; }
IN="$(abspath "$IN")"; OUT="$(abspath "$OUT")"
mkdir -p "$(dirname "$OUT")"; chmod 777 "$(dirname "$OUT")" 2>/dev/null || true
rel() { echo "/w/${1#$ROOT/}"; }

# encoder selection
if [ "$NVENC" = "1" ]; then
  VENC="-c:v hevc_nvenc -preset p5 -rc vbr -cq $CRF -b:v 0 -tag:v hvc1 -g $GOP"
  GPUFLAG="--gpus all"
else
  VENC="-c:v libx265 -preset ${PRESET:-medium} -crf $CRF -tag:v hvc1 -x265-params keyint=$GOP"
  GPUFLAG=""
fi
OUTOPT="-pix_fmt yuv420p -movflags +faststart -colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv"

# ---- pick fps for reassembly ----
SRCFPS=$(docker run --rm --user 0:0 -v "$ROOT":/w --entrypoint ffprobe $FF -v error \
  -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 "$(rel "$IN")" | head -1)

if [ "$ENGINE" = "ffmpeg" ]; then
  echo ">> [enhance] ffmpeg engine, mode=$MODE denoise=${DENOISE:-hqdn3d}"
  # denoiser: hqdn3d (fast, default — feasible on 8K) | nlmeans (higher quality, slow)
  case "${DENOISE:-hqdn3d}" in
    nlmeans) DN="nlmeans=s=1.5:p=7:r=7";;
    none)    DN="";;
    *)       DN="hqdn3d=2:1.5:3:3";;
  esac
  if [ "$MODE" = "2x" ]; then
    # denoise -> deband -> 2x lanczos -> sharpen
    VF="${DN:+$DN,}deband,scale=iw*${SCALE}:ih*${SCALE}:flags=lanczos,unsharp=5:5:0.8:5:5:0.0"
  else
    # same-res restoration: denoise + deband + detail enhance + faint grain
    VF="${DN:+$DN,}deband=range=16:blur=1,unsharp=5:5:1.0:5:5:0.0,noise=alls=2:allf=t"
  fi
  docker run --rm $GPUFLAG --user 0:0 -v "$ROOT":/w $FF -y -i "$(rel "$IN")" \
    -vf "$VF" $VENC $OUTOPT -an "$(rel "$OUT")" 2>&1 | tail -3
else
  # ---- realesrgan engine: frames -> model -> reassemble ----
  echo ">> [enhance] realesrgan engine x$SCALE"
  WK="$ROOT/p5_sr/_work/sr_work"; rm -rf "$WK"; mkdir -p "$WK/in" "$WK/out"
  chmod -R 777 "$ROOT/p5_sr/_work" 2>/dev/null || true
  echo ">> [1/3] extract frames"
  docker run --rm --user 0:0 -v "$ROOT":/w -v "$WK":/wk $FF -y -i "$(rel "$IN")" \
    -qscale:v 1 /wk/in/f_%06d.png 2>&1 | tail -1
  # fetch weights once
  WPATH="$ROOT/p5_sr/weights/$(basename "$MODEL")"; mkdir -p "$(dirname "$WPATH")"; chmod 777 "$(dirname "$WPATH")" 2>/dev/null || true
  if [ ! -f "$WPATH" ]; then
    echo ">> fetch weights $(basename "$MODEL")"
    curl -fsSL "$MODEL" -o "$WPATH" || { echo "weights download failed"; exit 1; }
  fi
  echo ">> [2/3] RRDBNet x$SCALE inference"
  FP=""; [ "$FP16" = "1" ] && FP="--fp16"
  docker run --rm --gpus all --ipc=host --user 0:0 -v "$ROOT":/w -v "$WK":/wk \
    -w /w $PT bash -c "
      pip install -q --no-deps opencv-python-headless 2>&1 | tail -1
      python /w/p5_sr/realesrgan_infer.py /wk/in /wk/out /w/${WPATH#$ROOT/} \
        --scale $SCALE --tile $TILE $FP
    " 2>&1 | grep -aE 'realesrgan|Error|Traceback' | tail -8
  echo ">> [3/3] reassemble @ ${SRCFPS}fps"
  docker run --rm $GPUFLAG --user 0:0 -v "$ROOT":/w -v "$WK":/wk $FF -y \
    -framerate "$SRCFPS" -i /wk/out/f_%06d.png $VENC $OUTOPT -an "$(rel "$OUT")" 2>&1 | tail -2
  rm -rf "$WK"
fi
if [ ! -s "$OUT" ]; then echo ">> [enhance] ERROR: no output produced (encoder/filter failed)"; exit 1; fi
echo ">> [enhance] DONE: $OUT ($(du -h "$OUT" 2>/dev/null | cut -f1))"
