#!/bin/bash
# task3 stage-3: enhance / super-resolve a standardized equirect mp4.
#
#   enhance.sh <in_equirect.mp4> <out.mp4> [MODE] [ENGINE]
#     MODE   : 2x | enhance     (2x = super-resolution; enhance = same-res quality boost)
#     ENGINE : ffmpeg | realesrgan | swinir | vsr
#   env: SCALE(2|4) TILE FP16(1) CRF GOP NVENC(0) MODEL(weights url/path) CUBE(0)
#        VSR_WIN(12) VSR_OVERLAP(2) DENOISE(hqdn3d|nlmeans|none)
#
# Engines (see p5_sr/sr_methods/SR_COMPARISON.md for the benchmark behind these):
#   ffmpeg     : hqdn3d/nlmeans denoise + deband + unsharp; 2x adds lanczos. Fast,
#                highest fidelity on light degradation. No real detail synthesis.
#   realesrgan : RRDBNet GAN SISR (tiled). Best for HEAVY real-world degradation.
#   swinir     : SwinIR transformer SISR (tiled, fp32). Best single-frame detail
#                recovery at minimal fidelity loss; slowest SISR.
#   vsr        : Real-BasicVSR x4 video SR (chunked windows). Best for video — uses
#                inter-frame info, ~19% less temporal flicker. SCALE forced to 4.
#   CUBE=1     : realesrgan/swinir operate on cubemap faces (only worth it when the
#                poles carry real detail; adds resample blur otherwise).
set -e
ROOT=/raid/git/gaussian-splatting
FF=linuxserver/ffmpeg; PT=nvcr.io/nvidia/pytorch:24.12-py3
IN="$1"; OUT="$2"; MODE="${3:-enhance}"; ENGINE="${4:-ffmpeg}"
SCALE="${SCALE:-2}"; TILE="${TILE:-512}"; CRF="${CRF:-18}"; GOP="${GOP:-60}"
CUBE="${CUBE:-0}"; VSR_WIN="${VSR_WIN:-12}"; VSR_OVERLAP="${VSR_OVERLAP:-2}"
# NVENC default OFF: compute GPUs (H100/A100) have no NVENC encoder block.
# Set NVENC=1 on machines with a consumer/pro GPU that has NVENC.
NVENC="${NVENC:-0}"; FP16="${FP16:-1}"
MODEL="${MODEL:-}"            # empty -> each engine picks its own default weights
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
  # ---- model engines (realesrgan | swinir | vsr): frames -> model -> reassemble ----
  # resolve engine -> method, default weights URL, pip deps, effective scale
  CUBESFX=""; [ "$CUBE" = "1" ] && CUBESFX="-cube"
  PIPS="opencv-python-headless"; FP=""; [ "$FP16" = "1" ] && FP="--fp16"
  case "$ENGINE" in
    realesrgan)
      METHOD="rrdbnet$CUBESFX"
      DEF="https://github.com/xinntao/Real-ESRGAN/releases/download/$([ "$SCALE" = 4 ] && echo v0.1.0/RealESRGAN_x4plus.pth || echo v0.2.1/RealESRGAN_x2plus.pth)" ;;
    swinir)
      METHOD="swinir$CUBESFX"; PIPS="opencv-python-headless timm"; FP=""   # SwinIR: fp32 (fp16 -> NaN)
      DEF="https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x${SCALE}.pth" ;;
    vsr)
      SCALE=4; METHOD="realbasicvsr"
      DEF="https://download.openmmlab.com/mmediting/restorers/real_basicvsr/realbasicvsr_c64b20_1x30x8_lr5e-5_150k_reds_20211104-52f77c2c.pth" ;;
    *) echo ">> [enhance] unknown ENGINE=$ENGINE"; exit 1 ;;
  esac
  MODEL="${MODEL:-$DEF}"
  echo ">> [enhance] $ENGINE engine (method=$METHOD x$SCALE)"
  # unique work dir per run (avoid collisions when several enhance.sh run at once)
  TAG="$(basename "$OUT" | tr -c 'A-Za-z0-9' _)_$$"
  WK="$ROOT/p5_sr/_work/sr_$TAG"; rm -rf "$WK"; mkdir -p "$WK/in" "$WK/out"
  chmod -R 777 "$ROOT/p5_sr/_work" 2>/dev/null || true
  echo ">> [1/3] extract frames"
  docker run --rm --user 0:0 -v "$ROOT":/w -v "$WK":/wk $FF -y -i "$(rel "$IN")" \
    -qscale:v 1 /wk/in/f_%06d.png 2>&1 | tail -1
  WPATH="$ROOT/p5_sr/weights/$(basename "$MODEL")"; mkdir -p "$(dirname "$WPATH")"; chmod 777 "$(dirname "$WPATH")" 2>/dev/null || true
  if [ ! -f "$WPATH" ]; then
    echo ">> fetch weights $(basename "$MODEL")"
    curl -fsSL "$MODEL" -o "$WPATH" || { echo "weights download failed"; exit 1; }
  fi
  echo ">> [2/3] $METHOD inference"
  if [ "$ENGINE" = "vsr" ]; then
    RUN="python /w/p5_sr/sr_methods/vsr_run_frames.py /wk/in /wk/out /w/${WPATH#$ROOT/} --win $VSR_WIN --overlap $VSR_OVERLAP $FP"
  else
    RUN="python /w/p5_sr/sr_methods/run_frames.py /wk/in /wk/out $METHOD /w/${WPATH#$ROOT/} --scale $SCALE --tile $TILE $FP"
  fi
  docker run --rm --gpus all --ipc=host --user 0:0 -v "$ROOT":/w -v "$WK":/wk -w /w $PT bash -c "
      pip install -q --no-deps $PIPS 2>&1 | tail -1
      $RUN
    " 2>&1 | grep -aE 'run_frames|vsr_run|realesrgan|Error|Traceback' | tail -10
  echo ">> [3/3] reassemble @ ${SRCFPS}fps"
  docker run --rm $GPUFLAG --user 0:0 -v "$ROOT":/w -v "$WK":/wk $FF -y \
    -framerate "$SRCFPS" -i /wk/out/f_%06d.png $VENC $OUTOPT -an "$(rel "$OUT")" 2>&1 | tail -2
  rm -rf "$WK"
fi
if [ ! -s "$OUT" ]; then echo ">> [enhance] ERROR: no output produced (encoder/filter failed)"; exit 1; fi
echo ">> [enhance] DONE: $OUT ($(du -h "$OUT" 2>/dev/null | cut -f1))"
