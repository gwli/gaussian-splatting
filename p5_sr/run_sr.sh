#!/bin/bash
# task3 orchestrator: consumer 360 source -> standardized equirect -> enhanced /
# super-resolved -> 360 metadata -> validated deliverable.
#
#   run_sr.sh <input.insv|.mp4> <out.mp4> [MODE] [ENGINE] [SECS] [SS]
#     MODE   : 2x | enhance     (default 2x; ignored by vsr which is always x4)
#     ENGINE : vsr | ffmpeg | realesrgan | swinir   (default vsr)
#     SECS   : trim source to N seconds (default full; set for quick runs)
#     SS     : start offset (default 0)
#   env: TARGET_W (final deliverable width, default 7680=8K) CAP_W IFOV SCALE TILE
#        FP16 CRF NVENC MODEL CUBE VSR_WIN VSR_OVERLAP
#
# DEFAULT = Real-BasicVSR video SR (best quality + temporal stability, see
# sr_methods/SR_COMPARISON.md). VSR is x4, so to land on TARGET_W the source is
# stitched to TARGET_W/4 (8K deliverable <- 1920x960 stitch). The per-engine SR
# factor auto-sizes the stitch width unless CAP_W is set explicitly.
#
# Pipeline:  probe -> standardize(stitch/passthrough) -> enhance -> inject360 -> validate
set -e
ROOT=/raid/git/gaussian-splatting
IN="$1"; OUT="$2"; MODE="${3:-2x}"; ENGINE="${4:-vsr}"; SECS="${5:-}"; SS="${6:-0}"
[ -z "$IN" ] && { echo "usage: run_sr.sh <input> <out.mp4> [2x|enhance] [vsr|ffmpeg|realesrgan|swinir] [secs] [ss]"; exit 1; }
# intermediates must live under $ROOT (docker mounts $ROOT:/w)
WORK="$ROOT/p5_sr/_work"; mkdir -p "$WORK"; chmod 777 "$WORK" 2>/dev/null || true
STD="$WORK/std_equirect.mp4"; ENH="$WORK/enhanced.mp4"
P5="$ROOT/p5_sr"

# ---- auto-size the stitch so SR lands on TARGET_W ----
TARGET_W="${TARGET_W:-7680}"
if [ "$ENGINE" = "vsr" ]; then     SRF=4
elif [ "$MODE" = "2x" ]; then      SRF="${SCALE:-2}"
else                               SRF=1; fi          # enhance = same-res
if [ -z "$CAP_W" ]; then export CAP_W=$(( TARGET_W / SRF )); fi

echo "================ task3 360 SR pipeline ================"
echo "in=$IN  mode=$MODE  engine=$ENGINE  target=${TARGET_W}w  stitch<=${CAP_W}w (x$SRF)  secs=${SECS:-full}"

echo ">> STAGE 1+2  standardize -> equirect"
bash "$P5/standardize.sh" "$IN" "$STD" auto "" "" "" "$SECS" "$SS"

echo ">> STAGE 3    enhance ($MODE/$ENGINE)"
bash "$P5/enhance.sh" "$STD" "$ENH" "$MODE" "$ENGINE"

echo ">> STAGE 5    inject 360 metadata"
bash "$P5/inject360.sh" "$ENH" "$OUT"

echo ">> STAGE 6    validate"
bash "$P5/validate.sh" "$OUT" 1 || echo ">> (validation reported failures — see above)"
echo "================ DONE: $OUT ================"
