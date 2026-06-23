#!/bin/bash
# task3 extra: extract a 16:9 flat (rectilinear) "virtual camera" shot from a 360
# equirect video — animated yaw pan + pitch tilt + slow push-in via ffmpeg v360 +
# sendcmd. Output is a normal flat video (NOT 360; no spherical metadata).
#
#   flat_shot.sh <in_equirect.mp4> <out.mp4> [RES] [SECS] [SS]
#   env: YAW0/YAW1 (pan start/end deg) PITCH0/PITCH1 (negative=look down)
#        HFOV0/HFOV1 (push-in deg) FPS(def src) CRF
#
# NOTE on non-level footage: these drone 360s are NOT gravity-leveled (the craft
# banks), so the horizon CANTS differently per yaw — a near-horizon pan looks
# tilted at mid-yaw. Defaults therefore use a STEEP look-down (top-down aerial)
# so the horizon stays out of frame and the move is canting-immune. For properly
# leveled sources, raise PITCH (e.g. -15) for a horizon-level pan, or pre-level
# the sphere with a v360 roll/pitch pass first.
set -e
ROOT=/raid/git/gaussian-splatting
FF=linuxserver/ffmpeg; PT=nvcr.io/nvidia/pytorch:24.12-py3
IN="$1"; OUT="$2"; RES="${3:-1920x1080}"; SECS="${4:-}"; SS="${5:-0}"
YAW0="${YAW0:--80}"; YAW1="${YAW1:--30}"; PITCH0="${PITCH0:--42}"; PITCH1="${PITCH1:--34}"
HFOV0="${HFOV0:-88}"; HFOV1="${HFOV1:-78}"; CRF="${CRF:-18}"; FP16="${FP16:-1}"
abspath(){ case "$1" in /*) echo "$1";; *) echo "$ROOT/$1";; esac; }
IN="$(abspath "$IN")"; OUT="$(abspath "$OUT")"
mkdir -p "$(dirname "$OUT")"; chmod 777 "$(dirname "$OUT")" 2>/dev/null || true
rel(){ echo "/w/${1#$ROOT/}"; }
W=${RES%x*}; H=${RES#*x}
TAG="$(basename "$OUT" | tr -c 'A-Za-z0-9' _)_$$"
WK="$ROOT/p5_sr/_work/flat_$TAG"; rm -rf "$WK"; mkdir -p "$WK/eq" "$WK/flat"
chmod -R 777 "$ROOT/p5_sr/_work"
SRCFPS=$(docker run --rm --user 0:0 -v "$ROOT":/w --entrypoint ffprobe $FF -v error \
  -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 "$(rel "$IN")" | head -1)
SRCFPS=${SRCFPS%%,*}                                  # ffprobe csv can append a trailing ','
rm -f "$OUT"                                          # don't let a stale OUT mask an encode failure
TRIM=""; [ -n "$SECS" ] && TRIM="-ss $SS -t $SECS"

# v360 params aren't runtime-animatable via sendcmd, so we reproject per frame in
# torch (flat_camera.py): equirect frames -> rectilinear views with eased
# yaw/pitch/h_fov path -> encode. NOTE: these drone 360s aren't gravity-leveled,
# so a near-horizon pan cants per-yaw; steep look-down defaults dodge that.
echo ">> [1/3] extract equirect frames"
docker run --rm --user 0:0 -v "$ROOT":/w -v "$WK":/wk $FF -y $TRIM -i "$(rel "$IN")" \
  -qscale:v 2 /wk/eq/f_%05d.png 2>&1 | tail -1
echo ">> [2/3] reproject ${W}x${H}  yaw $YAW0->$YAW1 pitch $PITCH0->$PITCH1 hfov $HFOV0->$HFOV1"
FP=""; [ "$FP16" = "1" ] && FP="--fp16"
docker run --rm --gpus all --ipc=host --user 0:0 -v "$ROOT":/w -v "$WK":/wk -w /w $PT bash -c "
  pip install -q --no-deps opencv-python-headless 2>&1 | tail -1
  python /w/p5_sr/flat_camera.py /wk/eq /wk/flat --res $RES \
    --yaw0 $YAW0 --yaw1 $YAW1 --pitch0 $PITCH0 --pitch1 $PITCH1 --hfov0 $HFOV0 --hfov1 $HFOV1 $FP
" 2>&1 | grep -aE "flat\]|Error|Traceback" | tail -6
echo ">> [3/3] encode @ ${SRCFPS}fps -> $(basename "$OUT")"
docker run --rm --user 0:0 -v "$ROOT":/w -v "$WK":/wk $FF -y -framerate "$SRCFPS" -i /wk/flat/f_%05d.png \
  -c:v libx265 -preset medium -crf $CRF -pix_fmt yuv420p -tag:v hvc1 -movflags +faststart -an "$(rel "$OUT")" 2>&1 | tail -2
rm -rf "$WK"
[ -s "$OUT" ] || { echo ">> [flat] ERROR: no output"; exit 1; }
echo ">> [flat] DONE: $OUT ($(du -h "$OUT" 2>/dev/null | cut -f1))"
