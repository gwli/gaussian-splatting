#!/bin/bash
# task3 stage-2: standardize a consumer 360 source to a canonical equirect mp4.
#
#   standardize.sh <input> <out_equirect.mp4> [MODE] [W] [H] [FPS] [SECS] [SS]
#     MODE   : auto | stitch | passthrough            (auto = probe-driven)
#     W,H    : output equirect size (default: probe's equirect_out, capped by CAP_W)
#     FPS    : output fps           (default: source fps)
#     SECS   : trim to N seconds    (default: full; use for quick tests)
#     SS     : start offset seconds (default: 0)
#
# Dual-fisheye (Insta360 X3 / Antigravity A1 raw .insv): two square HEVC lenses
# are reprojected fisheye->equirect (front/back hemispheres) and blended.
# Already-equirect sources are normalized (colorspace/fps/size) only.
set -e
ROOT=/raid/git/gaussian-splatting
FF=linuxserver/ffmpeg
IN="$1"; OUT="$2"; MODE="${3:-auto}"
W="${4:-}"; H="${5:-}"; FPS="${6:-}"; SECS="${7:-}"; SS="${8:-0}"
IFOV="${IFOV:-200}"            # per-lens fisheye field of view (deg)
CAP_W="${CAP_W:-7680}"         # cap output width (8K) to keep cost sane
TMP="${TMPDIR:-/tmp/claude}"; mkdir -p "$TMP"
abspath() { case "$1" in /*) echo "$1";; *) echo "$ROOT/$1";; esac; }
IN="$(abspath "$IN")"; OUT="$(abspath "$OUT")"
mkdir -p "$(dirname "$OUT")"; chmod 777 "$(dirname "$OUT")" 2>/dev/null || true

# ---- probe ----
PJ="$TMP/std_probe.json"
docker run --rm --user 0:0 -v "$ROOT":/w --entrypoint ffprobe $FF \
  -v error -print_format json -show_format -show_streams "/w/${IN#$ROOT/}" > "$PJ"
PROF="$TMP/std_prof.json"
python3 "$ROOT/p5_sr/probe360.py" --json "$PJ" "$(basename "$IN")" "$PROF" >/dev/null
read -r P_LAYOUT P_INGEST P_W P_H P_FPS < <(python3 - "$PROF" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
print(p["layout"], p["ingest"], p["equirect_out"]["w"], p["equirect_out"]["h"], p.get("fps",30))
PY
)
[ "$MODE" = "auto" ] && MODE="$P_INGEST"
W="${W:-$P_W}"; H="${H:-$P_H}"; FPS="${FPS:-$P_FPS}"
# cap width (keep 2:1)
if [ "$W" -gt "$CAP_W" ]; then H=$(( CAP_W * H / W )); W=$CAP_W; fi
W=$((W - W%2)); H=$((H - H%2))
echo ">> [standardize] layout=$P_LAYOUT mode=$MODE -> ${W}x${H}@${FPS}fps  out=$(basename "$OUT")"

TRIM=""; [ -n "$SECS" ] && TRIM="-ss $SS -t $SECS"
COLOR="-colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv"
ENC="-c:v libx265 -preset medium -crf 18 -pix_fmt yuv420p -tag:v hvc1 -movflags +faststart"

if [ "$MODE" = "stitch" ]; then
  # dual-fisheye -> equirect (front lens yaw 0, back lens yaw 180), blend the seam.
  FC="[0:0]v360=input=fisheye:output=equirect:ih_fov=$IFOV:iv_fov=$IFOV:pitch=90,scale=$W:$H[a];\
[0:1]v360=input=fisheye:output=equirect:ih_fov=$IFOV:iv_fov=$IFOV:pitch=-90,scale=$W:$H[b];\
[a][b]blend=all_mode=average,fps=$FPS,format=yuv420p[v]"
  docker run --rm --gpus all --user 0:0 -v "$ROOT":/w $FF -y $TRIM -i "/w/${IN#$ROOT/}" \
    -filter_complex "$FC" -map "[v]" $COLOR $ENC -an -sn "/w/${OUT#$ROOT/}" 2>&1 | tail -2
else
  # passthrough: normalize size/fps/colorspace of an already-equirect source
  VF="scale=$W:$H"; [ -n "$FPS" ] && VF="$VF,fps=$FPS"; VF="$VF,format=yuv420p"
  docker run --rm --user 0:0 -v "$ROOT":/w $FF -y $TRIM -i "/w/${IN#$ROOT/}" \
    -vf "$VF" -map 0:v:0 $COLOR $ENC -an -sn "/w/${OUT#$ROOT/}" 2>&1 | tail -2
fi
echo ">> [standardize] DONE: $OUT ($(du -h "$OUT" 2>/dev/null | cut -f1))"
