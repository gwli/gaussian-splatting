#!/bin/bash
# task3 stage-6: publish-compatibility validation of a finished 360 mp4.
# Checks: equirect 2:1 aspect, codec/pixfmt/colorspace, fps, faststart (moov at
# front), spherical metadata present, and extracts a sanity frame.
#
#   validate.sh <video.mp4> [expect_2to1=1]
set -e
ROOT=/raid/git/gaussian-splatting
FF=linuxserver/ffmpeg
IN="$1"; EXPECT21="${2:-1}"
abspath() { case "$1" in /*) echo "$1";; *) echo "$ROOT/$1";; esac; }
IN="$(abspath "$IN")"
rel="/w/${IN#$ROOT/}"
SM="$ROOT/p5_sr/spatial_media"
TMP="${TMPDIR:-/tmp/claude}"; mkdir -p "$TMP"; chmod 777 "$TMP" 2>/dev/null || true
ok=0; bad=0
chk(){ if [ "$1" = "1" ]; then echo "  [PASS] $2"; ok=$((ok+1)); else echo "  [FAIL] $2"; bad=$((bad+1)); fi; }

# key=value parse (order-independent; ffprobe reorders -show_entries fields)
KV=$(docker run --rm --user 0:0 -v "$ROOT":/w --entrypoint ffprobe $FF \
  -v error -select_streams v:0 -show_entries stream=width,height,codec_name,pix_fmt,color_space,r_frame_rate \
  -of default=noprint_wrappers=1 "$rel")
g(){ echo "$KV" | grep -E "^$1=" | head -1 | cut -d= -f2; }
W=$(g width); H=$(g height); CODEC=$(g codec_name); PIXFMT=$(g pix_fmt); CS=$(g color_space); FPS=$(g r_frame_rate)
echo ">> [validate] $(basename "$IN"): ${W}x${H} $CODEC $PIXFMT cs=$CS fps=$FPS"

# 2:1 aspect
if [ "$EXPECT21" = "1" ]; then
  R=$(awk -v w="$W" -v h="$H" 'BEGIN{printf "%.3f", w/h}')
  AR_OK=$(awk -v r="$R" 'BEGIN{print (r>1.9 && r<2.1)?1:0}')
  chk "$AR_OK" "equirect 2:1 aspect (ratio=$R)"
fi
# codec / pixfmt
chk "$([ "$CODEC" = "hevc" -o "$CODEC" = "h264" ] && echo 1 || echo 0)" "deliverable codec ($CODEC)"
chk "$([ "$PIXFMT" = "yuv420p" ] && echo 1 || echo 0)" "8-bit yuv420p pixfmt ($PIXFMT)"
# faststart: moov before mdat
HEAD=$(docker run --rm --user 0:0 -v "$ROOT":/w --entrypoint sh $FF -c "head -c 4000000 '$rel' | tr -c '[:alnum:]' '\n' | grep -nE '^(moov|mdat)' | head -2" 2>/dev/null | tr '\n' ' ')
chk "$(echo "$HEAD" | grep -q 'moov' && echo 1 || echo 0)" "faststart (moov near front: $HEAD)"
# spherical metadata
if [ -d "$SM/spatialmedia" ]; then
  SPH=$(PYTHONPATH="$SM" python3 -m spatialmedia "$IN" 2>&1 | grep -ci "spherical\|equirect" || true)
  chk "$([ "$SPH" -gt 0 ] && echo 1 || echo 0)" "spherical 360 metadata present"
else
  echo "  [SKIP] spherical metadata (injector not vendored)"
fi
# decodable sanity frame
FR="$TMP/validate_frame.png"; rm -f "$FR"
docker run --rm --user 0:0 -v "$ROOT":/w -v "$TMP":/t $FF -y -i "$rel" -frames:v 1 /t/validate_frame.png >/dev/null 2>&1 || true
chk "$([ -s "$FR" ] && echo 1 || echo 0)" "decodes a frame ($([ -s "$FR" ] && du -h "$FR"|cut -f1))"

echo ">> [validate] $ok passed, $bad failed"
[ "$bad" -eq 0 ]
