#!/bin/bash
# task2 step-7 post-production: color grade + fade in/out + optional title + music.
# ffmpeg-only, no re-render. Usage:
#   post.sh <in.mp4> <out.mp4> [TITLE] [MUSIC.(mp3|m4a|wav)]
set -e
ROOT=/raid/git/gaussian-splatting
FF=linuxserver/ffmpeg
IN="$1"; OUT="$2"; TITLE="${3:-}"; MUSIC="${4:-}"
[ -f "$ROOT/$IN" ] || { echo "ERROR: $IN not found"; exit 1; }
DUR=$(docker run --rm --user 0:0 -v "$ROOT":/w $FF -i "/w/$IN" 2>&1 | grep -m1 Duration | awk '{print $2}' | tr -d ',' | awk -F: '{print ($1*3600+$2*60+$3)}')
FOUT=$(awk -v d="$DUR" 'BEGIN{printf "%.2f", (d>2? d-1.0 : d*0.9)}')

# color grade: gentle contrast/saturation lift + mild S-curve + vignette;
# cinematic fade in (0.6s) and fade out (last 1s).
VF="eq=contrast=1.08:saturation=1.18:gamma=0.98:brightness=0.01,unsharp=5:5:0.6,vignette=PI/5,fade=t=in:st=0:d=0.6,fade=t=out:st=${FOUT}:d=1.0"
if [ -n "$TITLE" ]; then
  VF="$VF,drawtext=text='${TITLE}':fontcolor=white:fontsize=h/18:x=(w-text_w)/2:y=h*0.82:box=1:boxcolor=black@0.35:boxborderw=14:enable='between(t,0.4,3.5)':alpha='if(lt(t,0.6),(t-0.4)/0.2,if(gt(t,3.1),(3.5-t)/0.4,1))'"
fi

mkdir -p "$(dirname "$ROOT/$OUT")"
if [ -n "$MUSIC" ] && [ -f "$ROOT/$MUSIC" ]; then
  echo ">> grade + fade + title + music -> $OUT"
  docker run --rm --user 0:0 -v "$ROOT":/w $FF -y -i "/w/$IN" -i "/w/$MUSIC" \
    -vf "$VF" -af "afade=t=in:st=0:d=0.8,afade=t=out:st=${FOUT}:d=1.0" \
    -map 0:v -map 1:a -shortest -c:v libx264 -crf 18 -pix_fmt yuv420p -c:a aac -movflags +faststart "/w/$OUT" 2>&1 | tail -2
else
  echo ">> grade + fade + title (no music) -> $OUT"
  docker run --rm --user 0:0 -v "$ROOT":/w $FF -y -i "/w/$IN" \
    -vf "$VF" -c:v libx264 -crf 18 -pix_fmt yuv420p -movflags +faststart "/w/$OUT" 2>&1 | tail -2
fi
echo ">> DONE: $OUT ($(du -h "$ROOT/$OUT" 2>/dev/null | cut -f1))"
