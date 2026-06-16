#!/bin/bash
# task2: beat-synced auto-cut tour. Renders shots as clips, detects music beats,
# hard-cuts between shots on the beat, muxes the music.
#   $1 ply  $2 cams  $3 out.mp4  [SHOTS] [RES] [FPS] [SECS] [MUSIC.wav] [BPM] [K]
# If MUSIC is empty, a click track at BPM is synthesized.
set -e
ROOT=/raid/git/gaussian-splatting
PT=nvcr.io/nvidia/pytorch:24.12-py3; FF=linuxserver/ffmpeg
PLY="$1"; CAMS="$2"; OUT="$3"
SHOTS="${4:-orbit,fly,dolly}"; RES="${5:-1280x720}"; FPS="${6:-30}"; SECS="${7:-8}"
MUSIC="${8:-}"; BPM="${9:-120}"; K="${10:-2}"
XF="$ROOT/p4_tour/_bframes"; rm -rf "$XF"; mkdir -p "$XF"
mkdir -p "$ROOT/p3_pano/.torch_ext_cache"
GLM="$ROOT/p3_pano/gsplat/gsplat/cuda/csrc/third_party/glm"
[ -f "$GLM/glm/gtc/type_ptr.hpp" ] || cp -r "$ROOT/submodules/diff-gaussian-rasterization/third_party/glm/glm" "$GLM/" 2>/dev/null || true
PATCH="$ROOT/p3_pano/gsplat_equirect_kernel.patch"
if [ -f "$PATCH" ] && ! grep -q 'EQUIRECT' "$ROOT/p3_pano/gsplat/gsplat/cuda/include/Common.h" 2>/dev/null; then ( cd "$ROOT/p3_pano/gsplat" && git apply "$PATCH" ) || true; fi

echo ">> [1/4] render split shots"
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --user 0:0 \
  -e TORCH_EXTENSIONS_DIR=/w/p3_pano/.torch_ext_cache -e TORCH_CUDA_ARCH_LIST=9.0 \
  -e PYTHONPATH=/w/p3_pano/gsplat -v "$ROOT":/w -w /w $PT bash -c "
  pip install -q --no-deps ninja rich jaxtyping plyfile 2>&1 | tail -1
  python /w/p4_tour/tour_render.py /w/$PLY /w/$CAMS /w/p4_tour/_bframes \
    --shots $SHOTS --res $RES --fps $FPS --secs $SECS --split
" 2>&1 | grep -aE "tour\]|Error|Traceback" | tail -6
[ -f "$XF/segments.json" ] || { echo ">> no manifest"; exit 1; }
mapfile -t DIRS < <(python3 -c "import json;[print(s['dir']) for s in json.load(open('$XF/segments.json'))['segments']]")
N=${#DIRS[@]}
for i in $(seq 0 $((N-1))); do
  docker run --rm --user 0:0 -v "$ROOT":/w $FF -y -framerate $FPS -i "/w/p4_tour/_bframes/${DIRS[$i]}/frame_%05d.png" \
    -c:v libx264 -pix_fmt yuv420p -crf 16 "/w/p4_tour/_bframes/clip$i.mp4" >/dev/null 2>&1
done

echo ">> [2/4] music + beat detection"
TOTAL=$(awk -v n=$N -v s=$SECS 'BEGIN{print n*s}')
if [ -z "$MUSIC" ]; then
  MUSIC="p4_tour/_bframes/click.wav"
  docker run --rm --user 0:0 -v "$ROOT":/w $PT python3 /w/p4_tour/beat_sync.py gen-click "/w/$MUSIC" "$BPM" "$TOTAL" 2>&1 | grep -aE 'beat\]' || true
fi
docker run --rm --user 0:0 -v "$ROOT":/w $PT python3 /w/p4_tour/beat_sync.py plan "/w/$MUSIC" /w/p4_tour/_bframes/segments.json "$K" /w/p4_tour/_bframes/plan.json 2>&1 | grep -aE 'beat\]' || true

echo ">> [3/4] trim beat segments"
NSEG=$(python3 -c "import json;print(len(json.load(open('$XF/plan.json'))['plan']))")
: > "$XF/concat.txt"
for j in $(seq 0 $((NSEG-1))); do
  read -r CLIP SS DUR < <(python3 -c "import json;p=json.load(open('$XF/plan.json'))['plan'][$j];print(p['clip'],p['src_start'],p['dur'])")
  docker run --rm --user 0:0 -v "$ROOT":/w $FF -y -ss "$SS" -t "$DUR" -i "/w/p4_tour/_bframes/clip$CLIP.mp4" \
    -c:v libx264 -pix_fmt yuv420p -crf 18 "/w/p4_tour/_bframes/seg$j.mp4" >/dev/null 2>&1
  echo "file 'seg$j.mp4'" >> "$XF/concat.txt"
done

echo ">> [4/4] concat + mux music -> $OUT"
mkdir -p "$(dirname "$ROOT/$OUT")"
docker run --rm --user 0:0 -v "$ROOT":/w $FF -y -f concat -safe 0 -i /w/p4_tour/_bframes/concat.txt \
  -i "/w/$MUSIC" -map 0:v -map 1:a -shortest -c:v libx264 -pix_fmt yuv420p -crf 18 \
  -c:a aac -movflags +faststart "/w/$OUT" 2>&1 | tail -2
echo ">> DONE: $OUT ($(du -h "$ROOT/$OUT" 2>/dev/null | cut -f1))"
