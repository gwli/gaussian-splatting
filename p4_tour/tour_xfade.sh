#!/bin/bash
# task2: multi-segment CROSSFADE tour. Renders each shot as its own clip, then
# chains ffmpeg xfade transitions between consecutive shots.
#   $1 ply  $2 cams  $3 out.mp4  [SHOTS] [RES] [FPS] [SECS] [XDUR] [TRANSITION]
set -e
ROOT=/raid/git/gaussian-splatting
PT=nvcr.io/nvidia/pytorch:24.12-py3; FF=linuxserver/ffmpeg
PLY="$1"; CAMS="$2"; OUT="$3"
SHOTS="${4:-orbit,fly,dolly}"; RES="${5:-1920x1080}"; FPS="${6:-30}"; SECS="${7:-6}"
XDUR="${8:-1.0}"; TRANS="${9:-fade}"
XF="$ROOT/p4_tour/_xframes"; rm -rf "$XF"; mkdir -p "$XF"
mkdir -p "$ROOT/p3_pano/.torch_ext_cache"
GLM="$ROOT/p3_pano/gsplat/gsplat/cuda/csrc/third_party/glm"
[ -f "$GLM/glm/gtc/type_ptr.hpp" ] || cp -r "$ROOT/submodules/diff-gaussian-rasterization/third_party/glm/glm" "$GLM/" 2>/dev/null || true
PATCH="$ROOT/p3_pano/gsplat_equirect_kernel.patch"
if [ -f "$PATCH" ] && ! grep -q 'EQUIRECT' "$ROOT/p3_pano/gsplat/gsplat/cuda/include/Common.h" 2>/dev/null; then ( cd "$ROOT/p3_pano/gsplat" && git apply "$PATCH" ) || true; fi

echo ">> [1/3] render split frames ($SHOTS)"
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --user 0:0 \
  -e TORCH_EXTENSIONS_DIR=/w/p3_pano/.torch_ext_cache -e TORCH_CUDA_ARCH_LIST=9.0 \
  -e PYTHONPATH=/w/p3_pano/gsplat -v "$ROOT":/w -w /w $PT bash -c "
  pip install -q --no-deps ninja rich jaxtyping plyfile 2>&1 | tail -1
  python /w/p4_tour/tour_render.py /w/$PLY /w/$CAMS /w/p4_tour/_xframes \
    --shots $SHOTS --res $RES --fps $FPS --secs $SECS --split
" 2>&1 | grep -aE "tour\]|Error|Traceback" | tail -8

[ -f "$XF/segments.json" ] || { echo ">> no manifest"; exit 1; }
mapfile -t DIRS < <(python3 -c "import json;[print(s['dir']) for s in json.load(open('$XF/segments.json'))['segments']]")
N=${#DIRS[@]}; echo ">> [2/3] encode $N clips"
for i in $(seq 0 $((N-1))); do
  docker run --rm --user 0:0 -v "$ROOT":/w $FF -y -framerate $FPS -i "/w/p4_tour/_xframes/${DIRS[$i]}/frame_%05d.png" \
    -c:v libx264 -pix_fmt yuv420p -crf 16 "/w/p4_tour/_xframes/clip$i.mp4" >/dev/null 2>&1
done

echo ">> [3/3] xfade-chain ($TRANS, ${XDUR}s)"
if [ "$N" -eq 1 ]; then
  cp "$XF/clip0.mp4" "$ROOT/$OUT"
else
  INPUTS=""; for i in $(seq 0 $((N-1))); do INPUTS="$INPUTS -i /w/p4_tour/_xframes/clip$i.mp4"; done
  # build xfade filter chain with cumulative offsets (python reads the manifest)
  FILTER=$(python3 - "$XF/segments.json" "$FPS" "$XDUR" "$TRANS" <<'PY'
import json, sys
mf, fps, X, T = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
D = [s["nframes"] / fps for s in json.load(open(mf))["segments"]]
prev, cum, parts = "[0:v]", D[0], []
for k in range(1, len(D)):
    off = cum - X
    out = "[vout]" if k == len(D) - 1 else f"[v{k}]"
    parts.append(f"{prev}[{k}:v]xfade=transition={T}:duration={X}:offset={off:.3f}{out}")
    prev, cum = out, cum + D[k] - X
print(";".join(parts))
PY
)
  mkdir -p "$(dirname "$ROOT/$OUT")"
  docker run --rm --user 0:0 -v "$ROOT":/w $FF -y $INPUTS -filter_complex "$FILTER" \
    -map "[vout]" -c:v libx264 -pix_fmt yuv420p -crf 18 -movflags +faststart "/w/$OUT" 2>&1 | tail -2
fi
echo ">> DONE: $OUT ($(du -h "$ROOT/$OUT" 2>/dev/null | cut -f1))"
