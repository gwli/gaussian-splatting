#!/bin/bash
# task2: render a virtual tour highlight video from a trained pano 3DGS model.
# Renders frames with gsplat (T-F8 env), then encodes to mp4 with ffmpeg.
#   $1 = ply   $2 = pano_cams.json   $3 = out.mp4   [SHOTS] [RES] [FPS] [SECS] [MODE]
set -e
ROOT=/raid/git/gaussian-splatting
PT=nvcr.io/nvidia/pytorch:24.12-py3; FF=linuxserver/ffmpeg
PLY="$1"; CAMS="$2"; OUT="$3"
SHOTS="${4:-orbit,fly,dolly}"; RES="${5:-1920x1080}"; FPS="${6:-30}"; SECS="${7:-8}"; MODE="${8:-perspective}"
FRAMES="$ROOT/p4_tour/_frames"; rm -rf "$FRAMES"; mkdir -p "$FRAMES"
mkdir -p "$ROOT/p3_pano/.torch_ext_cache"
GLM_DST="$ROOT/p3_pano/gsplat/gsplat/cuda/csrc/third_party/glm"
[ -f "$GLM_DST/glm/gtc/type_ptr.hpp" ] || cp -r "$ROOT/submodules/diff-gaussian-rasterization/third_party/glm/glm" "$GLM_DST/" 2>/dev/null || true
PATCH="$ROOT/p3_pano/gsplat_equirect_kernel.patch"
if [ -f "$PATCH" ] && ! grep -q 'EQUIRECT' "$ROOT/p3_pano/gsplat/gsplat/cuda/include/Common.h" 2>/dev/null; then
  ( cd "$ROOT/p3_pano/gsplat" && git apply "$PATCH" ) || true
fi

echo ">> [1/2] render frames ($SHOTS, $RES, $MODE)"
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --user 0:0 \
  -e TORCH_EXTENSIONS_DIR=/w/p3_pano/.torch_ext_cache -e TORCH_CUDA_ARCH_LIST=9.0 \
  -e PYTHONPATH=/w/p3_pano/gsplat -v "$ROOT":/w -w /w $PT bash -c "
  pip install -q --no-deps ninja rich jaxtyping plyfile 2>&1 | tail -1
  python /w/p4_tour/tour_render.py /w/$PLY /w/$CAMS /w/p4_tour/_frames \
    --shots $SHOTS --res $RES --fps $FPS --secs $SECS --mode $MODE
" 2>&1 | grep -vaE "DEPRECATION|notice|satisfied|Copyright|Various|governed|developer|terms|^==|PyTorch Version|NVIDIA Release|Idiap|Caffe|Google|NEC|Deepmind|Facebook|reserved|NYU|This container|By pulling|^$|SHMEM|recommend|insufficient"

NF=$(ls "$FRAMES"/frame_*.png 2>/dev/null | wc -l)
[ "$NF" -gt 0 ] || { echo ">> NO FRAMES rendered"; exit 1; }
echo ">> [2/2] encode $NF frames -> $OUT"
mkdir -p "$(dirname "$ROOT/$OUT")"
docker run --rm --user 0:0 -v "$ROOT":/w $FF -y -framerate $FPS -i /w/p4_tour/_frames/frame_%05d.png \
  -c:v libx264 -pix_fmt yuv420p -crf 18 -movflags +faststart "/w/$OUT" 2>&1 | tail -2
echo ">> DONE: $OUT  ($(du -h "$ROOT/$OUT" 2>/dev/null | cut -f1))"
