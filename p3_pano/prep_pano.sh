#!/bin/bash
# Clean prep for direct-pano training: stitch panoramas (KEPT), crop, run VGGT.
# Everything stays index-aligned by construction.
# Usage: prep_pano.sh <scene_name> <insv> <n_panos> [SEL=uniform|fps] [POOLX=3]
#   SEL=fps (T-G): stitch a POOLX*N pool, coarse 1-crop VGGT, farthest-point-select
#   N panos by camera pose (max viewpoint spread, ~+1 dB vs uniform). Default uniform.
set -e
S="$1"; INSV="$2"; NP="${3:-90}"; SEL="${4:-uniform}"; POOLX="${5:-3}"
ROOT=/raid/git/gaussian-splatting
D=$ROOT/data/8kpano/scenes/${S}_pano
FF=linuxserver/ffmpeg; PT=nvcr.io/nvidia/pytorch:24.12-py3
rm -rf "$D"; mkdir -p "$D/panoramas" "$D/images"; chmod -R 777 "$D"

if [ "$SEL" = "fps" ]; then STITCH_NP=$((NP*POOLX)); else STITCH_NP=$NP; fi
DUR=$(docker run --rm --user 0:0 -v $ROOT/data/8kpano:/d $FF -i /d/$INSV 2>&1 | grep Duration | awk '{print $2}' | tr -d ',' | awk -F: '{print ($1*3600+$2*60+$3)}')
FPS=$(awk -v t=$STITCH_NP -v d=$DUR 'BEGIN{printf "%.5f", t/d}')
echo "[$S] dur=${DUR}s fps=$FPS for $STITCH_NP panos (SEL=$SEL)"

echo "[1/3] stitch panoramas (kept)..."
docker run --rm --gpus all --user 0:0 -v $ROOT/data/8kpano:/data $FF \
  -hwaccel cuda -i /data/$INSV \
  -filter_complex "[0:0]v360=input=fisheye:output=equirect:ih_fov=200:iv_fov=200:pitch=90[a];[0:1]v360=input=fisheye:output=equirect:ih_fov=200:iv_fov=200:pitch=-90[b];[a][b]blend=all_mode=average,fps=$FPS,scale=4096:2048" \
  -q:v 2 /data/scenes/${S}_pano/panoramas/pano_%04d.jpg 2>&1 | tail -1
echo "  panos: $(ls $D/panoramas/*.jpg | wc -l)"

if [ "$SEL" = "fps" ]; then
  echo "[1b] FPS-select $NP of $STITCH_NP via coarse 1-crop VGGT..."
  rm -rf "$D/_pool"; mkdir -p "$D/_pool/images"; chmod -R 777 "$D/_pool"
  docker run --rm --user 0:0 -v $ROOT:/w $PT bash -c "
    pip install -q --no-deps opencv-python 2>/dev/null
    cd /w && python pano_pipeline/pano_to_perspective.py \
      -i /w/data/8kpano/scenes/${S}_pano/panoramas \
      -o /w/data/8kpano/scenes/${S}_pano/_pool/images \
      --directions '0,0' --fov 120 --size 1024 --quality 92" 2>&1 | tail -1
  docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --user 0:0 \
    -e TORCH_HOME=/th -v $ROOT:/w -v $ROOT/p2_vggt/weights:/wcache:ro $PT bash -c "
    mkdir -p /th
    pip install -q --no-deps einops safetensors trimesh huggingface_hub 'pycolmap==3.10.0' 2>&1 | tail -1
    cd /w/p2_vggt/vggt && export PYTHONPATH=\$PWD
    python demo_colmap.py --scene_dir /w/data/8kpano/scenes/${S}_pano/_pool --conf_thres_value 1.5 2>&1 | tail -2" 2>&1 | tail -2
  if [ -f "$D/_pool/sparse/cameras.bin" ]; then mkdir -p "$D/_pool/sparse/0"; mv "$D/_pool/sparse"/*.bin "$D/_pool/sparse/0/" 2>/dev/null||true; fi
  docker run --rm --user 0:0 -v $ROOT:/w $PT python /w/p3_pano/fps_select.py /w/data/8kpano/scenes/${S}_pano/_pool $NP 2>/dev/null \
    | grep -E '^pano_[0-9]+$' > "$D/_keep.txt"
  KEEP=$(wc -l < "$D/_keep.txt")
  echo "  FPS kept $KEEP panos"
  # prune panoramas/ to the selected set
  mkdir -p "$D/_sel"
  while read -r nm; do [ -f "$D/panoramas/${nm}.jpg" ] && mv "$D/panoramas/${nm}.jpg" "$D/_sel/"; done < "$D/_keep.txt"
  rm -rf "$D/panoramas" "$D/_pool"; mv "$D/_sel" "$D/panoramas"
  # renumber to pano_0001.. for downstream index-alignment
  i=1; for f in $(ls "$D/panoramas"/*.jpg | sort); do mv "$f" "$D/panoramas/$(printf '_r%04d.jpg' $i)"; i=$((i+1)); done
  for f in "$D/panoramas"/_r*.jpg; do mv "$f" "${f/_r/pano_}"; done
  echo "  panos after FPS: $(ls $D/panoramas/*.jpg | wc -l)"
fi

echo "[2/3] crop to perspective (FOV120 standard)..."
docker run --rm --user 0:0 -v $ROOT:/w $PT bash -c "
  pip install -q --no-deps opencv-python 2>/dev/null
  cd /w && python pano_pipeline/pano_to_perspective.py \
    -i /w/data/8kpano/scenes/${S}_pano/panoramas \
    -o /w/data/8kpano/scenes/${S}_pano/images \
    --fov 120 --size 1280 --preset standard --quality 92" 2>&1 | tail -2
echo "  crops: $(ls $D/images/*.jpg | wc -l)"

# --- curate crops to <=300 for VGGT (O(N^2) attention memory); keep panoramas full ---
echo "[2b] curating crops -> <=300 (<=3/pano, front-first)..."
mv "$D/images" "$D/images_full"; mkdir "$D/images"
python3 - "$D/images_full" "$D/images" <<'PY'
import os, re, glob, shutil, sys
src, dst = sys.argv[1], sys.argv[2]
byp = {}
for f in sorted(glob.glob(src + "/*.jpg")):
    m = re.search(r"pano_(\d+)_", os.path.basename(f))
    if m: byp.setdefault(int(m.group(1)), []).append(f)
cap, per = 300, 3
# shrink per-pano if too many panos
while len(byp) * per > cap and per > 1: per -= 1
tot = 0
for idx, fs in byp.items():
    fs = sorted(fs, key=lambda p: (("y+000_p+00" not in p), p))  # front first
    pick = [fs[0]] + ([fs[len(fs)//2]] if len(fs) > 1 else []) + ([fs[-1]] if len(fs) > 2 else [])
    for p in pick[:per]:
        shutil.copy(p, dst); tot += 1
print(f"  curated {tot} crops over {len(byp)} panos ({per}/pano)")
PY
chmod -R 777 "$D/images"
echo "  curated: $(ls $D/images/*.jpg | wc -l)"

echo "[3/3] VGGT SfM on crops..."
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --user 0:0 \
  -e TORCH_HOME=/th -v $ROOT:/w -v $ROOT/p2_vggt/weights:/wcache:ro $PT bash -c "
  mkdir -p /th
  pip install -q --no-deps einops safetensors trimesh huggingface_hub 'pycolmap==3.10.0' 2>&1 | tail -1
  cd /w/p2_vggt/vggt && export PYTHONPATH=\$PWD
  python demo_colmap.py --scene_dir /w/data/8kpano/scenes/${S}_pano --conf_thres_value 1.5 2>&1 | tail -6
" 2>&1 | grep -vE "DEPRECATION|notice|satisfied|Copyright|Various|governed|developer|terms|^==|PyTorch Version|NVIDIA Release|Idiap|Caffe|Google|NEC|Deepmind|Facebook|reserved|NYU|This container|By pulling|^$"
# restructure sparse → sparse/0
if [ -f "$D/sparse/cameras.bin" ]; then mkdir -p "$D/sparse/0"; mv "$D/sparse"/*.bin "$D/sparse/0/" 2>/dev/null||true; mv "$D/sparse"/*.ply "$D/sparse/0/" 2>/dev/null||true; fi
echo "[$S] PREP DONE: $(ls $D/sparse/0/*.bin 2>/dev/null|wc -l) bin, panos=$(ls $D/panoramas/*.jpg|wc -l), crops=$(ls $D/images/*.jpg|wc -l)"
