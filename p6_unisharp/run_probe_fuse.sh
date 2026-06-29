#!/bin/bash
# Direction 3 (motion-scale probe) + Direction 2 (multi-frame fusion) for scene 027.
set -e
ROOT=/raid/git/gaussian-splatting; US=$ROOT/p6_unisharp/UniSHARP; FF=linuxserver/ffmpeg

echo "############ Direction 3: motion-scale probe (forward up to 6m) ############"
UNI_FORWARD_M=6.0 UNI_FWD_FRAC=0.30 UNI_ROTATE_M=3.0 UNI_ROT_FRAC=0.15 \
  bash $ROOT/p6_unisharp/run_unisharp.sh inputs/watertown_027.jpg outputs/probe_027 \
  > $ROOT/p6_unisharp/probe.log 2>&1 || true
echo "probe frames: $(ls $US/outputs/probe_027/inputs_watertown_027/forward_erp/*.png 2>/dev/null | wc -l)"
grep -aE "forward_distance_m\"|rotate_radius_m\"" $US/outputs/probe_027/inputs_watertown_027/metadata.json 2>/dev/null | head

echo "############ Direction 2: fuse 3 frames of scene_027hf via VGGT poses ############"
for k in 060 120 180; do
  SRC=$ROOT/data/8kpano/scenes/scene_027hf_pano/panoramas/pano_0${k}.jpg
  [ -f "$SRC" ] || { echo "no $SRC"; continue; }
  docker run --rm --user 0:0 -v "$ROOT":/w $FF -y -i "/w/${SRC#$ROOT/}" -vf scale=2048:1024 \
    "/w/p6_unisharp/UniSHARP/inputs/fuse_027_${k}.jpg" >/dev/null 2>&1
  bash $ROOT/p6_unisharp/run_unisharp.sh "inputs/fuse_027_${k}.jpg" "outputs/fuse_027_${k}" \
    > $ROOT/p6_unisharp/fuse_${k}.log 2>&1 || true
  echo "  frame $k ply: $(ls -la $US/outputs/fuse_027_${k}/inputs_fuse_027_${k}/gaussians.ply 2>/dev/null|awk '{printf "%.0fMB",$5/1e6}')"
done
echo ">> fusing via pano_cams_scene_027hf poses (idx 60/120/180)"
docker run --rm --user 0:0 -v "$ROOT":/w nvcr.io/nvidia/pytorch:24.12-py3 bash -c "
  pip install -q --no-deps plyfile 2>/dev/null
  cd /w && python p6_unisharp/fuse_unisharp.py p3_pano/pano_cams_scene_027hf.json p6_unisharp/UniSHARP/outputs/fused_027.ply --scale 30 \
    --plys 60:p6_unisharp/UniSHARP/outputs/fuse_027_060/inputs_fuse_027_060/gaussians.ply \
           120:p6_unisharp/UniSHARP/outputs/fuse_027_120/inputs_fuse_027_120/gaussians.ply \
           180:p6_unisharp/UniSHARP/outputs/fuse_027_180/inputs_fuse_027_180/gaussians.ply" 2>&1 | grep -aE "fuse\]|\+[0-9]|Error|Traceback" | tail -8
PLY=$US/outputs/fused_027.ply
[ -f "$PLY" ] && bash $ROOT/pano_pipeline/ply_to_ksplat.sh "$PLY" $ROOT/data/8kpano/scenes/unisharp_027_fused.ksplat 1 2>&1 | tail -1
echo "############ DONE ############"
