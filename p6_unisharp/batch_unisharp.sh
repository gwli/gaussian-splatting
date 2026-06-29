#!/bin/bash
# Direction 1: batch UniSHARP over all scenes — one representative pano each ->
# sharp monocular 3DGS -> ksplat served for the viewer.
set -u
ROOT=/raid/git/gaussian-splatting
US=$ROOT/p6_unisharp/UniSHARP; FF=linuxserver/ffmpeg
for S in 021 022 023 025 026 027 028; do
  echo "########## UniSHARP scene_$S ##########"
  SRC=$ROOT/data/8kpano/scenes/scene_${S}hf_pano/panoramas/pano_0120.jpg
  KS=$ROOT/data/8kpano/scenes/unisharp_${S}.ksplat
  [ -f "$KS" ] && { echo "[$S] ksplat exists, skip"; continue; }
  # 1) representative frame -> 2048x1024 ERP
  docker run --rm --user 0:0 -v "$ROOT":/w $FF -y -i "/w/${SRC#$ROOT/}" \
    -vf scale=2048:1024 "/w/p6_unisharp/UniSHARP/inputs/uni_${S}.jpg" >/dev/null 2>&1
  # 2) UniSHARP infer + save ply (env cached)
  bash $ROOT/p6_unisharp/run_unisharp.sh "inputs/uni_${S}.jpg" "outputs/uni_${S}" \
    > $ROOT/p6_unisharp/uni_${S}.log 2>&1
  PLY=$US/outputs/uni_${S}/inputs_uni_${S}/gaussians.ply   # subdir = input path with / -> _
  if [ ! -f "$PLY" ]; then echo "[$S] FAILED (no ply) — see uni_${S}.log"; tail -3 $ROOT/p6_unisharp/uni_${S}.log; continue; fi
  # 3) ply -> ksplat (served at /data/8kpano/scenes/unisharp_$S.ksplat)
  bash $ROOT/pano_pipeline/ply_to_ksplat.sh "$PLY" "$KS" 1 >/dev/null 2>&1
  echo "[$S] done: $(ls -la "$KS" 2>/dev/null | awk '{printf "%.0fMB", $5/1e6}')"
done
echo "########## UniSHARP BATCH DONE ##########"
for S in 021 022 023 025 026 027 028; do
  echo -n "scene_$S: "; ls -la $ROOT/data/8kpano/scenes/unisharp_${S}.ksplat 2>/dev/null | awk '{printf "%.0fMB\n",$5/1e6}' || echo "MISSING"
done
