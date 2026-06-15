#!/bin/bash
# task2 follow-up: batch tour highlights over all scenes + a 360 equirect tour.
#   produces p4_tour/out/<scene>_highlight.mp4 (+ graded), and scene_023 equirect.
set -e
ROOT=/raid/git/gaussian-splatting
RT=$ROOT/p4_tour/run_tour.sh; POST=$ROOT/p4_tour/post.sh
OUT=p4_tour/out; mkdir -p "$ROOT/$OUT"
for s in 021 022 023 025 026 027 028; do
  PLY=data/8kpano/scenes/scene_${s}_pano/output_pano/point_cloud/iteration_7000/point_cloud.ply
  [ -f "$ROOT/$PLY" ] || { echo "[$s] no ply, skip"; continue; }
  echo "########## scene_$s ##########"
  bash "$RT" "$PLY" p3_pano/pano_cams_scene_${s}.json "$OUT/scene_${s}_highlight.mp4" \
    orbit,fly,dolly 1920x1080 30 6 perspective > "$ROOT/data/8kpano/tour_${s}.log" 2>&1 \
    && bash "$POST" "$OUT/scene_${s}_highlight.mp4" "$OUT/scene_${s}_graded.mp4" "SCENE ${s} · 3DGS AERIAL TOUR" \
       > "$ROOT/data/8kpano/post_${s}.log" 2>&1 \
    && echo "[$s] OK" || echo "[$s] FAILED"
done
echo "########## 360 equirect tour (scene_023) ##########"
bash "$RT" data/8kpano/scenes/scene_023_pano/output_pano/point_cloud/iteration_7000/point_cloud.ply \
  p3_pano/pano_cams_scene_023.json "$OUT/scene_023_equirect360.mp4" \
  orbit,fly 0x0 30 6 equirect > "$ROOT/data/8kpano/tour_023_eq.log" 2>&1 \
  && echo "[023 eq] OK" || echo "[023 eq] FAILED"
echo "########## SUMMARY ##########"
ls -la "$ROOT/$OUT"/*.mp4 2>/dev/null | awk '{print $5, $NF}'
