#!/bin/bash
set -u
ROOT=/raid/git/gaussian-splatting; cd $ROOT
declare -A V=( [021]=VID_20260321_171111_021.insv [022]=VID_20260321_171737_022.insv \
  [025]=VID_20260406_115153_025.insv [026]=VID_20260417_162702_026.insv \
  [027]=VID_20260502_053911_027.insv [028]=VID_20260502_164639_028.insv )
for S in 021 022 025 026 027 028; do
  SC=scene_${S}hf; INSV=data/8kpano/${V[$S]}
  echo "########## $SC ##########"
  DUR=$(docker run --rm --user 0:0 -v $ROOT/data/8kpano:/d linuxserver/ffmpeg -i /d/${V[$S]} 2>&1 | grep Duration | awk '{print $2}' | tr -d ',' | awk -F: '{print ($1*3600+$2*60+$3)}')
  echo "dur=$DUR"
  docker run --rm --user 0:0 -v "$ROOT":/w -w /w nvcr.io/nvidia/pytorch:24.12-py3 bash -c \
    ". /w/p6_unisharp/venv/bin/activate; python /w/p6_unisharp/ft/gps_hybrid_scene.py --scene $SC --insv /w/$INSV --dur $DUR" 2>&1 | grep -aE "\[$SC\]|Error|Traceback|Assertion" | tail -2
  chmod -R 777 p3_pano 2>/dev/null
  [ -f p3_pano/pano_cams_${SC}_gps3.json ] || { echo "[$S] pose build FAILED, skip"; continue; }
  bash p3_pano/run_pano_gsplat_train.sh p3_pano/pano_cams_${SC}_gps3.json data/8kpano/scenes/${SC}_pano/output_gps3 7000 1024 512 sph 2>&1 | grep -aE "EVAL|Error" | tail -1
done
echo "########## BATCH DONE ##########"
