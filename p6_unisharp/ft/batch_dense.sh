#!/bin/bash
set -u; ROOT=/raid/git/gaussian-splatting; cd $ROOT
declare -A V=( [021]=VID_20260321_171111_021.insv [022]=VID_20260321_171737_022.insv \
  [025]=VID_20260406_115153_025.insv [026]=VID_20260417_162702_026.insv \
  [027]=VID_20260502_053911_027.insv [028]=VID_20260502_164639_028.insv )
for S in 021 022 025 026 027 028; do
  SC=scene_${S}d; D=data/8kpano/scenes/${SC}_pano; mkdir -p $D/panoramas; chmod -R 777 $D
  echo "########## $SC ##########"
  DUR=$(docker run --rm --user 0:0 -v $ROOT/data/8kpano:/d linuxserver/ffmpeg -i /d/${V[$S]} 2>&1 | grep Duration | awk '{print $2}' | tr -d ',' | awk -F: '{print ($1*3600+$2*60+$3)}')
  FPS=$(awk -v d=$DUR 'BEGIN{printf "%.5f",1140/d}')
  docker run --rm --gpus all --user 0:0 -v $ROOT/data/8kpano:/data linuxserver/ffmpeg \
    -hwaccel cuda -i /data/${V[$S]} \
    -filter_complex "[0:0]v360=input=fisheye:output=equirect:ih_fov=200:iv_fov=200:pitch=90[a];[0:1]v360=input=fisheye:output=equirect:ih_fov=200:iv_fov=200:pitch=-90[b];[a][b]blend=all_mode=average,fps=$FPS,scale=4096:2048" \
    -q:v 2 /data/scenes/${SC}_pano/panoramas/pano_%04d.jpg >/dev/null 2>&1
  N=$(ls $D/panoramas/*.jpg 2>/dev/null | wc -l); echo "dur=$DUR panos=$N"
  [ "$N" -lt 500 ] && { echo "[$S] stitch FAILED"; continue; }
  python3 -c "
import json
cams=[{'idx':k,'image':f'data/8kpano/scenes/${SC}_pano/panoramas/pano_{k:04d}.jpg'} for k in range(1,$N+1)]
json.dump({'scene_dir':'data/8kpano/scenes/${SC}_pano','point_cloud':'','cameras_extent':300.0,'cameras':cams},open('p3_pano/pano_cams_${SC}.json','w'))"
  docker run --rm --user 0:0 -v "$ROOT":/w -w /w nvcr.io/nvidia/pytorch:24.12-py3 bash -c \
    ". /w/p6_unisharp/venv/bin/activate; python /w/p6_unisharp/ft/gps_hybrid_scene.py --scene $SC --insv /w/data/8kpano/${V[$S]} --dur $DUR --n $N" 2>&1 | grep -aE "\[$SC\]|Error" | tail -1
  chmod -R 777 p3_pano
  bash p3_pano/run_pano_gsplat_train.sh p3_pano/pano_cams_${SC}_gps3.json data/8kpano/scenes/${SC}_pano/output_gps3 7000 1024 512 sph 2>&1 | grep -aE "EVAL|Error" | tail -1
done; echo DENSE_BATCH_DONE
