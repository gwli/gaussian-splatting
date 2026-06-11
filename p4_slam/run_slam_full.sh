#!/bin/bash
# Host wrapper. Builds a persistent image ONCE (docker commit), then runs main.py
# against it cheaply. Re-run this script to iterate on the run without rebuilding.
#   $1 = dataset dir (default p4_slam/seq_023_front)   $2 = save-as tag (default seq023front)
set -e
REPO=/raid/git/gaussian-splatting
IMG=mast3r-slam:built
BASE=nvcr.io/nvidia/pytorch:24.12-py3
DS=${1:-/w/p4_slam/seq_023_front}
TAG=${2:-seq023front}

if ! docker image inspect "$IMG" >/dev/null 2>&1; then
  echo ">> building persistent image $IMG (one-time, ~12 min)"
  docker rm -f slam_build >/dev/null 2>&1 || true
  docker run --name slam_build --gpus all --ipc=host -v "$REPO":/w -w /w \
      "$BASE" bash /w/p4_slam/_slam_build.sh
  echo ">> committing -> $IMG"
  docker commit slam_build "$IMG" >/dev/null
  docker rm -f slam_build >/dev/null 2>&1 || true
else
  echo ">> reusing cached image $IMG"
fi

echo ">> RUN main.py --no-viz on $DS (no-calib)"
docker run --rm --gpus all --ipc=host -v "$REPO":/w -w /w/p4_slam/MASt3R-SLAM \
    -e QT_QPA_PLATFORM=offscreen \
    "$IMG" \
    python main.py --no-viz --dataset "$DS" --config config/base.yaml --save-as "$TAG"

echo ">> outputs:"
ls -la "$REPO"/p4_slam/MASt3R-SLAM/logs/ 2>/dev/null || true
echo ">> DONE"
