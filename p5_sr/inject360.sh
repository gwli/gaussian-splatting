#!/bin/bash
# task3 stage-5: write 360 spherical metadata so YouTube/VR players treat the
# mp4 as monoscopic equirect. Uses Google's spatial-media injector (vendored on
# first run from github). Pure-python, runs on host python3.
#
#   inject360.sh <in.mp4> <out.mp4>     # adds SphericalVideo (equirect mono)
set -e
ROOT=/raid/git/gaussian-splatting
IN="$1"; OUT="$2"
abspath() { case "$1" in /*) echo "$1";; *) echo "$ROOT/$1";; esac; }
IN="$(abspath "$IN")"; OUT="$(abspath "$OUT")"; mkdir -p "$(dirname "$OUT")"
SM="$ROOT/p5_sr/spatial_media"

if [ ! -d "$SM/spatialmedia" ]; then
  echo ">> [inject360] vendoring google/spatial-media"
  mkdir -p "$SM"
  curl -fsSL https://github.com/google/spatial-media/archive/refs/heads/master.tar.gz \
    | tar -xz -C "$SM" --strip-components=1 spatial-media-master/spatialmedia
fi

if [ ! -s "$IN" ]; then echo ">> [inject360] ERROR: input $IN missing/empty"; exit 1; fi

echo ">> [inject360] injecting equirect mono metadata"
PYTHONPATH="$SM" python3 -m spatialmedia -i "$IN" "$OUT" 2>&1 | tail -4

if [ ! -s "$OUT" ]; then echo ">> [inject360] ERROR: output not written"; exit 1; fi
# verify the box landed (injector prints "Spherical: true" when present)
if PYTHONPATH="$SM" python3 -m spatialmedia "$OUT" 2>&1 | grep -qiE "spherical:[[:space:]]*true|GSpherical|equirectangular"; then
  echo ">> [inject360] DONE: $OUT (spherical 360 metadata present)"
else
  echo ">> [inject360] WARNING: could not confirm spherical metadata in $OUT"; exit 1
fi
