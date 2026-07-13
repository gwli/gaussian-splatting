#!/usr/bin/env python3
"""Per-scene prep after raw extraction (runs in pytorch container):
1) stitch3 1140 ERPs from 3840 dual fisheye (reuses rig023.npz — same rig)
2) downscale down-stream to 1920 for SfM
3) delete the 3840 dirs (reproducible from .insv)
usage: prep_scene.py <S>   e.g. 021
"""
import sys, os, subprocess, shutil
from PIL import Image
from joblib import Parallel, delayed

S = sys.argv[1]
SC = f"/w/data/8kpano/scenes"
RAW = f"{SC}/fish{S}d"
OUT = f"{SC}/scene_{S}rigd_pano/panoramas"
SFM = f"{SC}/fish{S}1140/images"
os.makedirs(OUT, exist_ok=True); os.makedirs(SFM, exist_ok=True)

n = len([f for f in os.listdir(f"{RAW}/images") if f.endswith(".jpg")])
nu = len([f for f in os.listdir(f"{RAW}/images_up") if f.endswith(".jpg")])
assert n == nu and n >= 1100, f"frame counts {n}/{nu}"
print(f"[prep {S}] {n} frame pairs", flush=True)

env = dict(os.environ, FISH_DOWN=f"{RAW}/images", FISH_UP=f"{RAW}/images_up", FISH_W="3840")
subprocess.run(["python", "/w/p6_unisharp/ft/stitch3.py", OUT, str(n)], env=env, check=True)

def one(k):
    Image.open(f"{RAW}/images/f_{k:04d}.jpg").resize((1920, 1920), Image.LANCZOS).save(
        f"{SFM}/f_{k:04d}.jpg", quality=95)
Parallel(n_jobs=12)(delayed(one)(k) for k in range(1, n+1))
print(f"[prep {S}] downscaled {n} -> 1920", flush=True)

shutil.rmtree(f"{RAW}/images"); shutil.rmtree(f"{RAW}/images_up")
print(f"[prep {S}] 3840 raw dirs removed; PREP_DONE", flush=True)
