# P1.3 — Direct equirectangular (panorama) 3DGS

Goal: train 3DGS **directly on equirectangular panoramas**, skipping the
perspective-crop Stage 2 (14× less data, no seam/overlap issues).

Standard `diff-gaussian-rasterization` assumes a PINHOLE projection, so this
requires a rasterizer that projects 3D Gaussians onto the spherical (lon/lat)
image plane. We ported the CUDA core from **OmniGS** (GPLv3, Li & Huang et al.
2024 — itself a derivative of 3DGS) into a pip-installable PyTorch extension.

## `diff_gaussian_rasterization_pano/` — the ported rasterizer

A self-contained pip extension exposing the same API as the original 2023
diff-gaussian-rasterization, plus a `camera_type` field
(`CAMERA_PINHOLE=1`, `CAMERA_LONLAT=3`).

- CUDA core (`cuda_rasterizer/`): OmniGS forward/backward/impl, including the
  `LonlatRasterizer`, `preprocessLonlatCUDA`, `computeCov2DLonlat`,
  `point3ToLonlatScreen` equirect kernels.
- `rasterize_points.cu` dispatches camera_type 1/3.
- `ext.cpp`, `setup.py`, `diff_gaussian_rasterization_pano/__init__.py` are
  newly written to match OmniGS's 6-tuple forward / 8-tuple backward signature.

### Porting fixes (already applied)
1. `M_1_PIf32` / `M_2_PIf32` GNU `_Float32` math macros → plain float literals.
   Their `f32` suffix expands to C23 `std::float32_t`, which triggers an nvcc
   codegen ICE: `"unsupported float variant!"`.
2. Added `#include <cstdint>` per source (NOT via `-include`, which re-triggers
   the float-variant ICE) for C++17/CUDA 12.6.
3. `__init__.py`: explicit `None` checks (not `tensor or ...`, which is ambiguous).

## Setup

```bash
cd p3_pano
git clone https://github.com/liquorleaf/OmniGS.git           # source of CUDA core (already vendored here)
cp -r ../submodules/diff-gaussian-rasterization/third_party \
      diff_gaussian_rasterization_pano/third_party             # glm headers
# build (inside the nvcr pytorch container)
pip install --no-deps -e diff_gaussian_rasterization_pano
```

## Status

- ✅ **Rasterizer builds & imports** (`IMPORT_OK lonlat=3`).
- ✅ **Forward LONLAT render validated** — `pano_render.py` renders a coherent
  equirectangular image from a trained `.ply` (157k gaussians, 118k visible,
  signature horizon band + pole stretch). Confirms the ported forward + the
  Python autograd wrapper work end-to-end.

- ✅ **P1.3b pose derivation** (`make_pano_dataset.py`): per-pano camera =
  `R_wp = R_off(yaw,pitch) · R_v` (R_off from the crop filename, R_v from VGGT),
  center = mean of the pano's crops. 90/90 panos recovered.
- ✅ **P1.3c direct-pano training** (`train_pano.py`, reuses GaussianModel +
  densification, LONLAT render): converges (loss 0.11→0.04).

### Results — scene_023 (held-out test, every-8th-pano holdout)

| | Perspective (VGGT crops) | **Direct-pano** |
|---|---|---|
| training images | 1260 crops | **90 panoramas (14× fewer)** |
| held-out PSNR | 17.05 | **19.12** |
| rasterizer | pinhole | equirect (ported OmniGS LONLAT) |
| SfM | VGGT (per-crop) | VGGT (270 curated crops) → per-pano poses |

Direct-panorama training matches/beats the perspective pipeline on held-out
PSNR while training on **14× fewer images** and rendering the full 360° per
view — validating the P1.3 thesis end to end.

### Pipeline (one scene, from .insv)
```bash
bash p3_pano/prep_pano.sh scene_023 VID_20260326_073432_023.insv 90   # stitch(kept)+crop+VGGT
# curate crops to <=300 for VGGT memory, then:
python p3_pano/make_pano_dataset.py <scene_pano_dir> pano_cams.json    # per-pano poses
python p3_pano/train_pano.py pano_cams.json <out> 7000 1024            # direct equirect train + eval
```

### Remaining / future
- Re-stitch is CPU-bound (~30 min/scene for v360); a GPU stitch (P0.4) would help.
- `prep_pano.sh` should cap crops fed to VGGT at ≤300 automatically (done
  manually here).
- Batch the other scenes; tune conf_thres / iters per scene.

## `pano_render.py`

Standalone sanity tool: `python pano_render.py <ply> <out.png> [width]` —
renders an equirectangular view with `camera_type=LONLAT`.
