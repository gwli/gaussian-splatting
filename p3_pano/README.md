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

### Remaining (P1.3b/c)
- Per-panorama camera poses: each pano's 14 crops share a camera center;
  recover pano pose from a crop's VGGT extrinsic by removing the known
  yaw/pitch offset (encoded in the crop filename). Point cloud reused from the
  existing `scenes/<S>/vggt/sparse/0`.
- Equirect `Camera` + panorama dataloader (no FoV/projection; just W2V + campos).
- `train_pano.py` wiring the pano rasterizer; train one scene directly on
  panoramas; compare data volume / time / held-out metrics vs the
  perspective-crop VGGT pipeline.

## `pano_render.py`

Standalone sanity tool: `python pano_render.py <ply> <out.png> [width]` —
renders an equirectangular view with `camera_type=LONLAT`.
