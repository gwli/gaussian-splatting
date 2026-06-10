# License notice — p3_pano/

⚠️ **Licensing is mixed in this subtree. Read before redistributing.**

`p3_pano/diff_gaussian_rasterization_pano/` contains a CUDA rasterizer whose
core (`cuda_rasterizer/`, `rasterize_points.cu/.h`) is **derived from
[OmniGS](https://github.com/liquorleaf/OmniGS), which is licensed GPLv3**
(itself a derivative of 3D Gaussian Splatting). The Python scaffold
(`ext.cpp`, `setup.py`, `__init__.py`) we added is glue around that GPLv3 core,
so the **`diff_gaussian_rasterization_pano` package as a whole is effectively
GPLv3**.

By contrast, the rest of this repository (the INRIA 3D Gaussian Splatting code)
is under the **INRIA non-commercial research license** (see top-level
`LICENSE.md`), which is *not* GPL-compatible for redistribution purposes.

### Practical guidance
- The GPLv3-derived code is **isolated under `p3_pano/`** and is only imported
  by the optional direct-panorama path (`train_pano.py`, `pano_render.py`).
  The core perspective/VGGT pipeline does **not** depend on it.
- For **research / internal use** on this fork, the mix is fine.
- For **redistribution or any commercial use**, do **not** ship
  `p3_pano/diff_gaussian_rasterization_pano/` together with the INRIA-licensed
  code under a single license. Either (a) keep the GPLv3 component in a
  separate repo and depend on it, or (b) re-implement the equirect projection
  from scratch under a compatible license.

### Provenance
- Equirect CUDA core: OmniGS © 2024 Li & Huang et al. (GPLv3).
- Vendored copies (`p2_vggt/vggt`, `p3_pano/OmniGS`, glm) are git-ignored and
  fetched at setup; see the respective `README.md` files.
