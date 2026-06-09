from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

here = os.path.dirname(os.path.abspath(__file__))
glm_inc = os.path.join(here, "third_party/glm/")

setup(
    name="diff_gaussian_rasterization_pano",
    packages=['diff_gaussian_rasterization_pano'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization_pano._C",
            sources=[
                "cuda_rasterizer/rasterizer_impl.cu",
                "cuda_rasterizer/forward.cu",
                "cuda_rasterizer/backward.cu",
                "rasterize_points.cu",
                "ext.cpp",
            ],
            extra_compile_args={
                # NOTE: do NOT force-include <cstdint> globally — on CUDA 12.x it
                # exposes C23 _Float32 and the M_*f32 macros trigger an nvcc
                # codegen ICE ("unsupported float variant"). cstdint is included
                # per-file in the sources instead, and the f32 macros were
                # replaced with plain float literals.
                "nvcc": ["-I" + glm_inc, "-std=c++17"],
            },
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)
