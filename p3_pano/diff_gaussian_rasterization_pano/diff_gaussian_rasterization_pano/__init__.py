#
# Equirectangular (LONLAT) Gaussian rasterizer — Python autograd wrapper.
# CUDA core derived from OmniGS (GPLv3, Li & Huang et al. 2024), itself a
# derivative of 3D Gaussian Splatting (INRIA). API mirrors the original 2023
# diff-gaussian-rasterization (6-tuple forward, no antialiasing/invdepth) plus
# a `camera_type` field: 1 = PINHOLE, 3 = LONLAT (equirectangular).
#
from typing import NamedTuple
import torch
import torch.nn as nn
from . import _C

# Camera model ids (match OmniGS camera.h CameraModelType)
CAMERA_PINHOLE = 1
CAMERA_LONLAT = 3


def cpu_deep_copy_tuple(input_tuple):
    return tuple(x.cpu().clone() if isinstance(x, torch.Tensor) else x for x in input_tuple)


def rasterize_gaussians(means3D, means2D, sh, colors_precomp, opacities,
                        scales, rotations, cov3Ds_precomp, raster_settings):
    return _RasterizeGaussians.apply(
        means3D, means2D, sh, colors_precomp, opacities,
        scales, rotations, cov3Ds_precomp, raster_settings,
    )


class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means3D, means2D, sh, colors_precomp, opacities,
                scales, rotations, cov3Ds_precomp, raster_settings):
        args = (
            raster_settings.bg,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.camera_type,
            raster_settings.render_depth,
        )
        num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = \
            _C.rasterize_gaussians(*args)

        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations,
                              cov3Ds_precomp, radii, sh, geomBuffer,
                              binningBuffer, imgBuffer)
        return color, radii

    @staticmethod
    def backward(ctx, grad_out_color, _grad_radii):
        num_rendered = ctx.num_rendered
        rs = ctx.raster_settings
        (colors_precomp, means3D, scales, rotations, cov3Ds_precomp,
         radii, sh, geomBuffer, binningBuffer, imgBuffer) = ctx.saved_tensors

        args = (
            rs.bg,
            means3D,
            radii,
            colors_precomp,
            scales,
            rotations,
            rs.scale_modifier,
            cov3Ds_precomp,
            rs.viewmatrix,
            rs.projmatrix,
            rs.tanfovx,
            rs.tanfovy,
            grad_out_color,
            sh,
            rs.sh_degree,
            rs.campos,
            geomBuffer,
            num_rendered,
            binningBuffer,
            imgBuffer,
            rs.camera_type,
        )
        (grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D,
         grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations) = \
            _C.rasterize_gaussians_backward(*args)

        return (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )


class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    bg: torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    sh_degree: int
    campos: torch.Tensor
    prefiltered: bool
    camera_type: int = CAMERA_LONLAT
    render_depth: bool = False


class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        with torch.no_grad():
            rs = self.raster_settings
            return _C.mark_visible(positions, rs.viewmatrix, rs.projmatrix, rs.camera_type)

    def forward(self, means3D, means2D, opacities, shs=None, colors_precomp=None,
                scales=None, rotations=None, cov3D_precomp=None):
        rs = self.raster_settings
        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Provide exactly one of shs or colors_precomp.')
        if ((scales is None or rotations is None) and cov3D_precomp is None) or \
           ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Provide exactly one of (scales, rotations) or cov3D_precomp.')

        if shs is None: shs = torch.Tensor([])
        if colors_precomp is None: colors_precomp = torch.Tensor([])
        if scales is None: scales = torch.Tensor([])
        if rotations is None: rotations = torch.Tensor([])
        if cov3D_precomp is None: cov3D_precomp = torch.Tensor([])

        return rasterize_gaussians(means3D, means2D, shs, colors_precomp, opacities,
                                   scales, rotations, cov3D_precomp, rs)
