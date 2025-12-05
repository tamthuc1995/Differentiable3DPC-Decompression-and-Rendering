import torch
from . import _C

from typing import NamedTuple

class CameraSettings(NamedTuple):
    image_width: int
    image_height: int
    tanfovx: float
    tanfovy: float
    cx: float
    cy: float
    w2c_matrix: torch.Tensor
    c2w_matrix: torch.Tensor


class RenderSettings(NamedTuple):
    num_sample_per_vox: int
    bg_color: float = 0
    near: float = 0.1
    need_depth: bool = False
    track_max_w: bool = False
    debug: bool = False


def gather_geometry_and_color(
        visible_vox:  torch.Tensor,
        vox2corners: torch.Tensor,
        geometry_params: torch.Tensor,
        color_params: torch.Tensor,    
    ):


    # Gather the density values at the eight corners of each voxel.
    geos = GatherGeoParams.apply(
        visible_vox,
        vox2corners,
        geometry_params,
    )

    # Gather/Compute voxel colors
    rgbs = GatherColorParams.appeny(
        visible_vox,
        color_params
    )

    # Pack everything
    vox_params = {
        'geos': geos,
        'rgbs': rgbs,
        'subdiv_p': None, # Dummy param to record subdivision priority
    }
    if vox_params['subdiv_p'] is None:
        vox_params['subdiv_p'] = torch.ones([geometry_params.shape[0], 1], device=geometry_params.device)

    return vox_params


def rasterize_voxels_main(
        camera_settings: CameraSettings,
        render_settings: RenderSettings,
        vox_roots: torch.Tensor,
        vox_length: torch.Tensor,
        vox2corners: torch.Tensor,
        vox_params_geo: torch.Tensor,
        vox_params_color: torch.Tensor,
    ):

    # Checking
    if not isinstance(camera_settings, CameraSettings):
        raise Exception("Expect RasterSettings as first argument.")
    if render_settings.num_sample_per_vox > _C.MAX_NUM_SAMPLE or render_settings.num_sample_per_vox < 1:
        raise Exception(f"num_sample_per_vox should be in range [1, {_C.MAX_NUM_SAMPLE}].")


    N = vox_roots.shape[0]
    device = vox_roots.device
    ##
    if len(vox_roots.shape) != 2 or vox_roots.shape[1] != 3:
        raise Exception("Expect vox_centers in shape [N, 3].")
    ##
    if camera_settings.w2c_matrix.device != device or \
            camera_settings.c2w_matrix.device != device or \
            vox_roots.device != device:
        raise Exception("Device mismatch.")


    # # Preprocess octree
    n_duplicates, voxelDataBuffer = _C.rasterize_preprocess(
        # Cam setting
        camera_settings.image_width,
        camera_settings.image_height,
        camera_settings.tanfovx,
        camera_settings.tanfovy,
        camera_settings.cx,
        camera_settings.cy,
        camera_settings.w2c_matrix,
        camera_settings.c2w_matrix,

        # Render setting
        render_settings.near,

        # Geometry data
        vox_roots,
        vox_length,
        vox2corners,
        vox_params_geo,
        vox_params_color,

        # Debug flag
        render_settings.debug,
    )
    torch.cuda.synchronize()


    # Gather 3D scene paramerters
    visible_vox_idx = torch.where(n_duplicates > 0)[0]
    # Forward voxel parameters

    vox_params = gather_geometry_and_color(visible_vox_idx, vox2corners, vox_params_geo, vox_params_color)
    geos = vox_params['geos']
    rgbs = vox_params['rgbs']
    subdiv_p = vox_params['subdiv_p']


    # Some voxel parameters checking
    if geos.shape != (N, 8):
        raise Exception(f"Expect geos in ({N}, 8) but got", geos.shape)
    if rgbs.shape[0] != N:
        raise Exception(f"Expect rgbs in ({N}, 3) but got", rgbs.shape)
    if subdiv_p.shape[0] != N:
        raise Exception(f"Expect subdiv_p in ({N}, 1) but got", subdiv_p.shape)

    if geos.device != device:
        raise Exception("Device mismatch: geos.")
    if rgbs.device != device:
        raise Exception("Device mismatch: rgbs.")
    if subdiv_p.device != device:
        raise Exception("Device mismatch: subdiv_p.")


    result_rendered = VoxelRasterizer.apply(
        camera_settings,
        render_settings,
        voxelDataBuffer,

        # Geometry data
        vox_roots,
        vox_length,

        # 3d scene parameters
        geos,
        rgbs,
        subdiv_p
    )

    return result_rendered


class VoxelRasterizer(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        camera_settings,
        raster_settings,
        voxelDataBuffer,

        # Geometry data
        vox_roots,
        vox_length,

        geos,
        rgbs,
        subdiv_p
    ):


        args = (
            camera_settings.image_width,
            camera_settings.image_height,
            camera_settings.tanfovx,
            camera_settings.tanfovy,
            camera_settings.cx,
            camera_settings.cy,
            camera_settings.w2c_matrix,
            camera_settings.c2w_matrix,

            raster_settings.n_samp_per_vox,
            raster_settings.bg_color,
            raster_settings.need_depth,
            raster_settings.track_max_w,

            vox_roots,
            vox_length,
            geos,
            rgbs,
            voxelDataBuffer,

            raster_settings.debug,
        )

        num_vox_duplicated, voxels2raysBuffer, raysBuffer, out_color, out_T, max_w = _C.voxels_rasterizing(*args)

        # Keep relevant tensors for backward
        ctx.camera_settings    = camera_settings
        ctx.raster_settings    = raster_settings
        ctx.num_vox_duplicated = num_vox_duplicated

        ctx.save_for_backward(
            vox_roots, vox_length,
            geos, rgbs,
            voxelDataBuffer, voxels2raysBuffer, raysBuffer, out_T)
        ctx.mark_non_differentiable(max_w)
        return out_color, out_T, max_w

    @staticmethod
    def backward(ctx, dL_dout_color):
        # Restore necessary values from context
        camera_settings = ctx.camera_settings
        raster_settings = ctx.raster_settings
        num_vox_duplicated = ctx.num_vox_duplicated
        (
            vox_roots, vox_length, 
            geos, rgbs, voxelDataBuffer, 
            voxels2raysBuffer, raysBuffer, out_T
        ) = ctx.saved_tensors

        
        args = (
            num_vox_duplicated,

            camera_settings.image_width,
            camera_settings.image_height,
            camera_settings.tanfovx,
            camera_settings.tanfovy,
            camera_settings.cx,
            camera_settings.cy,
            camera_settings.w2c_matrix,
            camera_settings.c2w_matrix,

            raster_settings.n_samp_per_vox,
            raster_settings.bg_color,

            vox_roots,
            vox_length,
            geos,
            rgbs,

            voxelDataBuffer,
            voxels2raysBuffer,
            raysBuffer,
            out_T,

            dL_dout_color,

            raster_settings.debug,
        )

        dL_dgeos, dL_drgbs, subdiv_p_bw = _C.voxels_backward_rasterizing(*args)

        grads = (
            None, # => camera_settings
            None, # => raster_settings
            None, # => voxelDataBuffer
            None, # => vox_roots
            None, # => vox_length
            dL_dgeos, # => geos
            dL_drgbs, # => rgbs
            subdiv_p_bw, # => subdivision priority
        )

        return grads



class GatherGeoParams(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        visible_vox,
        vox2corners,
        geometry_params,
    ):
        assert len(vox2corners.shape) == 2 and vox2corners.shape[1] == 8
        assert len(visible_vox.shape) == 1
        assert geometry_params.shape[0] == geometry_params.numel()

        gathered_geo_params = _C.gather_geo_params(visible_vox, vox2corners, geometry_params)

        ctx.num_corners = geometry_params.shape[0]
        ctx.save_for_backward(vox2corners, visible_vox)
        return gathered_geo_params

    @staticmethod
    def backward(ctx, dL_dgeo_params):
        # Restore necessary values from context
        num_corners = ctx.num_corners
        vox2corners, visible_vox = ctx.saved_tensors

        dL_dgeo_corners= _C.gather_geo_params_bw(visible_vox, vox2corners, num_corners, dL_dgeo_params)

        return None, None, dL_dgeo_corners



class GatherColorParams(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        visible_vox,
        color_params,
    ):
        assert len(visible_vox.shape) == 1
        assert color_params.shape[0] == color_params.numel() // 3

        gathered_rgb_params = _C.gather_color_params(visible_vox, color_params)

        ctx.num_voxels = color_params.shape[0]
        ctx.save_for_backward(visible_vox)
        return gathered_rgb_params

    @staticmethod
    def backward(ctx, dL_drgb_params):
        # Restore necessary values from context
        num_voxels = ctx.num_voxels
        visible_vox = ctx.saved_tensors

        dL_dcolor_params = _C.gather_color_params_bw(visible_vox, num_voxels, dL_drgb_params)

        return None, None, dL_dcolor_params
