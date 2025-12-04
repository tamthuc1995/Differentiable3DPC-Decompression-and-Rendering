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



    # Gather/Compute voxel colors



    # Pack everything
    vox_params = {
        'geos': None,
        'rgbs': None,
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
    n_duplicates, geomBuffer, outTemp = _C.rasterize_preprocess(
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
        
        # Debug flag
        render_settings.debug,
    )
    torch.cuda.synchronize()


    # Gather 3D scene paramerters
    in_frusts_idx = torch.where(n_duplicates > 0)[0]
    # Forward voxel parameters
    vox_params = gather_geometry_and_color(in_frusts_idx, vox_params_geo, vox_params_color)
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


    return (n_duplicates, geomBuffer, outTemp)
    # result_rendered = _RasterizeVoxels.apply(
    #     camera_settings,
    #     render_settings,
    #     geomBuffer,

    #     # Geometry data
    #     vox_roots,
    #     vox_length,
        
    #     # 3d scene parameters
    #     geos,
    #     rgbs,
    #     subdiv_p,
    # )
    

    # return result_rendered

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


# class _RasterizeVoxels(torch.autograd.Function):
#     @staticmethod
#     def forward(
#         ctx,
#         raster_settings,
#         geomBuffer,
#         octree_paths,
#         vox_centers,
#         vox_lengths,
#         geos,
#         rgbs,
#         subdiv_p,
#     ):

#         need_distortion = raster_settings.lambda_dist > 0

#         args = (
#             raster_settings.n_samp_per_vox,
#             raster_settings.image_width,
#             raster_settings.image_height,
#             raster_settings.tanfovx,
#             raster_settings.tanfovy,
#             raster_settings.cx,
#             raster_settings.cy,
#             raster_settings.w2c_matrix,
#             raster_settings.c2w_matrix,
#             raster_settings.bg_color,
#             raster_settings.need_depth,
#             need_distortion,
#             raster_settings.need_normal,
#             raster_settings.track_max_w,

#             octree_paths,
#             vox_centers,
#             vox_lengths,
#             geos,
#             rgbs,

#             geomBuffer,

#             raster_settings.debug,
#         )

#         num_rendered, binningBuffer, imgBuffer, out_color, out_depth, out_normal, out_T, max_w = _C.rasterize_voxels(*args)

#         # Keep relevant tensors for backward
#         ctx.raster_settings = raster_settings
#         ctx.num_rendered = num_rendered
#         ctx.save_for_backward(
#             octree_paths, vox_centers, vox_lengths,
#             geos, rgbs,
#             geomBuffer, binningBuffer, imgBuffer, out_T, out_depth, out_normal)
#         ctx.mark_non_differentiable(max_w)
#         return out_color, out_depth, out_normal, out_T, max_w

#     @staticmethod
#     def backward(ctx, dL_dout_color):
#         # Restore necessary values from context
#         raster_settings = ctx.raster_settings
#         num_rendered = ctx.num_rendered
#         octree_paths, vox_centers, vox_lengths, \
#             geos, rgbs, \
#             geomBuffer, binningBuffer, imgBuffer, out_T, out_depth, out_normal = ctx.saved_tensors

#         args = (
#             num_rendered,
#             raster_settings.n_samp_per_vox,
#             raster_settings.image_width,
#             raster_settings.image_height,
#             raster_settings.tanfovx,
#             raster_settings.tanfovy,
#             raster_settings.cx,
#             raster_settings.cy,
#             raster_settings.w2c_matrix,
#             raster_settings.c2w_matrix,
#             raster_settings.bg_color,

#             octree_paths,
#             vox_centers,
#             vox_lengths,
#             geos,
#             rgbs,

#             geomBuffer,
#             binningBuffer,
#             imgBuffer,
#             out_T,

#             dL_dout_color,

#             raster_settings.lambda_R_concen,
#             raster_settings.gt_color,
#             raster_settings.lambda_ascending,
#             raster_settings.lambda_dist,
#             raster_settings.need_depth,
#             raster_settings.need_normal,
#             out_depth,
#             out_normal,

#             raster_settings.debug,
#         )

#         dL_dgeos, dL_drgbs, subdiv_p_bw = _C.rasterize_voxels_backward(*args)

#         grads = (
#             None, # => raster_settings
#             None, # => geomBuffer
#             None, # => octree_paths
#             None, # => vox_centers
#             None, # => vox_lengths
#             dL_dgeos, # => geos
#             dL_drgbs, # => rgbs
#             subdiv_p_bw, # => subdivision priority
#         )

#         return grads