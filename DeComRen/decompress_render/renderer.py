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



def rasterize_voxels(
        camera_settings: CameraSettings,
        render_settings: RenderSettings,
        vox_roots: torch.Tensor,
        vox_length: torch.Tensor,
        vox_fn,
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
    
    return (n_duplicates, geomBuffer, outTemp)
