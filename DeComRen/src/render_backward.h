
#ifndef VOXEL_RASTERIZER_BACKWARD_H_INCLUDED
#define VOXEL_RASTERIZER_BACKWARD_H_INCLUDED

#include <torch/extension.h>

namespace VOXEL_RASTERIZER_BACKWARD
{

// Interface for python to run backward pass of voxel rasterization.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
voxels_backward_rasterizing(
    const int R,
    
    const int image_width,
    const int image_height,
    const float tan_fovx,
    const float tan_fovy,
    const float cx,
    const float cy,
    const torch::Tensor& w2c_matrix,
    const torch::Tensor& c2w_matrix,

    
    const int n_samp_per_vox,
    const float bg_color,

    const torch::Tensor& vox_roots,
    const torch::Tensor& vox_length,
    const torch::Tensor& geos,
    const torch::Tensor& rgbs,

    const torch::Tensor& voxelDataBuffer,
    const torch::Tensor& binningVox2RayBuffer,
    const torch::Tensor& rayGroupsBuffer,
    const torch::Tensor& out_T,

    const torch::Tensor& dL_dout_color,

    const bool debug);

}


#endif