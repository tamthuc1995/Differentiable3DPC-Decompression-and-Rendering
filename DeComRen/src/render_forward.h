
#ifndef VOXEL_RASTERIZER_FORWARD_H_INCLUDED
#define VOXEL_RASTERIZER_FORWARD_H_INCLUDED

#include <torch/extension.h>

namespace VOXEL_RASTERIZER {

// Interface for python to run forward rasterization.
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
voxels_rasterizing(
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
    const bool need_depth,
    const bool track_max_w,

    const torch::Tensor& vox_roots,
    const torch::Tensor& vox_length,
    const torch::Tensor& geos,
    const torch::Tensor& rgbs,
    const torch::Tensor& voxelDataBuffer,

    const bool debug);
}

#endif