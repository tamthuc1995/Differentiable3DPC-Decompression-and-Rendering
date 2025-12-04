#ifndef VOXEL_PARAMS_GATHER_H_INCLUDED
#define VOXEL_PARAMS_GATHER_H_INCLUDED

#include <torch/extension.h>

namespace VOXEL_PARAMS_GATHER {

// Python interface for gather grid points value into each voxel.
torch::Tensor gather_geo_params(
    const torch::Tensor& visible_vox,
    const torch::Tensor& vox2corners,
    const torch::Tensor& geometry_params);

torch::Tensor gather_geo_params_bw(
    const torch::Tensor& visible_vox,
    const torch::Tensor& vox2corners,
    const int num_corners,
    const torch::Tensor& dL_dgeo_params);

torch::Tensor gather_color_params(
    const torch::Tensor& visible_vox,
    const torch::Tensor& color_params);

torch::Tensor gather_color_params_bw(
    const torch::Tensor& visible_vox,
    const int num_voxels,
    const torch::Tensor& dL_drgb_params);
}

#endif
