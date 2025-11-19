
#ifndef RASTERIZE_PREPROCESS_H_INCLUDED
#define RASTERIZE_PREPROCESS_H_INCLUDED

#include <torch/extension.h>

namespace PREPROCESS {

    // Interface for python to find the voxel to render and compute some init values.
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    rasterize_preprocess(
        // Cam setting
        const int image_width, const int image_height,
        const float tan_fovx, const float tan_fovy,
        const float cx, const float cy,
        const torch::Tensor& w2c_matrix,
        const torch::Tensor& c2w_matrix,
        // render setting
        const float near,
        // Geo 
        const torch::Tensor& vox_roots,
        const torch::Tensor& vox_length,
        // Debug flag
        const bool debug
    );
}

#endif