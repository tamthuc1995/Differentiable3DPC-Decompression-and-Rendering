
#include "render_forward.h"
#include "raster_data.h"
#include "auxiliary.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

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

    const bool debug)
{
    if (vox_roots.ndimension() != 2 || vox_roots.size(1) != 3)
        AT_ERROR("vox_roots must have dimensions (num_voxels, 3)");
    if (rgbs.ndimension() != 2 || rgbs.size(1) != 3)
        AT_ERROR("rgbs should be either (num_voxels, 3)");
    if (vox_roots.size(0) != rgbs.size(0))
        AT_ERROR("num voxels mismatch between location and color");

    const int N = vox_roots.size(0);
    const int H = image_height;
    const int W = image_width;

    auto float_opts = torch::TensorOptions(torch::kFloat32).device(torch::kCUDA);
    auto byte_opts = torch::TensorOptions(torch::kByte).device(torch::kCUDA);

    torch::Tensor out_color = torch::full({3, H, W}, 0.f, float_opts);
    torch::Tensor out_T = torch::full({1, H, W}, 0.f, float_opts);
    torch::Tensor max_w = track_max_w ? torch::full({N, 1}, 0.f, float_opts) : torch::empty({0});

    torch::Tensor voxels2raysBuffer = torch::empty({0}, byte_opts);
    torch::Tensor raysBuffer = torch::empty({0}, byte_opts);
    // Neat functional trick to pass the decision to later process
    std::function<char*(size_t)> voxels2raysFunc = RASTER_DATA::getFuncResizeForTensor(voxels2raysBuffer);
    std::function<char*(size_t)> raysFunc = RASTER_DATA::getFuncResizeForTensor(raysBuffer);

    float* max_w_pointer = nullptr;
    if (track_max_w)
        max_w_pointer = max_w.contiguous().data_ptr<float>();

    int num_vox_duplicated = 0;

    // if(P != 0)
    //     rendered = rasterize_voxels_procedure(
    //         reinterpret_cast<char*>(voxelDataBuffer.contiguous().data_ptr()),
    //         binningFunc,
    //         imgFunc,
    //         P,
    //         n_samp_per_vox,

    //         W, H,
    //         tan_fovx, tan_fovy,
    //         cx, cy,
    //         w2c_matrix.contiguous().data_ptr<float>(),
    //         c2w_matrix.contiguous().data_ptr<float>(),
    //         bg_color,
    //         need_depth,
    //         need_distortion,
    //         need_normal,

    //         octree_paths.contiguous().data_ptr<int64_t>(),
    //         vox_centers.contiguous().data_ptr<float>(),
    //         vox_lengths.contiguous().data_ptr<float>(),
    //         geos.contiguous().data_ptr<float>(),
    //         rgbs.contiguous().data_ptr<float>(),

    //         out_color.contiguous().data_ptr<float>(),
    //         out_depth.contiguous().data_ptr<float>(),
    //         out_normal.contiguous().data_ptr<float>(),
    //         out_T.contiguous().data_ptr<float>(),
    //         max_w_pointer,

    //         debug);

    return std::make_tuple(num_vox_duplicated, voxels2raysBuffer, raysBuffer, out_color, out_T, max_w);
}


}