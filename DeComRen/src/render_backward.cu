#include "render_backward.h"
#include "raster_data.h"
#include "auxiliary.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

namespace VOXEL_RASTERIZER_BACKWARD {


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
    const torch::Tensor& voxels2raysBuffer,
    const torch::Tensor& raysBuffer,
    const torch::Tensor& out_T,

    const torch::Tensor& dL_dout_color,

    const bool debug)
{
    if (vox_roots.ndimension() != 2 || vox_roots.size(1) != 3)
        AT_ERROR("vox_roots must have dimensions (num_points, 3)");

    const int N = vox_roots.size(0);

    if (N == 0)
    {
        torch::Tensor dL_dgeos = torch::empty({0});
        torch::Tensor dL_drgbs = torch::empty({0});
        torch::Tensor subdiv_p_bw = torch::empty({0});
        return std::make_tuple(dL_dgeos, dL_drgbs, subdiv_p_bw);
    }

    torch::Tensor dL_dvox = torch::zeros({N, geos.size(1)+3+1}, vox_roots.options());
    dim3 tile_grid((image_width + BLOCK2D_X - 1) / BLOCK2D_X, (image_height + BLOCK2D_Y - 1) / BLOCK2D_Y, 1);
    dim3 block(BLOCK2D_X, BLOCK2D_Y, 1);

    // Retrive raster state from pytorch tensor
    char* voxdataB_ptr = reinterpret_cast<char*>(voxelDataBuffer.contiguous().data_ptr());
    char* vox2rayB_ptr = reinterpret_cast<char*>(voxels2raysBuffer.contiguous().data_ptr());
    char* raysB_ptr = reinterpret_cast<char*>(raysBuffer.contiguous().data_ptr());
    RASTER_DATA::VoxelData voxelData = RASTER_DATA::VoxelData::sizeAloc(voxdataB_ptr, N);
    RASTER_DATA::BindingVoxel2RayData binningVox2RayData = RASTER_DATA::BindingVoxel2RayData::sizeAloc(vox2rayB_ptr, R);
    RASTER_DATA::GroupRaysData raysData = RASTER_DATA::GroupRaysData::sizeAloc(
        raysB_ptr, 
        image_width * image_height, 
        tile_grid.x * tile_grid.y
    );

    // // Compute loss gradients w.r.t. surface property and voxel color.
    // render(
    //     tile_grid, block,
    //     imgState.ranges,
    //     binningState.vox_list,
    //     n_samp_per_vox,
    //     image_width, image_height,
    //     tan_fovx, tan_fovy,
    //     cx, cy,
    //     c2w_matrix.contiguous().data_ptr<float>(),
    //     bg_color,

    //     geomState.bboxes,
    //     (float3*)(vox_centers.contiguous().data_ptr<float>()),
    //     vox_lengths.contiguous().data_ptr<float>(),
    //     geos.contiguous().data_ptr<float>(),
    //     (float3*)(rgbs.contiguous().data_ptr<float>()),

    //     out_T.contiguous().data_ptr<float>(),
    //     imgState.tile_last,
    //     imgState.n_contrib,

    //     dL_dout_color.contiguous().data_ptr<float>(),
    //     dL_dout_depth.contiguous().data_ptr<float>(),
    //     dL_dout_normal.contiguous().data_ptr<float>(),
    //     dL_dout_T.contiguous().data_ptr<float>(),

    //     lambda_R_concen,
    //     gt_color.contiguous().data_ptr<float>(),
    //     lambda_ascending,
    //     lambda_dist,
    //     need_depth,
    //     need_normal,
    //     out_D.contiguous().data_ptr<float>(),
    //     out_N.contiguous().data_ptr<float>(),

    //     dL_dvox.contiguous().data_ptr<float>());
    // CHECK_CUDA(debug);

    std::vector<torch::Tensor> gradient_lst = dL_dvox.split({geos.size(1), 3, 1}, 1);
    torch::Tensor dL_dgeos = gradient_lst[0].contiguous();
    torch::Tensor dL_drgbs = gradient_lst[1].contiguous();
    torch::Tensor subdiv_p_bw = gradient_lst[2].contiguous();

    return std::make_tuple(dL_dgeos, dL_drgbs, subdiv_p_bw);
}

}