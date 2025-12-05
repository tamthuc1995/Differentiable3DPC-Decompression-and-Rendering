#include "gather_voxel_params.h"
#include "auxiliary.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

namespace VOXEL_PARAMS_GATHER {

    __global__ void gather_geo_params_cuda(
        const int num_visible,
        const int64_t* __restrict__ visible_vox,
        const int64_t* __restrict__ vox2corners,
        const float* __restrict__ geometry_params,
        float* __restrict__ gathered_geo_params)
    {
        auto tid = cg::this_grid().thread_rank();
        if (tid >= num_visible)
            return;

        // Load from global mem
        const int idx = visible_vox[tid];
        int corner_id[8];
        for(int i=0; i<8; ++i)
            corner_id[i] = vox2corners[idx * 8 + i];

        float params[8];
        for(int i=0; i<8; ++i)
            params[i] = geometry_params[corner_id[i]];

        // Write to voxel geo param
        for(int i=0; i<8; ++i)
            gathered_geo_params[idx * 8 + i] = params[i];
    }

    __global__ void gather_geo_params_bw_cuda(
        const int num_visible,
        const int64_t* __restrict__ visible_vox,
        const int64_t* __restrict__ vox2corners,
        const float* __restrict__ dL_dgeo_params,
        float* __restrict__ dL_dcorners_pts)
    {
        auto tid = cg::this_grid().thread_rank();
        if (tid >= num_visible)
            return;

        // Load from global mem
        const int idx = visible_vox[tid];
        int corner_id[8];
        for(int i=0; i<8; ++i)
            corner_id[i] = vox2corners[idx * 8 + i];

        float dL_dparams[8];
        for(int i=0; i<8; ++i)
            dL_dparams[i] = dL_dgeo_params[idx * 8 + i];

        // Write to voxel geo param
        for(int i=0; i<8; ++i)
            atomicAdd(dL_dcorners_pts + corner_id[i], dL_dparams[i]);
    }

    __global__ void gather_color_params_cuda(
        const int num_visible,
        const int64_t* __restrict__ visible_vox,
        const float3* __restrict__ color_params,
        float3* __restrict__ gathered_color_params)
    {
        auto tid = cg::this_grid().thread_rank();
        if (tid >= num_visible)
            return;

        // Load from global mem
        const int idx = visible_vox[tid];
        // Color now only have RGB so this is a very simple gather operation

        // Write to voxel geo param
        gathered_color_params[idx] = color_params[idx];
    }


    __global__ void gather_color_params_bw_cuda(
        const int num_visible,
        const int64_t* __restrict__ visible_vox,
        const float* __restrict__ dL_drgb_params,
        float* __restrict__ dL_dcolor_params)
    {
        auto tid = cg::this_grid().thread_rank();
        if (tid >= num_visible)
            return;

        // Load from global mem
        const int idx = visible_vox[tid];
        
        // Color now only have RGB so this is a very simple attributing operation

        // Write to voxel geo param
        atomicAdd(dL_dcolor_params + 3*idx + 0, dL_drgb_params[3*idx+0]);
        atomicAdd(dL_dcolor_params + 3*idx + 1, dL_drgb_params[3*idx+1]);
        atomicAdd(dL_dcolor_params + 3*idx + 2, dL_drgb_params[3*idx+2]);
    }

    // Python interface for gather corners value into each voxel.
    torch::Tensor gather_geo_params(
        const torch::Tensor& visible_vox,
        const torch::Tensor& vox2corners,
        const torch::Tensor& geometry_params)
    {
        const int num_vox = vox2corners.size(0);
        const int num_visible = visible_vox.size(0);
        torch::Tensor gathered_geo_params = torch::empty({num_vox, 8}, geometry_params.options());

        if (num_visible > 0)
            gather_geo_params_cuda <<<(num_visible + 255) / 256, 256>>> (
                num_visible,
                visible_vox.contiguous().data_ptr<int64_t>(),
                vox2corners.contiguous().data_ptr<int64_t>(),
                geometry_params.contiguous().data_ptr<float>(),
                gathered_geo_params.contiguous().data_ptr<float>()
            );

        return gathered_geo_params;
    }

    torch::Tensor gather_geo_params_bw(
        const torch::Tensor& visible_vox,
        const torch::Tensor& vox2corners,
        const int num_corners,
        const torch::Tensor& dL_dgeo_params)
    {
        const int num_vox = vox2corners.size(0);
        const int num_visible = visible_vox.size(0);
        torch::Tensor dL_dcorners_pts = torch::zeros({num_corners, 1}, dL_dgeo_params.options());

        if (num_visible > 0)
            gather_geo_params_bw_cuda <<<(num_visible + 255) / 256, 256>>> (
                num_visible,
                visible_vox.contiguous().data_ptr<int64_t>(),
                vox2corners.contiguous().data_ptr<int64_t>(),
                dL_dgeo_params.contiguous().data_ptr<float>(),
                dL_dcorners_pts.contiguous().data_ptr<float>()
            );

        return dL_dcorners_pts;
    }

    torch::Tensor gather_color_params(
        const torch::Tensor& visible_vox,
        const torch::Tensor& color_params)
    {
        const int num_vox = color_params.size(0);
        const int num_color_params = color_params.size(1);
        const int num_visible = visible_vox.size(0);
        torch::Tensor gathered_color_params = torch::zeros({num_vox, num_color_params}, color_params.options());

        if (num_color_params != 3)
            AT_ERROR("Only support n_dim=3 now.");

        if (num_visible > 0)
            gather_color_params_cuda <<<(num_visible + 255) / 256, 256>>> (
                num_visible,
                visible_vox.contiguous().data_ptr<int64_t>(),
                (float3*)(color_params.contiguous().data_ptr<float>()),
                (float3*)(gathered_color_params.contiguous().data_ptr<float>())
            );

        return gathered_color_params;
    }

    torch::Tensor gather_color_params_bw(
        const torch::Tensor& visible_vox,
        const int num_voxels,
        const torch::Tensor& dL_drgb_params)
    {
        const int num_visible = visible_vox.size(0);
        const int num_color_params = dL_drgb_params.size(1);
        torch::Tensor dL_dcolor_params = torch::zeros({num_voxels, num_color_params}, dL_drgb_params.options());

        if (dL_drgb_params.size(0) != num_voxels)
            AT_ERROR("Mismatch voxel size in dL_drgb_params and input num_voxels");

        if (num_color_params != 3)
            AT_ERROR("Only support n_dim=3");

        if (num_visible > 0)
            gather_color_params_bw_cuda <<<(num_visible + 255) / 256, 256>>> (
                num_visible,
                visible_vox.contiguous().data_ptr<int64_t>(),
                dL_drgb_params.contiguous().data_ptr<float>(),
                dL_dcolor_params.contiguous().data_ptr<float>()
            );
        return dL_dcolor_params;
    }
}
