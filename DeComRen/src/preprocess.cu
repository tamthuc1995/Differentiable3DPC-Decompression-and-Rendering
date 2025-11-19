#include "preprocess.h"
#include "raster_data.h"
#include "auxiliary.h"

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cooperative_groups.h>
namespace cg = cooperative_groups;



namespace PREPROCESS {

    // CUDA kernel 
    __global__ void preprocessCUDA(
        const int N,
        const int W, const int H,
        const float tan_fovx, const float tan_fovy,
        const float fx, const float fy,
        const float cx, const float cy,
        const float* __restrict__ w2c_matrix,
        const float* __restrict__ c2w_matrix,
        const float near,

        const float3* __restrict__ vox_roots,
        const float* __restrict__ vox_length,

        int* __restrict__ output_ndup_per_vox,
        float2* __restrict__ output_img_bbox,
        uint32_t* __restrict__ ndup_per_vox,
        uint2* __restrict__ bboxes,
        uint32_t* __restrict__ cam_quadrant_bitsets,

        const dim3 parallel_2d_grid)
    {   
        // Standard kernel id
        auto idx = cg::this_grid().thread_rank();
        if (idx >= N){
            return;
        } 

        // Init result value
        output_ndup_per_vox[idx] = 0;
        ndup_per_vox[idx] = 0;

        // 
        output_img_bbox[idx*2 + 0] = {1e9f, 1e9f};
        output_img_bbox[idx*2 + 1] = {-1e9f, -1e9f};


        // Load from global memory.
        const float3 vox_coner_000 = vox_roots[idx];
        const float vox_len = vox_length[0]; // VOXELs HAVE SAME LENGTH at a given scale.
        const float3 cam_org = get_cam_position(c2w_matrix);
        float w2c[12];
        for (int i = 0; i < 12; i++)
            w2c[i] = w2c_matrix[i];

        // clipping in a sphere R=Near 
        const float3 dir_diff = make_float3(
            vox_coner_000.x + 0.5f * vox_len - cam_org.x,
            vox_coner_000.y + 0.5f * vox_len - cam_org.y,
            vox_coner_000.z + 0.5f * vox_len - cam_org.z
        );
        if (dot(dir_diff, dir_diff) < near * near)
            return;

        // for (int i=0; i<8; ++i)
        // {   

        //     float3 world_corner = make_float3(
        //         vox_coner_000.x + (float)(((i&4)>>2)) * vox_len,
        //         vox_coner_000.y + (float)(((i&2)>>1)) * vox_len,
        //         vox_coner_000.z + (float)(((i&1)   )) * vox_len
        //     );
        //     float3 cam_corner = transform_3x4(w2c, world_corner);
        //     float2 cam_corner_scaled = make_float2(
        //         cam_corner.x / cam_corner.z,
        //         cam_corner.y / cam_corner.z
        //     );

        //     printf(" + i = %d \n", i);
        //     printf(" + world_corner = (%f, %f, %f)\n", world_corner.x, world_corner.y, world_corner.z);
        //     printf(" + cam_corner = (%f, %f, %f)\n", cam_corner.x, cam_corner.y, cam_corner.z);
        //     printf(" + cam_corner_scaled = (%f, %f)", cam_corner_scaled.x, cam_corner_scaled.y);
        //     output_project_coner[idx*8 + i] = cam_corner;
        // }


        // // Check the eight voxel corners and do:
        // // 1. Estimate the bounded bbox of the projected voxel for duplication.
        // // 2. Compute possible camera direction for duplication: 8 possible quadrants
        // // Number of duplication is number of 2D blocks times possible camera direction 
        uint32_t quadrant_bitset = 0;
        float2 cam_bbox_min = {1e9f, 1e9f};
        float2 cam_bbox_max = {-1e9f, -1e9f};
        for (int i=0; i<8; ++i)
        {
            float3 world_corner = make_float3(
                vox_coner_000.x + (float)(((i&4)>>2)) * vox_len,
                vox_coner_000.y + (float)(((i&2)>>1)) * vox_len,
                vox_coner_000.z + (float)(((i&1)   )) * vox_len
            );
            float3 cam_corner = transform_3x4(w2c, world_corner);

            if (cam_corner.z < near)
                continue;

            float2 cam_corner_scaled = make_float2(
                cam_corner.x / cam_corner.z,
                cam_corner.y / cam_corner.z
            );
            int quadrant_id = compute_cam_to_point_quadrant_id(world_corner, cam_org);
            
            // Accumulate bounded bbox
            cam_bbox_min = make_float2(
                min(cam_bbox_min.x, cam_corner_scaled.x), 
                min(cam_bbox_min.y, cam_corner_scaled.y)
            );
            cam_bbox_max = make_float2(
                max(cam_bbox_max.x, cam_corner_scaled.x), 
                max(cam_bbox_max.y, cam_corner_scaled.y)
            );
            quadrant_bitset |= (1 << quadrant_id);
            
        }

        // Get pixels index of bbox corners
        // K = [
        //    |f_x |0   |c_x |\\
        //    |0   |f_y |c_y |\\
        //    |0   |0   |1   |
        // ]
        float cx_h = cx - 0.5f;
        float cy_h = cy - 0.5f;
        float2 img_bbox_min = {
            max(fx * cam_bbox_min.x + cx_h, 0.0f),
            max(fy * cam_bbox_min.y + cy_h, 0.0f)
        };
        float2 img_bbox_max = {
            min(fx * cam_bbox_max.x + cx_h, (float)W),
            min(fy * cam_bbox_max.y + cy_h, (float)H)
        };
        if (img_bbox_min.x > img_bbox_max.x || img_bbox_min.y > img_bbox_max.y)
            return; // Bbox outside image plane.

        // Squeeze bbox info into 2 uint 32 bits. Image resolution can not pass 65536 x 65536
        const uint2 img_bbox = {
            (((uint)lrintf(img_bbox_min.x)) << 16) | ((uint)lrintf(img_bbox_min.y)),
            (((uint)lrintf(img_bbox_max.x)) << 16) | ((uint)lrintf(img_bbox_max.y))
        };

        
        output_img_bbox[idx*2 + 0] = img_bbox_min;
        output_img_bbox[idx*2 + 1] = img_bbox_max;
    
        // Compute tile range.
        // Make sure that tile possition is in range (0, parallel_2d_grid.x) x (0, parallel_2d_grid.y)
        uint2 min_block2d, max_block2d;
        min_block2d = {
            (uint32_t)max(0, min(((int)parallel_2d_grid.x)-1, (int)(img_bbox_min.x / BLOCK2D_X))),
            (uint32_t)max(0, min(((int)parallel_2d_grid.y)-1, (int)(img_bbox_min.y / BLOCK2D_Y)))
        };
        max_block2d = {
            (uint32_t)max(0, min(((int)parallel_2d_grid.x)-1, (int)(img_bbox_max.x / BLOCK2D_X))),
            (uint32_t)max(0, min(((int)parallel_2d_grid.y)-1, (int)(img_bbox_max.y / BLOCK2D_Y)))
        };
        int num_block2d_touched = (1 + max_block2d.y - min_block2d.y) * (1 + max_block2d.x - min_block2d.x);
        // if (num_block2d_touched <= 0)
        // {
        //     // TODO: remove sanity check.
        //     printf("num_block2d_touched <= 0 !???");
        //     __trap();
        // }

        // Write back the results.
        const int quadrant_touched = __popc(quadrant_bitset);
        output_ndup_per_vox[idx] = num_block2d_touched * quadrant_touched;
        ndup_per_vox[idx] = num_block2d_touched * quadrant_touched;
        bboxes[idx] = img_bbox;
        cam_quadrant_bitsets[idx] = quadrant_bitset;
    }

    
    // Interface for python to preprocess voxels and find intersected vox
    //  also doing some raster data preparation 
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    rasterize_preprocess(
        const int image_width, const int image_height,
        const float tan_fovx, const float tan_fovy,
        const float cx, const float cy,
        const torch::Tensor& w2c_matrix,
        const torch::Tensor& c2w_matrix,
        const float near,

        const torch::Tensor& vox_roots,
        const torch::Tensor& vox_length,

        const bool debug)
    {
        if (vox_roots.ndimension() != 2 || vox_roots.size(1) != 3)
            AT_ERROR("vox_roots must have dimensions (N, 3)");
        
        const int N = vox_roots.size(0);
        printf("N: %d \n", N);
        if (N == 0)
             AT_ERROR("Are you trying to render from zero voxels ??");

        auto t_opt_byte = torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA);
        auto t_opt_int32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
        auto t_opt_float32 = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);

        torch::Tensor voxelDataBuffer = torch::empty({0}, t_opt_byte);
        torch::Tensor output_ndup_per_vox = torch::full({N}, 0, t_opt_int32);
        torch::Tensor output_temp = torch::full({N*2*2}, 0, t_opt_float32);

        // Allocate Voxel data for rendering
        size_t chunk_size = RASTER_DATA::size_required<RASTER_DATA::VoxelData>(N);
        printf("chunk_size: %d \n", chunk_size);
        voxelDataBuffer.resize_({(long long)chunk_size});
        char* chunkptr = reinterpret_cast<char*>(voxelDataBuffer.contiguous().data_ptr());
        RASTER_DATA::VoxelData voxData = RASTER_DATA::VoxelData::sizeAloc(chunkptr, N);
        
        // Parallel rendering block grid size
        dim3 parallel_2d_grid((image_width + BLOCK2D_X - 1) / BLOCK2D_X, (image_height + BLOCK2D_Y - 1) / BLOCK2D_Y, 1);
        // Get Camera Intrinsic Matrix parameters
        // K = [
        //    |f_x |0   |c_x |\\
        //    |0   |f_y |c_y |\\
        //    |0   |0   |1   |
        // ]
        const float fx = 0.5f * image_width / tan_fovx;
        const float fy = 0.5f * image_height / tan_fovy;
        
        // Lanching CUDA kernel
        preprocessCUDA <<<(N + 63) / 64, 64>>> (
            N,
            image_width, image_height,
            tan_fovx, tan_fovy,
            fx, fy, cx, cy,
            w2c_matrix.contiguous().data_ptr<float>(),
            c2w_matrix.contiguous().data_ptr<float>(),
            near,

            (float3*)(vox_roots.contiguous().data_ptr<float>()),
            vox_length.contiguous().data_ptr<float>(),

            output_ndup_per_vox.contiguous().data_ptr<int>(),
            (float2*)(output_temp.contiguous().data_ptr<float>()),
            voxData.ndup_per_vox,
            voxData.bboxes,
            voxData.cam_quadrant_bitsets,

            parallel_2d_grid);
        CHECK_CUDA(debug);

        return std::make_tuple(output_ndup_per_vox, voxelDataBuffer, output_temp);
        // const auto tensor_opt = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
        // torch::Tensor A = torch::empty({10, 3}, tensor_opt);
        // torch::Tensor B = torch::empty({10, 3}, tensor_opt);

    }
}