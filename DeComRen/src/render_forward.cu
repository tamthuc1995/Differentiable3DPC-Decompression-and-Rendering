
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

    // CUDA sparse voxel rendering.
    template <bool need_depth, bool track_max_w, int n_sample>
    __global__ void __launch_bounds__(BLOCK2D_X * BLOCK2D_Y)
    renderCUDA(
        const uint2* __restrict__ first2last,
        const uint32_t* __restrict__ vox_list,

        const int W, 
        const int H,
        const float tan_fovx, 
        const float tan_fovy,
        const float cx,
        const float cy,
        const float* __restrict__ c2w_matrix,
        
        const float bg_color,

        const uint2* __restrict__ bboxes,
        const float3* __restrict__ vox_roots,
        const float* __restrict__ vox_length,
        const float* __restrict__ geos,
        const float3* __restrict__ rgbs,

        uint32_t* __restrict__ tile_last_vox,
        uint32_t* __restrict__ actual_scanned_vox,

        float* __restrict__ out_color,
        float* __restrict__ out_depth,
        float* __restrict__ out_T,
        float* __restrict__ max_w)
    {
        // Identify current tile and associated min/max pixel range.
        auto block = cg::this_thread_block();
        uint32_t horizontal_blocks = (W + BLOCK2D_X - 1) / BLOCK2D_X;
        int thread_id = block.thread_rank();
        int tile_id = block.group_index().y * horizontal_blocks + block.group_index().x;
        uint2 pix_min = { block.group_index().x * BLOCK2D_X, block.group_index().y * BLOCK2D_X };
        uint2 pix_max = { min(pix_min.x + BLOCK2D_X, W), min(pix_min.y + BLOCK2D_Y , H) };

        uint2 pix;
        uint32_t pix_id;
        float2 pixf;
        if (BLOCK2D_X % 8 == 0 && BLOCK2D_Y % 4 == 0)
        {
            // Pack the warp threads into a 4x8 macro blocks.
            // It could reduce idle warp threads as the voxels to render
            // are more coherent in 4x8 than 2x16 rectangle.
            int macro_x_num = BLOCK2D_X / 8;
            int macro_id = thread_id / 32;
            int macro_xid = macro_id % macro_x_num;
            int macro_yid = macro_id / macro_x_num;
            int micro_id = thread_id % 32;
            int micro_xid = micro_id % 8;
            int micro_yid = micro_id / 8;
            pix = { pix_min.x + macro_xid * 8 + micro_xid, pix_min.y + macro_yid * 4 + micro_yid};
            pix_id = W * pix.y + pix.x;
            pixf = { (float)pix.x, (float)pix.y };
        }
        else
        {
            pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
            pix_id = W * pix.y + pix.x;
            pixf = { (float)pix.x, (float)pix.y };
        }

        // Compute camera info.
        const float3 cam_rd = {
            (pixf.x + 0.5f - cx) * 2.f * tan_fovx / (float)W,
            (pixf.y + 0.5f - cy) * 2.f * tan_fovy / (float)H,
            1.f
        };
        const float rd_norm = sqrtf(dot(cam_rd, cam_rd));
        const float rd_norm_inv = 1.f / rd_norm;
        const float3 ro = last_col_3x4(c2w_matrix);
        const float3 rd_raw = rotate_3x4(c2w_matrix, cam_rd);
        const float3 rd = rd_raw * rd_norm_inv;
        const float3 rd_inv = {1.f/ rd.x, 1.f / rd.y, 1.f / rd.z};
        const uint32_t pix_quad_id = compute_direction_quadrant_id(rd);

        // Check if this thread is associated with a valid pixel or outside.
        bool inside = (pix.x < W) && (pix.y < H);
        // Done threads can help with fetching, but don't rasterize
        bool done = !inside;

        // Load start/end range of IDs to process in vox2rays_data.
        uint2 vox_range = first2last[tile_id];
        const int rounds = ((vox_range.y - vox_range.x + BLOCK2D_N - 1) / BLOCK2D_N);
        int toDo = vox_range.y - vox_range.x;

        // Init the last non-occluded range index of the tile.
        if (thread_id == 0)
            tile_last_vox[tile_id] = vox_range.x;


        // Allocate storage for batches of collectively fetched data.
        __shared__ int collected_vox_id[BLOCK2D_N];
        __shared__ int collected_quad_id[BLOCK2D_N];
        __shared__ uint2 collected_bbox[BLOCK2D_N];
        __shared__ float3 collected_vox_r[BLOCK2D_N];
        __shared__ float collected_vox_l[BLOCK2D_N];
        __shared__ float collected_geo_params[BLOCK2D_N * 8];
        __shared__ float3 collected_rgb[BLOCK2D_N];


        // Initialize helper variables.
        float T = 1.f;
        uint32_t contributor = 0;
        uint32_t last_contributor = 0;
        float3 C = {0.f, 0.f, 0.f};
        float3 N = {0.f, 0.f, 0.f};
        float D = 0.f;
        int D_med_vox_id = -1;
        float D_med_T;
        float D_med = 0.f;
        float Ddist = 0.f;
        int j_lst[BLOCK2D_N];
        // Iterate over batches until all done or range is complete.
        for (int i = 0; i < rounds; i++, toDo -= BLOCK2D_N)
        {
            // End if entire block votes that it is done rasterizing.
            int num_done = __syncthreads_count(done);
            if (num_done == BLOCK2D_N)
                break;

            
            // Collectively fetch batch of voxel data from global to shared.
            int progress = i * BLOCK2D_N + thread_id;
            if (vox_range.x + progress < vox_range.y)
            {
                uint32_t vox_val = vox_list[vox_range.x + progress];
                uint32_t vox_id = (vox_val << 3) >> 3;
                uint32_t quad_id = vox_val >> 29;
                collected_vox_id[thread_id] = vox_id;
                collected_quad_id[thread_id] = quad_id;
                collected_bbox[thread_id] = bboxes[vox_id];
                collected_vox_r[thread_id] = vox_roots[vox_id];
                collected_vox_l[thread_id] = vox_length[vox_id];
                for (int k=0; k<8; ++k)
                    collected_geo_params[thread_id*8 + k] = geos[vox_id*8 + k];
                collected_rgb[thread_id] = rgbs[vox_id];
            }
            block.sync();

            // Iterate over current batch.
            const int end_j = min(BLOCK2D_N, toDo);
            int j_lst_top = -1;
            for (int j = 0; !done && j < end_j; j++)
            {
                if (!pix_in_bbox(pix, collected_bbox[j]) || pix_quad_id != collected_quad_id[j])
                    continue;

                // Compute ray aabb intersection
                const float3 vox_r = collected_vox_r[j];
                const float vox_l = collected_vox_l[j];
                const float2 ab = ray_aabb(vox_r, vox_l, ro, rd_inv);
                const float a = ab.x;
                const float b = ab.y;
                if (a > b)
                    continue;  // Skip if no intersection.

                j_lst_top += 1;
                j_lst[j_lst_top] = j;
            }


            int contributor_inc = 0;
            for (int jj = 0; !done && jj <= j_lst_top; jj++)
            {
                int j = j_lst[jj];
                const int vox_id = collected_vox_id[j];

                // Keep track of current position in range.
                contributor_inc = j + 1;

                // Compute ray aabb intersection
                const float3 vox_r = collected_vox_r[j];
                const float vox_l = collected_vox_l[j];
                const float2 ab = ray_aabb(vox_r, vox_l, ro, rd_inv);
                const float a = ab.x;
                const float b = ab.y;

                float geo_params[8];
                for (int k=0; k<8; ++k)
                    geo_params[k] = collected_geo_params[j*8 + k];


                // Compute volume density
                float vol_int = 0.f;
                float interp_w[8];
                float local_alphas[n_sample];

                // Quadrature integral from trilinear sampling.
                float vox_l_inv = 1.f / vox_l;
                const float step_sz = (b - a) * (1.f / n_sample);
                const float3 step = step_sz * rd;
                float3 pt = ro + (a + 0.5f * step_sz) * rd;
                float3 qt = (pt - vox_r) * vox_l_inv; // scale and shift the voxel to unit cube (0, 1)^3
                const float3 qt_step = step * vox_l_inv;

                #pragma unroll
                for (int k=0; k<n_sample; k++, qt=qt+qt_step)
                {   
                    // trilinear interppolate weight
                    float wx[2] = {1.f - qt.x, qt.x};
                    float wy[2] = {1.f - qt.y, qt.y};
                    float wz[2] = {1.f - qt.z, qt.z};
                    interp_w[0] = wx[0] * wy[0] * wz[0];
                    interp_w[1] = wx[0] * wy[0] * wz[1];
                    interp_w[2] = wx[0] * wy[1] * wz[0];
                    interp_w[3] = wx[0] * wy[1] * wz[1];
                    interp_w[4] = wx[1] * wy[0] * wz[0];
                    interp_w[5] = wx[1] * wy[0] * wz[1];
                    interp_w[6] = wx[1] * wy[1] * wz[0];
                    interp_w[7] = wx[1] * wy[1] * wz[1];

                    float d = 0.f;
                    for (int iii=0; iii<8; ++iii)
                        d += geo_params[iii] * interp_w[iii];

                    const float local_vol_int = STEP_SZ_SCALE * step_sz * exp_linear_11(d);
                    vol_int += local_vol_int;

                    if (need_depth && n_sample > 1)
                        local_alphas[k] = min(MAX_ALPHA, 1.f - expf(-local_vol_int));
                }

                // Compute alpha from volume integral.
                float alpha = min(MAX_ALPHA, 1.f - expf(-vol_int));
                if (alpha < MIN_ALPHA)
                    continue;

                // Accumulate to the pixel.
                float pt_w = T * alpha;
                C = C + pt_w * collected_rgb[j];

                if (need_depth)
                {
                    // Mean depth
                    float dval;
                    if (n_sample == 3)
                    {
                        float step_sz = 0.3333333f * (b - a);
                        float a0 = local_alphas[0], a1 = local_alphas[1], a2 = local_alphas[2];
                        float t0 = a + 0.5f * step_sz;
                        float t1 = a + 1.5f * step_sz;
                        float t2 = a + 2.5f * step_sz;
                        dval = a0*t0 + (1.f-a0)*a1*t1 + (1.f-a0)*(1.f-a1)*a2*t2;
                    }
                    else if (n_sample == 2)
                    {
                        float step_sz = 0.5f * (b - a);
                        float a0 = local_alphas[0], a1 = local_alphas[1];
                        float t0 = a + 0.5f * step_sz;
                        float t1 = a + 1.5f * step_sz;
                        dval = a0*t0 + (1.f-a0)*a1*t1;
                    }
                    else
                    {
                        dval = alpha * 0.5f * (a + b);
                    }
                    D = D + T * dval;

                    // Median depth
                    if (T > 0.5f)
                    {
                        D_med_vox_id = vox_id;
                        D_med_T = T;
                    }
                }

                T *= (1.f - alpha);
                done |= (T < EARLY_STOP_T);

                // Keep track of last range entry to update this pixel.
                last_contributor = contributor + contributor_inc;

                // Keep track of the maxiumum importance weight of each voxel.
                if (track_max_w)
                    atomicMax(((int*)max_w) + vox_id, *((int*)(&pt_w)));

            }
            contributor += done ? contributor_inc : end_j;
        }


        // Extract depth median base on provided D_med_vox_id
        if (need_depth && inside && D_med_vox_id != -1)
        {
            // Finest sampling of median depth
            const int n_samp_dmed = 16;

            float3 vox_r = vox_roots[D_med_vox_id];
            float vox_l = vox_length[D_med_vox_id];
            float geo_params[8];
            for (int k=0; k<8; ++k)
                geo_params[k] = geos[D_med_vox_id*8 + k];
            const float2 ab = ray_aabb(vox_r, vox_l, ro, rd_inv);
            const float a = ab.x;
            const float b = ab.y;

            float vox_l_inv = 1.f / vox_l;
            const float step_sz = (b - a) * (1.f / n_samp_dmed);
            const float3 step = step_sz * rd;
            float3 pt = ro + (a + 0.5f * step_sz) * rd;
            float3 qt = (pt - vox_r) * vox_l_inv;
            const float3 qt_step = step * vox_l_inv;

            D_med = a - 0.5f * step_sz;
            for (int k=0; k<n_samp_dmed && D_med_T > 0.5f; k++, qt=qt+qt_step)
            {
                D_med += step_sz;

                float interp_w[8];
                
                // trilinear interppolate weight
                float wx[2] = {1.f - qt.x, qt.x};
                float wy[2] = {1.f - qt.y, qt.y};
                float wz[2] = {1.f - qt.z, qt.z};
                interp_w[0] = wx[0] * wy[0] * wz[0];
                interp_w[1] = wx[0] * wy[0] * wz[1];
                interp_w[2] = wx[0] * wy[1] * wz[0];
                interp_w[3] = wx[0] * wy[1] * wz[1];
                interp_w[4] = wx[1] * wy[0] * wz[0];
                interp_w[5] = wx[1] * wy[0] * wz[1];
                interp_w[6] = wx[1] * wy[1] * wz[0];
                interp_w[7] = wx[1] * wy[1] * wz[1];

                float d = 0.f;
                for (int iii=0; iii<8; ++iii)
                    d += geo_params[iii] * interp_w[iii];

                const float vol_int = STEP_SZ_SCALE * step_sz * exp_linear_11(d);

                D_med_T *= expf(-vol_int);
            }
        }

        // All threads that treat valid pixel write out their final
        // rendering data to the frame and auxiliary buffers.
        if (inside)
        {
            actual_scanned_vox[pix_id] = last_contributor;
            out_color[0 * H * W + pix_id] = C.x + T * bg_color;
            out_color[1 * H * W + pix_id] = C.y + T * bg_color;
            out_color[2 * H * W + pix_id] = C.z + T * bg_color;
            out_T[pix_id] = T;  // Equal to (1 - alpha).
            if (need_depth)
            {
                out_depth[H * W * 0 + pix_id] = D * rd_norm_inv;
                out_depth[H * W * 2 + pix_id] = D_med * rd_norm_inv;
            }
            atomicMax(tile_last_vox + tile_id, vox_range.x + last_contributor);
        }
    }
    #ifndef FwRendFunc
    // Dirty trick. The argument name must be aligned with FORWARD::render.
    #define FwRendFunc(...) \
        ( \
            (need_depth && track_max_w) ? \
                renderCUDA<true,  true, __VA_ARGS__> :\
            (need_depth && !track_max_w) ?\
                renderCUDA<true, false, __VA_ARGS__> :\
            (!need_depth && track_max_w) ?\
                renderCUDA<false, true, __VA_ARGS__> :\
                renderCUDA<false, false, __VA_ARGS__> \
        )
    #endif


    // Lowest-level C interface for launching the CUDA.
    void render(
        const dim3 tile_grid,
        const dim3 block,
        const uint2* first2last,
        const uint32_t* vox_list,

        const int W, 
        const int H,
        const float tan_fovx, 
        const float tan_fovy,
        const float cx,
        const float cy,
        const float* c2w_matrix,
        
        const int num_sample_per_vox,
        const float bg_color,
        const bool need_depth,

        const uint2* bboxes,
        const float3* vox_roots,
        const float* vox_length,
        const float* geos,
        const float3* rgbs,

        uint32_t* tile_last_vox,
        uint32_t* actual_scanned_vox,

        float* out_color,
        float* out_depth,
        float* out_T,
        float* max_w)
    {
        const bool track_max_w = (max_w != nullptr);

        const auto kernel_func =
            (num_sample_per_vox == 3) ?
                FwRendFunc(3) :
            (num_sample_per_vox == 2) ?
                FwRendFunc(2) :
                FwRendFunc(1) ;

        kernel_func <<<tile_grid, block>>> (
            first2last,
            vox_list,

            W, 
            H,
            tan_fovx, 
            tan_fovy,
            cx, 
            cy,
            c2w_matrix,
            
            bg_color,

            bboxes,
            vox_roots,
            vox_length,
            geos,
            rgbs,

            tile_last_vox,
            actual_scanned_vox,

            out_color,
            out_depth,
            out_T,
            max_w);
    }



    // Duplicate each voxel by #tiles x #cam_quadrant it touches.
    __global__ void duplicateVoxelsWithKeys(
        int N,
        const int64_t* morton_codes,
        const uint2* bboxes,
        const uint32_t* cam_quadrant_bitsets,
        const uint32_t* ndup_per_vox,
        const uint32_t* ndup_per_vox_csum,
        uint64_t* vox_list_keys_unsorted,
        uint32_t* vox_list_unsorted,
        dim3 tile_grid)
    {
        auto idx = cg::this_grid().thread_rank();
        if (idx >= N || ndup_per_vox[idx] == 0)
            return;

        // Find this voxel's array offset in buffer for writing the key/value.
        uint32_t off = (idx == 0) ? 0 : ndup_per_vox_csum[idx - 1];
        uint2 tile_min, tile_max;

        // Extract image block from (xmin, ymin) to (xmax, ymax) in (uint2) bboxes 
        uint32_t xmin = (bboxes[idx].x >> 16);
        uint32_t ymin = (bboxes[idx].x << 16 >> 16);
        uint32_t xmax = (bboxes[idx].y >> 16);
        uint32_t ymax = (bboxes[idx].y << 16 >> 16);
        tile_min = {
            (uint32_t)max(0, min(((int)tile_grid.x)-1, (int)(xmin / BLOCK2D_X))),
            (uint32_t)max(0, min(((int)tile_grid.y)-1, (int)(ymin / BLOCK2D_Y)))
        };
        tile_max = {
            (uint32_t)max(0, min(((int)tile_grid.x)-1, (int)(xmax / BLOCK2D_X))),
            (uint32_t)max(0, min(((int)tile_grid.y)-1, (int)(ymax / BLOCK2D_Y)))
        };

        // For each tile that the bounding rect overlaps, emit a key/value pair.
        // so the voxels are first sorted by tile and then by order_ranks.
        const uint64_t vox_morton_code = morton_codes[idx];
        uint32_t quadrant_bitsets = cam_quadrant_bitsets[idx];
        for (int quadrant_id = 0; quadrant_id < 8; quadrant_id++)
        {
            if ((quadrant_bitsets & (1 << quadrant_id)) == 0)
                continue;

            // Compute order_rank for the voxel in this quadrant.
            uint64_t vox_directional_code = compute_directional_code(vox_morton_code, quadrant_id);

            // Duplicate result to touched tiles.
            for (int y = tile_min.y; y <= tile_max.y; y++)
            {
                for (int x = tile_min.x; x <= tile_max.x; x++)
                {
                    uint64_t tile_id = y * tile_grid.x + x;
                    // The key bit structure is [  tile ID  |  order_rank  ],
                    vox_list_keys_unsorted[off] = (tile_id << NBITS_DIRECTIONAL_MORTON_CODE) | vox_directional_code;

                    // The value bit structure is [  quadrant ID  |  voxel ID  ].
                    vox_list_unsorted[off] = (((uint32_t)quadrant_id) << 29) | idx;
                    off++;
                }
            }
        }

        if (off != ndup_per_vox_csum[idx])
        {
            // TODO: remove sanity check.
            printf("Number of duplication mismatch !??? %d != %d \n", off, ndup_per_vox_csum[idx]);
            __trap();
        }
    }

    // Helper function to find the next-highest bit of the MSB on the CPU.
    uint32_t getHigherMsb(uint32_t n)
    {
        uint32_t msb = sizeof(n) * 4;
        uint32_t step = msb;
        while (step > 1)
        {
            step /= 2;
            if (n >> msb)
                msb += step;
            else
                msb -= step;
        }
        if (n >> msb)
            msb++;
        return msb;
    }

    // The sorted vox_list_keys is now as:
    //   [--sorted voxels for tile 1--  --sorted voxels for tile 2--  ...]
    // We want to identify the start/end index of each tile from this list.
    __global__ void identifyTileFirst2Last(int Ndup, uint64_t* vox_list_keys, uint2* first2last)
    {
        auto idx = cg::this_grid().thread_rank();
        if (idx >= Ndup)
            return;

        // Read tile ID from key. Update start/end of tile range if at limit.
        uint64_t key = vox_list_keys[idx];
        uint32_t currtile = key >> NBITS_DIRECTIONAL_MORTON_CODE;
        if (idx == 0)
            first2last[currtile].x = 0;
        else
        {
            uint32_t prevtile = vox_list_keys[idx - 1] >> NBITS_DIRECTIONAL_MORTON_CODE;
            if (currtile != prevtile)
            {
                first2last[prevtile].y = idx;
                first2last[currtile].x = idx;
            }
        }
        if (idx == Ndup - 1)
            first2last[currtile].y = Ndup;
    }

    // Mid-level C interface for the entire rasterization procedure.
    int rasterize_voxels_procedure(
        char* voxelDataBuffer,
        std::function<char* (size_t)> raysFunc,
        std::function<char* (size_t)> voxels2raysFunc,

        const int width,
        const int height,
        const float tan_fovx,
        const float tan_fovy,
        const float cx, 
        const float cy,
        const float* w2c_matrix,
        const float* c2w_matrix,

        const int N,
        const int num_sample_per_vox,
        const float bg_color,
        const bool need_depth,

        const int64_t* morton_codes,
        const float* vox_roots,
        const float* vox_length,
        const float* geos,
        const float* rgbs,

        float* out_color,
        float* out_depth,
        float* out_T,
        float* max_w,

        bool debug)
    {
        dim3 tile_grid((width + BLOCK2D_X - 1) / BLOCK2D_X, (height + BLOCK2D_Y - 1) / BLOCK2D_Y, 1);
        dim3 block(BLOCK2D_X, BLOCK2D_Y, 1);

        // Recover the preprocessing results.
        RASTER_DATA::VoxelData vox_data = RASTER_DATA::VoxelData::sizeAloc(voxelDataBuffer, N);

        // resize rays-based auxiliary buffers during training.
        size_t rays_data_size = RASTER_DATA::size_required<RASTER_DATA::GroupRaysData>(width * height, tile_grid.x * tile_grid.y);
        char* rays_dataptr = raysFunc(rays_data_size);
        RASTER_DATA::GroupRaysData rays_data = RASTER_DATA::GroupRaysData::sizeAloc(rays_dataptr, width * height, tile_grid.x * tile_grid.y);

        // Compute prefix sum over full list of the number of voxel duplications.
        cub::DeviceScan::InclusiveSum(
            vox_data.temp_csum_storage,
            vox_data.temp_csum_bytes,
            vox_data.ndup_per_vox,
            vox_data.ndup_per_vox_csum,
            N);
        CHECK_CUDA(debug);

        // Retrieve total number of voxels after duplication.
        int num_vox_duplicated;
        cudaMemcpy(
            &num_vox_duplicated,
            vox_data.ndup_per_vox_csum + N - 1,
            sizeof(int),
            cudaMemcpyDeviceToHost);
        CHECK_CUDA(debug);

        size_t vox2ray_data_size = RASTER_DATA::size_required<RASTER_DATA::BindingVoxel2RayData>(num_vox_duplicated);
        char* vox2ray_data_ptr = voxels2raysFunc(vox2ray_data_size);
        RASTER_DATA::BindingVoxel2RayData vox2ray_data = RASTER_DATA::BindingVoxel2RayData::sizeAloc(vox2ray_data_ptr, num_vox_duplicated);

        // The sorting key are created as earlier works:
        //     For each voxel to be rendered, produce adequate [ tile ID | rank ] key
        //     and the corresponding dublicated voxel [ quadrant ID | voxel ID ] to be sorted.
        duplicateVoxelsWithKeys <<<(N + 255) / 256, 256>>> (
            N,
            morton_codes,
            vox_data.bboxes,
            vox_data.cam_quadrant_bitsets,
            vox_data.ndup_per_vox,
            vox_data.ndup_per_vox_csum,
            vox2ray_data.vox_list_keys_unsorted,
            vox2ray_data.vox_list_unsorted,
            tile_grid);
        CHECK_CUDA(debug);

        int bit = getHigherMsb(tile_grid.x * tile_grid.y);

        // Sort complete list of (duplicated) ID by keys.
        cub::DeviceRadixSort::SortPairs(
            vox2ray_data.temp_sorting_storage,
            vox2ray_data.temp_storage_bytes,
            vox2ray_data.vox_list_keys_unsorted, vox2ray_data.vox_list_keys,
            vox2ray_data.vox_list_unsorted, vox2ray_data.vox_list,
            num_vox_duplicated, 0, NBITS_DIRECTIONAL_MORTON_CODE + bit);
        CHECK_CUDA(debug);

        cudaMemset(rays_data.first2last, 0, tile_grid.x * tile_grid.y * sizeof(uint2));
        CHECK_CUDA(debug);

        // Identify start and end of per-tile workloads in sorted list.
        if (num_vox_duplicated > 0)
        {
            identifyTileFirst2Last <<<(num_vox_duplicated + 255) / 256, 256>>> (
                num_vox_duplicated,
                vox2ray_data.vox_list_keys,
                rays_data.first2last);
            CHECK_CUDA(debug);
        }

        // Let each tile blend its range of voxels independently in parallel.
        render(
            tile_grid, 
            block,
            rays_data.first2last,
            vox2ray_data.vox_list,
            
            width,
            height,
            tan_fovx,
            tan_fovy,
            cx,
            cy,
            c2w_matrix,

            num_sample_per_vox,
            bg_color,
            need_depth,

            vox_data.bboxes,
            (float3*)vox_roots,
            vox_length,
            geos,
            (float3*)rgbs,

            rays_data.tile_last_vox,
            rays_data.actual_scanned_vox,

            out_color,
            out_depth,
            out_T,
            max_w
        );
        CHECK_CUDA(debug);

        return num_vox_duplicated;
    }


    // Interface for python to run forward rasterization.
    std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    voxels_rasterizing(
        const int image_width,
        const int image_height,
        const float tan_fovx,
        const float tan_fovy,
        const float cx,
        const float cy,
        const torch::Tensor& w2c_matrix,
        const torch::Tensor& c2w_matrix,

        const int num_sample_per_vox,
        const float bg_color,
        const bool need_depth,
        const bool track_max_w,

        const torch::Tensor& morton_codes,
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
        torch::Tensor out_depth = need_depth ? torch::full({3, H, W}, 0.f, float_opts) : torch::empty({0});
        torch::Tensor out_T = torch::full({1, H, W}, 0.f, float_opts);
        torch::Tensor max_w = track_max_w ? torch::full({N, 1}, 0.f, float_opts) : torch::empty({0});

        torch::Tensor voxels2raysBuffer = torch::empty({0}, byte_opts);
        torch::Tensor raysBuffer = torch::empty({0}, byte_opts);
        // Neat functional trick for tensor array to pass the memory allocation decision to later process
        std::function<char*(size_t)> voxels2raysFunc = RASTER_DATA::getFuncResizeForTensor(voxels2raysBuffer);
        std::function<char*(size_t)> raysFunc = RASTER_DATA::getFuncResizeForTensor(raysBuffer);

        float* max_w_pointer = nullptr;
        if (track_max_w)
            max_w_pointer = max_w.contiguous().data_ptr<float>();

        int num_vox_duplicated = 0;
        if(N != 0)
            num_vox_duplicated = rasterize_voxels_procedure(
                reinterpret_cast<char*>(voxelDataBuffer.contiguous().data_ptr()),
                raysFunc,
                voxels2raysFunc,

                W, 
                H,
                tan_fovx,
                tan_fovy,
                cx, 
                cy,
                w2c_matrix.contiguous().data_ptr<float>(),
                c2w_matrix.contiguous().data_ptr<float>(),
                
                
                N,
                num_sample_per_vox,
                bg_color,
                need_depth,

                morton_codes.contiguous().data_ptr<int64_t>(),
                vox_roots.contiguous().data_ptr<float>(),
                vox_length.contiguous().data_ptr<float>(),
                geos.contiguous().data_ptr<float>(),
                rgbs.contiguous().data_ptr<float>(),

                out_color.contiguous().data_ptr<float>(),
                out_depth.contiguous().data_ptr<float>(),
                out_T.contiguous().data_ptr<float>(),
                max_w_pointer,

                debug);
        
        // torch::Device device(torch::kCUDA);
        // torch::TensorOptions options(torch::kInt32);
        // torch::Tensor cc_ndup_per_vox = torch::full({N}, 0, options.device(device));
        // torch::Tensor cc_ndup_per_vox_csum = torch::full({N}, 0, options.device(device));
        // // torch::Tensor bboxes = torch::full({N}, 0, options.device(device));
        // torch::Tensor cam_quadrant_bitsets = torch::full({N}, 0, options.device(device));

        // char* vox_data_ptr = reinterpret_cast<char*>(voxelDataBuffer.contiguous().data_ptr());
        // RASTER_DATA::VoxelData vox_data = RASTER_DATA::VoxelData::sizeAloc(vox_data_ptr, N);

        // cudaMemcpy(
        //     cam_quadrant_bitsets.contiguous().data_ptr<int>(),
        //     (int*)vox_data.cam_quadrant_bitsets,
        //     N * sizeof(int),
        //     cudaMemcpyDeviceToHost);
        // CHECK_CUDA(debug)

        // // cudaMemcpy(
        // //     bboxes.contiguous().data_ptr<int>(),
        // //     (int*)vox_data.bboxes,
        // //     N * 2 * sizeof(int),
        // //     cudaMemcpyDeviceToHost);
        // // CHECK_CUDA(debug)

        // cudaMemcpy(
        //     cc_ndup_per_vox.contiguous().data_ptr<int>(),
        //     (int*)vox_data.ndup_per_vox,
        //     N * sizeof(int),
        //     cudaMemcpyDeviceToHost);
        // CHECK_CUDA(debug)

        // cudaMemcpy(
        //     cc_ndup_per_vox_csum.contiguous().data_ptr<int>(),
        //     (int*)vox_data.ndup_per_vox_csum,
        //     N * sizeof(int),
        //     cudaMemcpyDeviceToHost);
        // CHECK_CUDA(debug)

        // return std::make_tuple(num_vox_duplicated, out_color, out_color, cam_quadrant_bitsets, cc_ndup_per_vox, cc_ndup_per_vox_csum);
        return std::make_tuple(num_vox_duplicated, voxels2raysBuffer, raysBuffer, out_color, out_depth, out_T, max_w);
    }
}