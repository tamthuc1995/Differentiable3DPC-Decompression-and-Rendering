#include "render_backward.h"
#include "raster_data.h"
#include "auxiliary.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

namespace VOXEL_RASTERIZER_BACKWARD {


// CUDA backward pass of sparse voxel rendering.
template <int n_sample>
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

    const float* __restrict__ out_T,
    const uint32_t* __restrict__ tile_last_vox,
    const uint32_t* __restrict__ actual_scanned_vox,

    const float* __restrict__ dL_dout_color,
    float* dL_dvox)
{
    
    // We rasterize again. Compute necessary block info.
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


    const uint2 vox_range_raw = first2last[tile_id];
    const uint2 vox_range = {vox_range_raw.x, tile_last_vox[tile_id]};

    if (vox_range.y > vox_range_raw.y)
    {
        printf("vox_range.y > vox_range.y - What !???");
        __trap();
    }
    if (vox_range_raw.x > vox_range_raw.y)
    {
        // TODO: remove sanity check.
        printf("range.x > range.y - What !???");
        __trap();
    }

    const int rounds = ((vox_range.y - vox_range.x + BLOCK2D_N - 1) / BLOCK2D_N);
    int toDo = vox_range.y - vox_range.x;

    // Allocate storage for batches of collectively fetched data.
    __shared__ int collected_vox_id[BLOCK2D_N];
    __shared__ int collected_quad_id[BLOCK2D_N];
    __shared__ uint2 collected_bbox[BLOCK2D_N];
    __shared__ float3 collected_vox_r[BLOCK2D_N];
    __shared__ float collected_vox_l[BLOCK2D_N];
    __shared__ float collected_geo_params[BLOCK2D_N * 8];
    __shared__ float3 collected_rgb[BLOCK2D_N];

    // In the forward render, we stored the final value for T, the
    // product of all (1 - alpha) factors.
    const float T_final = inside ? out_T[pix_id] : 0.f;
    float T = T_final;

    // We start from the back.
    // The last contributing voxel ID of each pixel is known from the forward.
    uint32_t contributor = toDo;
    const int last_contributor = inside ? actual_scanned_vox[pix_id] : 0;


    // Init gradient from the last computation node.
    float3 dL_dpix;
    float last_dL_dT;
    if (inside)
    {
        dL_dpix.x = dL_dout_color[0 * H * W + pix_id];
        dL_dpix.y = dL_dout_color[1 * H * W + pix_id];
        dL_dpix.z = dL_dout_color[2 * H * W + pix_id];
        last_dL_dT = bg_color * (dL_dpix.x + dL_dpix.y + dL_dpix.z);
    }


    int j_lst[BLOCK2D_N];
    // Traverse all voxels backward.
    for (int i = 0; i < rounds; i++, toDo -= BLOCK2D_N)
    {
        // Load auxiliary data into shared memory 
        //    start in the BACK and load them in revers order.
        block.sync();
        const int progress = i * BLOCK2D_N + thread_id;
        if (vox_range.x + progress < vox_range.y)
        {
            uint32_t vox_val = vox_list[vox_range.y - progress - 1];
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

        // Iterate over voxels.
        const int end_j = min(BLOCK2D_N, toDo);
        int j_lst_top = -1;
        for (int j = 0; !done && j < end_j; j++)
        {
            // Keep track of current voxel ID. Skip, if this one
            // is behind the last contributor for this pixel.
            contributor--;
            if (contributor >= last_contributor)
                continue;

            /**************************
            Below, we first compute blending values, as in the forward.
            **************************/

            // Check if the pixel in the projected bbox region.
            // Check if the quadrant id match the pixel.
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


        for (int jj = 0; !done && jj <= j_lst_top; jj++)
        {
            int j = j_lst[jj];

            // Compute ray aabb intersection
            const float3 vox_r = collected_vox_r[j];
            const float vox_l = collected_vox_l[j];
            const float2 ab = ray_aabb(vox_r, vox_l, ro, rd_inv);
            const float a = ab.x;
            const float b = ab.y;

            float geo_params[8];
            for (int k=0; k<8; ++k)
                geo_params[k] = collected_geo_params[j*8 + k];
            float dL_dgeo_params[8] = {0.f};

            float vol_int = 0.f;
            float dI_dgeo_params[8] = {0.f};
            float each_dI_dgeo_params[n_sample][8];
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

                const float dd_dd = STEP_SZ_SCALE * step_sz * exp_linear_11_bw(d);
                for (int iii=0; iii<8; ++iii)
                {
                    float tmp = dd_dd * interp_w[iii];
                    dI_dgeo_params[iii] += tmp;
                }
            }

            // Compute alpha from volume integral.
            // Follow 3DGS's alpha clamping to avoid numerical instabilities.
            const float exp_neg_vol_int = expf(-vol_int);
            float alpha = min(MAX_ALPHA, 1.f - exp_neg_vol_int);
            if (alpha < MIN_ALPHA)
                continue;

            // Recover the blending weight of this voxel.
            T = T / (1.f - alpha);
            const float pt_w = alpha * T;

            // Propagate gradients to per-voxel colors and 
            //   keep gradients w.r.t. voxel alpha.

            // Load from share memory.
            const int vox_id = collected_vox_id[j];
            const float3 c = collected_rgb[j];

            // The gradients w.r.t. voxel alpha.
            float dL_dpt_w = dot(dL_dpix, c);

            // Compute gradient accumulated to the alpha.
            const float dL_dalpha = T * (dL_dpt_w - last_dL_dT);

            /**************************
            Backprop from voxel volume integral to surface parameters.
            **************************/
            const float dL_dI = dL_dalpha * exp_neg_vol_int;

            /**************************
            Sum up the gradient from rendering below.
            **************************/
           
            float dL_drgb[3] = {pt_w * dL_dpix.x, pt_w * dL_dpix.y, pt_w * dL_dpix.z};
            for (int iii=0; iii<8; ++iii)
                dL_dgeo_params[iii] += dL_dI * dI_dgeo_params[iii];


            /**************************
            Write back the gradient below.
            **************************/
            float grad_pack[12];
            #pragma unroll
            for (int iii=0; iii<8; ++iii)
                grad_pack[iii] = dL_dgeo_params[iii];
            grad_pack[8] = dL_drgb[0];
            grad_pack[9] = dL_drgb[1];
            grad_pack[10] = dL_drgb[2];
            grad_pack[11] = fabs(dL_dalpha * alpha);

            const int base_id = cg::this_grid().thread_rank();
            #pragma unroll
            for (int iii=0; iii<12; ++iii)
                atomicAdd(dL_dvox + vox_id * 12 + (base_id+iii)%12, grad_pack[(base_id+iii)%12]);
        }

    }
}


#ifndef BwRendFunc
// Dirty trick. The argument name must be aligned with BACKWARD::render.
#define BwRendFunc(...) \
    renderCUDA<__VA_ARGS__>
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

    const uint2* bboxes,
    const float3* vox_roots,
    const float* vox_length,
    const float* geos,
    const float3* rgbs,

    const float* out_T,
    const uint32_t* tile_last_vox,
    const uint32_t* actual_scanned_vox,

    const float* dL_dout_color,
    float* dL_dvox)
{
    // The density_mode now is always EXP_LINEAR_11_MODE
    const auto kernel_func =
        (num_sample_per_vox == 3) ?
            BwRendFunc(3) :
        (num_sample_per_vox == 2) ?
            BwRendFunc(2) :
            BwRendFunc(1) ;

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

        out_T,
        tile_last_vox,
        actual_scanned_vox,

        dL_dout_color,
        dL_dvox
    );
}



// Interface for python to run backward pass of voxel rasterization.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
voxels_backward_rasterizing(
    const int Ndup,
    
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

    const torch::Tensor& morton_codes,
    const torch::Tensor& vox_roots,
    const torch::Tensor& vox_length,
    const torch::Tensor& geos,
    const torch::Tensor& rgbs,

    const torch::Tensor& voxelDataBuffer,
    const torch::Tensor& binningVox2RayBuffer,
    const torch::Tensor& rayGroupsBuffer,
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
    RASTER_DATA::VoxelData voxel_data = RASTER_DATA::VoxelData::sizeAloc(voxdataB_ptr, N);

    char* vox2rayB_ptr = reinterpret_cast<char*>(binningVox2RayBuffer.contiguous().data_ptr());
    RASTER_DATA::BindingVoxel2RayData vox2ray_data = RASTER_DATA::BindingVoxel2RayData::sizeAloc(vox2rayB_ptr, Ndup);
    
    char* raysB_ptr = reinterpret_cast<char*>(rayGroupsBuffer.contiguous().data_ptr());
    RASTER_DATA::GroupRaysData rays_data = RASTER_DATA::GroupRaysData::sizeAloc(
        raysB_ptr, 
        image_width * image_height, 
        tile_grid.x * tile_grid.y
    );

    // Compute loss gradients w.r.t. surface property and voxel color.
    render(
        tile_grid, 
        block,
        rays_data.first2last,
        vox2ray_data.vox_list,
        
        image_width,
        image_height,
        tan_fovx,
        tan_fovy,
        cx,
        cy,
        c2w_matrix.contiguous().data_ptr<float>(),

        num_sample_per_vox,
        bg_color,

        voxel_data.bboxes,
        (float3*)(vox_roots.contiguous().data_ptr<float>()),
        vox_length.contiguous().data_ptr<float>(),
        geos.contiguous().data_ptr<float>(),
        (float3*)(rgbs.contiguous().data_ptr<float>()),

        out_T.contiguous().data_ptr<float>(),
        rays_data.tile_last_vox,
        rays_data.actual_scanned_vox,

        dL_dout_color.contiguous().data_ptr<float>(),
        dL_dvox.contiguous().data_ptr<float>()
    );
    CHECK_CUDA(debug);

    std::vector<torch::Tensor> gradient_lst = dL_dvox.split({geos.size(1), 3, 1}, 1);
    torch::Tensor dL_dgeos = gradient_lst[0].contiguous();
    torch::Tensor dL_drgbs = gradient_lst[1].contiguous();
    torch::Tensor subdiv_p_bw = gradient_lst[2].contiguous();

    return std::make_tuple(dL_dgeos, dL_drgbs, subdiv_p_bw);
}

}