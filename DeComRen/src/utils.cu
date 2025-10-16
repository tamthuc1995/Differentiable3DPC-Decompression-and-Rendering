#include "utils.h"
#include "config.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

namespace UTILS {

    __global__ void position_to_mortoncode_cuda (
        const int N,
        const int64_t* __restrict__ p_position,
        const int8_t* __restrict__ p_octree_level,
        int64_t* __restrict__ p_mortoncode)
    {
        auto idx = cg::this_grid().thread_rank();
        if (idx >= N)
            return;

        const int curr_level = p_octree_level[0];
        int x = p_position[idx * 3 + 0];
        int y = p_position[idx * 3 + 1];
        int z = p_position[idx * 3 + 2];

        int64_t code = 0;
        for (int l=0; l<=curr_level; ++l)
        {
            // Extract 3bits sequence from x, y, z and shift them
            int64_t triplet_bits = (x & 1) << 2;
            triplet_bits |= (y & 1) << 1;
            triplet_bits |= (z & 1);
            code |= triplet_bits << (3 * l);

            // move on to next bits of x, y, z
            // should become 0 when the loop end
            x >>= 1;
            y >>= 1;
            z >>= 1;
        }

        p_mortoncode[idx] = code;
    }


    __global__ void mortoncode_to_position_cuda (
        const int N,
        const int64_t* __restrict__ p_mortoncode,
        const int8_t* __restrict__ p_octree_level,
        int64_t* __restrict__ p_position)
    {
        auto idx = cg::this_grid().thread_rank();
        if (idx >= N)
            return;

        const int curr_level = p_octree_level[0];
        int64_t code = p_mortoncode[idx];
        int x = 0, y = 0, z = 0;

        for (int l=0; l<=curr_level; ++l)
        {
            int triplet_bits = static_cast<int>(code & 0b111);
            x |= ((triplet_bits & 0b100) >> 2) << l;
            y |= ((triplet_bits & 0b010) >> 1) << l;
            z |= ((triplet_bits & 0b001)) << l;

            // path should be 0 after lv iterations.
            code >>= 3;
        }

        p_position[idx * 3 + 0] = x;
        p_position[idx * 3 + 1] = y;
        p_position[idx * 3 + 2] = z;
    }


    torch::Tensor position_to_mortoncode(const torch::Tensor& position, const torch::Tensor& octree_level)
    {
        const int N_l = position.size(0);

        const auto tensor_opt = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
        torch::Tensor mortoncode = torch::empty({N_l, 1}, tensor_opt);

        if (N_l > 0) 
            position_to_mortoncode_cuda <<<(N_l + BLOCK2D_N-1) / BLOCK2D_N, BLOCK2D_N>>> (
                N_l,
                position.contiguous().data_ptr<int64_t>(),
                octree_level.contiguous().data_ptr<int8_t>(),
                mortoncode.contiguous().data_ptr<int64_t>()
            );

        return mortoncode;
    }

    torch::Tensor mortoncode_to_position(const torch::Tensor& mortoncode, const torch::Tensor& octree_level)
    {
        const int N_l = mortoncode.size(0);

        const auto tensor_opt = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
        torch::Tensor position = torch::empty({N_l, 3}, tensor_opt);

        if (N_l > 0)
            mortoncode_to_position_cuda <<<(N_l + BLOCK2D_N-1) / BLOCK2D_N, BLOCK2D_N>>> (
                N_l,
                mortoncode.contiguous().data_ptr<int64_t>(),
                octree_level.contiguous().data_ptr<int8_t>(),
                position.contiguous().data_ptr<int64_t>()
            );

        return position;
    }
}