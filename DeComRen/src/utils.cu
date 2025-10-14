#include "utils.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

namespace UTILS {
    torch::Tensor position_to_mortoncode(const torch::Tensor& position, const torch::Tensor& octree_level)
    {
        const int N = position.size(0);

        const auto tensor_opt = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
        torch::Tensor mortoncode = torch::empty({N, 1}, tensor_opt);

        return mortoncode;
    }

    torch::Tensor mortoncode_to_position(const torch::Tensor& mortoncode, const torch::Tensor& octree_level)
    {
        const int N = mortoncode.size(0);

        const auto tensor_opt = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
        torch::Tensor position = torch::empty({N, 3}, tensor_opt);

        return position;
    }
}