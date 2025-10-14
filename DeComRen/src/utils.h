
#ifndef UTILS_H_INCLUDED
#define UTILS_H_INCLUDED

#include <torch/extension.h>

namespace UTILS {

    // Converting Morton code (Int64) <-> point positions (X, Y, Z)
    torch::Tensor position_to_mortoncode(const torch::Tensor& position, const torch::Tensor& octree_level);
    torch::Tensor mortoncode_to_position(const torch::Tensor& mortoncode, const torch::Tensor& octree_level);

}


#endif
