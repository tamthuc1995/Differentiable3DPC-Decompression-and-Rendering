
#include <torch/extension.h>
#include "src/config.h"
#include "src/utils.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, modulee) {
    modulee.def("position_to_mortoncode", &UTILS::position_to_mortoncode);
    modulee.def("mortoncode_to_position", &UTILS::mortoncode_to_position);

    // Read only constants
    modulee.attr("MAX_OCTREE_LEVELS") = pybind11::int_(MAX_OCTREE_LEVELS);
    modulee.attr("BLOCK2D_X")         = pybind11::int_(BLOCK2D_X);
    modulee.attr("BLOCK2D_Y")         = pybind11::int_(BLOCK2D_Y);
    modulee.attr("BLOCK2D_N")         = pybind11::int_(BLOCK2D_N);
}
