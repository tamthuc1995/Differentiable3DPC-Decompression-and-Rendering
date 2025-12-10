
#include <torch/extension.h>
#include "src/config.h"

#include "src/raster_data.h"
#include "src/preprocess.h"
#include "src/render_forward.h"
#include "src/render_backward.h"
#include "src/gather_voxel_params.h"

#include "src/utils.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, modulee) {
    // Utils
    modulee.def("position_to_mortoncode", &UTILS::position_to_mortoncode);
    modulee.def("mortoncode_to_position", &UTILS::mortoncode_to_position);

    // Preprocessing
    modulee.def("rasterize_preprocess", &PREPROCESS::rasterize_preprocess);
    
    // Rendering
    modulee.def("voxels_rasterizing", &VOXEL_RASTERIZER::voxels_rasterizing);
    modulee.def("voxels_backward_rasterizing", &VOXEL_RASTERIZER_BACKWARD::voxels_backward_rasterizing);

    // Voxel params gathering operation
    modulee.def("gather_geo_params", &VOXEL_PARAMS_GATHER::gather_geo_params);
    modulee.def("gather_geo_params_bw", &VOXEL_PARAMS_GATHER::gather_geo_params_bw);
    modulee.def("gather_color_params", &VOXEL_PARAMS_GATHER::gather_color_params);
    modulee.def("gather_color_params_bw", &VOXEL_PARAMS_GATHER::gather_color_params_bw);

    // Read only constants
    modulee.attr("MAX_NUM_SAMPLE")    = pybind11::int_(MAX_NUM_SAMPLE);
    modulee.attr("MAX_OCTREE_LEVELS") = pybind11::int_(MAX_OCTREE_LEVELS);
    modulee.attr("BLOCK2D_X")         = pybind11::int_(BLOCK2D_X);
    modulee.attr("BLOCK2D_Y")         = pybind11::int_(BLOCK2D_Y);
    modulee.attr("BLOCK2D_N")         = pybind11::int_(BLOCK2D_N);
}
