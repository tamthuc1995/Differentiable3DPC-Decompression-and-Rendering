
#ifndef RASTER_DATA_H_INCLUDED
#define RASTER_DATA_H_INCLUDED

#include <cuda_runtime.h>
#include <torch/extension.h>

namespace RASTER_DATA {

    // This hanndy function will return a Func:
    //  pin a tensor pointer with a functional resizing for that tensor alone.
    std::function<char*(size_t N)> getFuncResizeForTensor(torch::Tensor& t);

    // This function will asign array pointer "ptr" with next avalable mem from chunk of size "count" (with alignment bits), 
    // and move chunk to the next mem block after array &ptr
	template <typename T>
	static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment);

    // Check required mem for object T
    template<typename T>
    size_t size_required(size_t P);

    template<typename T>
    size_t size_required(size_t P, size_t Q);

    struct VoxelData
    {
        // Voxel raster variables. 
        // There are N voxels.

        // Render data
        uint32_t* n_duplicates; // Nx1 array to store num required duplication for each voxel: num_tiles * num_morton_direction
        uint32_t* n_duplicates_csum; // Nx1 array to store cumulative sum of n_duplicates, used for BindingData allocation

        // processing meta for cumulative sum later on
        char* temp_csum_storage;
        size_t temp_csum_bytes;

        // Nx2 array, The 2 (min, max)-coners define bbox enclosed the voxel. 
        // Very handy trick learned from Charles Loop's code to pack each (x, y) point into one 32bits integer, 
        // max pixels position is 16 bits anyway.
        uint2* bboxes;  

        // Voxel sorting related variables.
        uint32_t* cam_quadrant_bitsets;  // Nx1 array of 8 bits int to signify possible touched cammera quadrants (8 possible)

        // Init pointer to each data above
        static VoxelData sizeAloc(char*& chunk, size_t N);
    };

    struct BindingVoxel2RayData
    {   
        // Ndup is total duplicated voxels
        // Temporary data for sorting
        char* temp_sorting_storage;
        size_t temp_storage_bytes;

        // Data to store required duplicated voxels
        uint64_t* vox_list_keys_unsorted;
        uint32_t* vox_list_unsorted;

        // Data to store sorted duplicated voxels
        // The sorted vox_list_keys is now as:
        //   [sorted voxels for tile 1 ... sorted voxels for tile 2 ... sorted voxels for tile 3 ...]
        uint64_t* vox_list_keys;
        uint32_t* vox_list;
        
        static BindingVoxel2RayData sizeAloc(char*& chunk, size_t Ndup);
    };

    struct GroupRaysData
    {
        // Store rays data, a group of rays is of size BLOCK2D_X x BLOCK2D_Y, default is 16 x 16 pixels.
        // Array of size: num_tiles x 2 
        //  -- store pair of (first, last) index pointing to duplicated voxels vox_list_keys
        uint2* first2last;
        
        // Array of size: num_tiles x 1
        //  -- store the ACTUAL last index pointing to duplicated voxels vox_list_keys of the tile
        //  -- for the backward rendering stage.
        uint32_t* actual_last;
        
        // Array of size: num pixels x 1
        //  -- store last contributed sub-index of duplicated voxels in vox_list_keys (offset from first2last.first )
        uint32_t* actual_cost;

        static GroupRaysData sizeAloc(char*& chunk, size_t N, size_t n_tiles);
    };

    // DEBUGING UTIL
    // std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    // unpack_ImageState(
    //     const int image_width, const int image_height,
    //     const torch::Tensor& imageBuffer);

    // torch::Tensor filter_geomState(
    //     const int ori_P,
    //     const torch::Tensor& indices,
    //     const torch::Tensor& geomState);

}

#endif