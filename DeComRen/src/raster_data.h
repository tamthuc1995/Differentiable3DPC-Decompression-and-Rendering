
#ifndef RASTER_DATA_H_INCLUDED
#define RASTER_DATA_H_INCLUDED

#include <cuda_runtime.h>
#include <torch/extension.h>

namespace RASTER_DATA {

    // This hanndy function will return a Func:
    //  pin a tensor pointer with a functional resizing for that tensor alone.
    std::function<char*(size_t N)> getFuncResizeForTensor(torch::Tensor& t) {
        auto lambda = [&t](size_t N) {
            t.resize_({(long long)N});
            return reinterpret_cast<char*>(t.contiguous().data_ptr());
        };
        return lambda;
    }

	template <typename T>
	static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
	{
		std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
		ptr = reinterpret_cast<T*>(offset);
		chunk = reinterpret_cast<char*>(ptr + count);
	}

    template<typename T>
    size_t mem_required(size_t P);

    template<typename T>
    size_t mem_required(size_t P, size_t Q);

    struct VoxelData
    {
        // Voxel raster variables. 
        // There are N voxels.

        // Render data
        uint32_t* n_duplicates; // Nx1 array to store num required duplication for each voxel: num_tiles * num_morton_direction
        uint32_t* n_duplicates_cumsum; // Nx1 array to store cumulative sum of n_duplicates, used for BindingData allocation

        // processing meta for cumulative sum later on
        size_t cumsum_size;
        char* cumsum_temp_space;

        // Nx2 array, The 2 (min, max)-coners define bbox enclosed the voxel. 
        // Very handy trick learned from Charles Loop's code to pack each (x, y) point into one 32bits integer, 
        // max pixels position is 16 bits anyway.
        uint2* bboxes;  

        // Voxel sorting related variables.
        uint32_t* cam_quadrant_bitsets;  // Nx1 array of 8 bits int to signify possible touched cammera quadrants (8 possible)

        // Init pointer to each data above
        static VoxelData fromChunk(char*& chunk, size_t N);
    };

    struct BindingVoxelRayData
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
        
        static BindingVoxelRayData fromChunk(char*& chunk, size_t Ndup);
    };

    struct ImageData
    {   
        // Store tiles data, a tile is of size BLOCK2D_X x BLOCK2D_Y, default is 16 x 16.
        // Array of size: num_tiles x 2 
        //  -- store pair of (first, last) index pointing to duplicated voxels vox_list_keys
        uint2* ranges;
        
        // Array of size: num_tiles x 1
        //  -- store the ACTUAL last index pointing to duplicated voxels vox_list_keys of the tile
        //  -- for the backward rendering stage.
        uint32_t* tile_last;
        
        // Array of size: num pixels x 1
        //  -- store last contributed sub-index of duplicated voxels in vox_list_keys (offset from ranges.first )
        uint32_t* n_contrib;

        static ImageData fromChunk(char*& chunk, size_t N, size_t n_tiles);
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