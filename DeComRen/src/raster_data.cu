#include "raster_data.h"
#include "auxiliary.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>

#include <cooperative_groups.h>
namespace cg = cooperative_groups;


namespace RASTER_DATA {

    std::function<char*(size_t N)> getFuncResizeForTensor(torch::Tensor& t)
    {
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
    size_t size_required(size_t N)
    {
        char* size = nullptr;
        T::sizeAloc(size, N);
        return ((size_t)size) + 128;
    }

    template<typename T>
    size_t size_required(size_t N1, size_t N2);
    {
        char* size = nullptr;
        T::sizeAloc(size, N1, N2);
        return ((size_t)size) + 128;
    }

    // Explicit template initialization
    template size_t size_required<VoxelData>(size_t N);
    template size_t size_required<BindingVoxel2RayData>(size_t Ndup);
    template size_t size_required<GroupRaysData>(size_t nrays, size_t ngroups);

    // Get and asign pointers to VoxelData
    VoxelData VoxelData::sizeAloc(char*& chunk, size_t N)
    {
        VoxelData voxinfo;
        obtain(chunk, voxinfo.n_duplicates, N, 128);
        obtain(chunk, voxinfo.n_duplicates_csum, N, 128);
        obtain(chunk, voxinfo.bboxes, N, 128);
        obtain(chunk, voxinfo.cam_quadrant_bitsets, N, 128);

        // Prepare temporary space for scanning (prefix-sum).
        // input nullptr will init the sum process but do no work
        cub::DeviceScan::InclusiveSum(
            nullptr,
            voxinfo.temp_csum_bytes,
            voxinfo.n_duplicates,
            voxinfo.n_duplicates,
            N
        );
        obtain(chunk, voxinfo.temp_csum_storage, voxinfo.temp_csum_bytes, 128);

        return voxinfo;
    }

    BindingVoxel2RayData BindingVoxel2RayData::sizeAloc(char*& chunk, size_t Ndup)
    {
        BindingVoxel2RayData vox2ray;
        obtain(chunk, vox2ray.vox_list_keys_unsorted, Ndup, 128);
        obtain(chunk, vox2ray.vox_list_unsorted, Ndup, 128);
        obtain(chunk, vox2ray.vox_list_keys, Ndup, 128);
        obtain(chunk, vox2ray.vox_list, Ndup, 128);

        // Prepare temporary space for sorting.
        // input nullptr will init the sum process but do no work
        cub::DeviceRadixSort::SortPairs(
            nullptr,
            vox2ray.temp_storage_bytes,
            vox2ray.vox_list_keys_unsorted,
            vox2ray.vox_list_keys,
            vox2ray.vox_list_unsorted, 
            vox2ray.vox_list,
            Ndup
        );
        obtain(chunk, vox2ray.temp_sorting_storage, vox2ray.temp_storage_bytes, 128);
        return vox2ray;
    }

    GroupRaysData GroupRaysData::sizeAloc(char*& chunk, size_t nrays, size_t ngroups)
    {
        GroupRaysData groups;
        obtain(chunk, groups.first2last, ngroups, 128);
        obtain(chunk, groups.actual_last, ngroups, 128);
        obtain(chunk, groups.actual_cost, nrays, 128);
        return groups;
    }

    // DEBUGING UTIL
    // std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    // unpack_ImageState(
    //     const int image_width, const int image_height,
    //     const torch::Tensor& imageBuffer);

    // torch::Tensor filter_voxels_data(
    //     const int ori_P,
    //     const torch::Tensor& indices,
    //     const torch::Tensor& vox);

}

#endif