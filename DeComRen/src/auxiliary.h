
#ifndef RASTERIZER_AUXILIARY_H_INCLUDED
#define RASTERIZER_AUXILIARY_H_INCLUDED

#include "config.h"


__forceinline__ __device__ float dot(const float3& a, const float3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}


__forceinline__ __device__ float3 get_cam_position(const float* c2w_3x4matrix)
{
    float3 last_col = {c2w_matrix[3], c2w_matrix[7], c2w_matrix[11]};
    return last_col;
}

__forceinline__ __device__ float3 transform_3x4(const float* matrix, const float3& p)
{
    float3 transformed = {
        matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z + matrix[3],
        matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z + matrix[7],
        matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z + matrix[11]
    };
    return transformed;
}

__forceinline__ __device__ uint32_t compute_direction_quadrant_id(float3 dir)
{
    return ((dir.x < 0) << 2) | ((dir.y < 0) << 1) | (dir.z < 0);
}

__forceinline__ __device__ uint32_t compute_cam_to_point_quadrant_id(float3 point, float3 cam_org)
{
    return ((point.x < cam_org.x) << 2) | ((point.y < cam_org.y) << 1) | (point.z < cam_org.z);
}


// Debugging helper.
#define CHECK_CUDA(debug) \
if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}


#endif