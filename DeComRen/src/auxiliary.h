#ifndef RASTERIZER_AUXILIARY_H_INCLUDED
#define RASTERIZER_AUXILIARY_H_INCLUDED

#include "config.h"

// Octant ordering tables
template<uint64_t id, int n>
struct repeat_3bits
{
    static constexpr uint64_t value = (repeat_3bits<id, n-1>::value << 3) | id;
};

template<uint64_t id>
struct repeat_3bits<id, 1>
{
    static constexpr uint64_t value = id;
};

__constant__ uint64_t order_tables[8] = {
    repeat_3bits<0ULL, MAX_OCTREE_LEVELS>::value,
    repeat_3bits<1ULL, MAX_OCTREE_LEVELS>::value,
    repeat_3bits<2ULL, MAX_OCTREE_LEVELS>::value,
    repeat_3bits<3ULL, MAX_OCTREE_LEVELS>::value,
    repeat_3bits<4ULL, MAX_OCTREE_LEVELS>::value,
    repeat_3bits<5ULL, MAX_OCTREE_LEVELS>::value,
    repeat_3bits<6ULL, MAX_OCTREE_LEVELS>::value,
    repeat_3bits<7ULL, MAX_OCTREE_LEVELS>::value
};

__forceinline__ __device__ uint64_t compute_directional_code(uint64_t vox_morton_code, int quadrant_id)
{
    return vox_morton_code ^ order_tables[quadrant_id];
}


__forceinline__ __device__ float dot(const float3& a, const float3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__forceinline__ __device__ float3 last_col_3x4(const float* matrix)
{
    float3 last_col = {matrix[3], matrix[7], matrix[11]};
    return last_col;
}

__forceinline__ __host__ __device__ float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__forceinline__ __host__ __device__ float3 operator-(float b, float3 a)
{
    return make_float3(b - a.x, b - a.y, b - a.z);
}

__forceinline__ __host__ __device__ float3 operator-(float3 a, float b)
{
    return make_float3(a.x - b, a.y - b, a.z - b);
}

__forceinline__ __host__ __device__ float3 operator-(float3 a)
{
    return make_float3(-a.x, -a.y, -a.z);
}

__forceinline__ __host__ __device__ float3 operator*(float3 a, float3 b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__forceinline__ __host__ __device__ float3 operator*(float b, float3 a)
{
    return make_float3(b * a.x, b * a.y, b * a.z);
}

__forceinline__ __host__ __device__ float3 operator*(float3 a, float b)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__forceinline__ __host__ __device__ float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__forceinline__ __host__ __device__ float3 operator+(float b, float3 a)
{
    return make_float3(a.x + b, a.y + b, a.z + b);
}

__forceinline__ __host__ __device__ float3 operator+(float3 a, float b)
{
    return make_float3(a.x + b, a.y + b, a.z + b);
}

__forceinline__ __host__ __device__ float2 operator-(float2 a, float2 b)
{
    return make_float2(a.x - b.x, a.y - b.y);
}

__forceinline__ __host__ __device__ float2 operator+(float2 a, float2 b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}

__forceinline__ __host__ __device__ float2 operator*(float b, float2 a)
{
    return make_float2(a.x * b, a.y * b);
}

__forceinline__ __host__ __device__ float2 operator*(float2 a, float b)
{
    return make_float2(a.x * b, a.y * b);
}


__forceinline__ __device__ float3 get_cam_position(const float* c2w_3x4matrix)
{
    float3 last_col = {c2w_3x4matrix[3], c2w_3x4matrix[7], c2w_3x4matrix[11]};
    return last_col;
}


__forceinline__ __device__ float3 rotate_3x4(const float* matrix, const float3& p)
{
    float3 transformed = {
        matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
        matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
        matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z
    };
    return transformed;
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


__forceinline__ __device__ bool pix_in_bbox(const uint2& pix, const uint2& bbox)
{
    bool valid_xmin = pix.x >= (bbox.x >> 16);
    bool valid_ymin = pix.y >= (bbox.x << 16 >> 16);
    bool valid_xmax = pix.x <= (bbox.y >> 16);
    bool valid_ymax = pix.y <= (bbox.y << 16 >> 16);
    return valid_xmin && valid_ymin && valid_xmax && valid_ymax;
}

__forceinline__ __device__ float2 ray_aabb(float3 vox_root, float vox_length, float3 cam_org, float3 ray_dir_inv)
{   
    
    float3 vox_center = vox_root + 0.5f * vox_length;
    float vox_radius = 0.5f * vox_length;
    float3 ray_dir = vox_center - cam_org;
    float3 c0_ = (ray_dir - vox_radius) * ray_dir_inv;
    float3 c1_ = (ray_dir + vox_radius) * ray_dir_inv;

    float3 c0 = make_float3(min(c0_.x, c1_.x), min(c0_.y, c1_.y), min(c0_.z, c1_.z));
    float3 c1 = make_float3(max(c0_.x, c1_.x), max(c0_.y, c1_.y), max(c0_.z, c1_.z));
    float2 ab = make_float2(
        max(max(c0.x, c0.y), c0.z),
        min(min(c1.x, c1.y), c1.z)
    );
    return ab;
}

__forceinline__ __device__ float exp_linear_11(float x)
{
    return (x > 1.1f) ? x : expf(0.909090909091f * x - 0.904689820196f);
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