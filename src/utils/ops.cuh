/* 
device code used only in cuda kernels
*/

#pragma once
#include <cooperative_groups.h>

namespace ops {

namespace cg = cooperative_groups;

__device__ __forceinline__ float mul_accum(const float4 &a, const float4 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}
__device__ __forceinline__ float mul_accum(const float4 a, const float b) {
    return a.x * b + a.y * b + a.z * b + a.w * b;
} 
__device__ __forceinline__ float mul_accum(const float a, const float4 b) {
    return mul_accum(b, a);
}

template <int tile_sz, typename T>
__device__ T reduce_sum_shfl_down(cg::thread_block_tile<tile_sz> g, T val) {
    #pragma unroll
    for (int i = g.size() / 2; i > 0; i /= 2) {
        val += g.shfl_down(val, i);
    }
    return val;
}

template <int tile_sz, typename T>
__device__ T reduce_max_shfl_down(cg::thread_block_tile<tile_sz> g, T val) {
    #pragma unroll
    for (int i = g.size() / 2; i > 0; i /= 2) {
        val = max(val, g.shfl_down(val, i));
    }
    return val;
}

template <int tile_sz, typename T>
__device__ __forceinline__ T reduce_sum_shfl_down(T val, unsigned mask) {
    #pragma unroll
    for (int i = tile_sz / 2; i > 0; i /= 2) {
        val += __shfl_down_sync(mask, val, i, tile_sz);
    }
    return val;
}

template <int tile_sz, typename T>
__device__ __forceinline__ T reduce_max_shfl_down(T val, unsigned mask) {
    #pragma unroll
    for (int i = tile_sz / 2; i > 0; i /= 2) {
        val = max(val, __shfl_down_sync(mask, val, i, tile_sz));
    }
    return val;
}

template <int tile_sz, typename T, int N>
__device__ __forceinline__ void reduce_sum_shfl_down_vec(T* val, unsigned mask) {
    #pragma unroll
    for (int i = tile_sz / 2; i > 0; i /= 2) {
        #pragma unroll
        for (int j = 0; j < N; j++) {
            val[j] += __shfl_down_sync(mask, val[j], i, tile_sz);
        }
    }
}

__device__ __forceinline__ float4 mul(const float4 a, const float4 b) {
    float4 c;
    c.x = a.x * b.x; c.y = a.y * b.y; c.z = a.z * b.z; c.w = a.w * b.w;
    return c;
}

__device__ __forceinline__ float4 mul(const float4 a, const float b) {
    float4 c;
    c.x = a.x * b; c.y = a.y * b; c.z = a.z * b; c.w = a.w * b;
    return c;
}

__device__ __forceinline__ float elem_add(float a, float b) {
    return a + b;
}

__device__ __forceinline__ float2 elem_add(float2 a, float2 b) {
    return {a.x + b.x, a.y + b.y};
}

template <typename T>
__device__ T softgate(const T x, const T a, const T b) {
    return 1 / (1 + __expf(b * (x - a)) + __expf(-b * (x + a)));
}

int next_power_of_2(int n) {
    n -= 1;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n += 1;
    return n;
}

}