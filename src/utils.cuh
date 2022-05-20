#pragma once
#include <tuple>
#include <string>

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

namespace utils {

void print() {
    std::cout << "\n";
}

template <typename T, typename... Args>
void print(T val, Args... args) {
    std::cout << val << " ";
    print(args...);
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void cudaLastError() {
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}

// return true is successful, false if otherwise
std::tuple<bool, std::string> cudaErrorState() {
    auto code1 = cudaPeekAtLastError();
    auto code2 = cudaDeviceSynchronize();
    auto success = (code1 == cudaSuccess) && (code2 == cudaSuccess);
    return std::tuple<bool, std::string>(success, cudaGetErrorString(code1));
}


// common ops used


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
__device__ T reduce_sum_tile_shuffle(cg::thread_block_tile<tile_sz> g, T val) {
    #pragma unroll
    for (int i = g.size() / 2; i > 0; i /= 2) {
        val += g.shfl_down(val, i);
    }
    return val;
}

__device__ __forceinline__ float4 mul(const float4 a, const float4 b) {
    float4 c;
    c.x = a.x * b.x; c.y = a.y * b.y; c.z = a.z * b.z; c.w = a.w * b.w;
    return c;
}

}