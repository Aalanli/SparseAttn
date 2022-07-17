//#define PyExtension
#include <torch/types.h>
#ifdef PyExtension
#include <torch/python.h>
#else
#include <iostream>
#endif
#include <vector>
#include <cooperative_groups.h>
#include <math.h>

#include "utils.cuh"
#include "torch_utils.cuh"

namespace cg = cooperative_groups;

using T = float;

template <int N>
using Tensor = torch::PackedTensorAccessor32<T, N, torch::RestrictPtrTraits>;

/*
Chunked implementation where the values of eps is unique for each attention of 
each batch
*/
__device__ T softgate(const T x, const T a, const T b) {
    return 1 / (1 + __expf(b * (x - a)) + __expf(-b * (x + a)));
}

__device__ T softbound_const(const T bound) {
    return log(1 / bound - 1);
}

__device__ T softbound(const T c, const T a, const T b) {
    return c / b + a;
}

__device__ __forceinline__ float2 broadcast_sum(float2 a, float b) {
    a.x += b;
    a.y += b;
    return a;
}

__device__ __forceinline__ float cumsum_warp(float a, int wid) {
    T temp2;
    #pragma unroll
    for (int i = 1; i < 32; i <<= 1) {
        temp2 = __shfl_up_sync(-1, a, i);
        if (wid >= i) a += temp2;
    }
    return a;
}

__device__ __forceinline__ float2 cumsum_warp(float2 a, int wid) {
    T temp = cumsum_warp(a.x + a.y, wid);
    a.x = temp - a.y;
    a.y = temp;
    return a;
}

/*
__global__ void compute_stats(
    const T* __restrict__ eps, // eps.shape = [B, H]
    int* __restrict__ indicies, // indicies.shape = [B * H + 1]
    const int N, // N = B * H
    const T beta,
    const T delta)
{
    const int stride = blockDim.x;
    const int tid = threadIdx.x;
    const int wid = tid % 32;
    const int wp  = tid / 32;

    __shared__ T constC, glob_sum;
    __shared__ T sumPool[33];
    if (tid == 0) {
        eps[-1] = 0;
        constC = softbound_const(delta);
        glob_sum = 0;
    }
    __syncthreads();

    float2* eps2 = (float2*) eps;
    float2* indices2 = (float2*) indices2;
    for (int i = 0; i < N / 2; i += stride) {
        int idx = i + tid;
        float2 val;
        if (idx < int(N / 2))
            val = eps2[i];
        val.x = softbound(constC, val.x, beta) + glob_sum;
        val.y = softbound(constC, val.y, beta) + glob_sum;

        val = cumsum_warp(val, wid);
        if (wid == 31) {
            if (wp == 31) {
                sumPool[0] = 0;
                sumPool[32] = val.y;
            }
            else
                sumPool[wp + 1] = val.y;
        }
        __syncthreads();
        if (wp == 0) {
            sumPool[wid] = cumsum_warp(sumPool[wid], wid);
            if (tid == 1023)
                glob_sum += sumPool[31] + sumPool[32];
        }
        __syncthreads();
        val = broadcast_sum(val, sumPool[wp]);
        if (idx < int(N / 2))
            indices2[i] = val;
    }
    if (tid == 0 && int(N / 2) * 2 < N) {
        indicies[N - 1] = softbound(constC, eps[N - 1], beta) + glob_sum;
    }
}
*/


__global__ void sparse_qk_kernel(
    const Tensor<4> Q,
    const Tensor<4> K,
    const Tensor<3> shift,
    const int* __restrict__ acc,
    T* __restrict__ A,
    const T beta,
    const int bk_per_row)
{
    // blockDim = {semiCol, head, batch}
    const int batch = blockIdx.z;
    const int head = blockIdx.y;
    const int row = blockIdx.x / bk_per_row;
    const int col = blockIdx.x % bk_per_row;

    
}

// extracts the first element from the gpu pointer and copies it to the cpu
template<typename T>
T extract_element(T* gpu_ptr) {
    T* cpu_ptr = nullptr;
    cudaMemcpy(cpu_ptr, gpu_ptr, sizeof(T), cudaMemcpyDeviceToHost);
    return *cpu_ptr;
}

// creates the integer accessors that the attn kernel needs to 
// access the sparse tensor
// TODO: make this a kernel later
torch::Tensor torch_accessor_impl(const torch::Tensor eps, const T beta, const T bound) {
    T softbound_const = std::log(1 / bound - 1);

    auto opt = torch::TensorOptions().device(eps.device()).dtype(torch::kInt32);
    auto eps_ = (2 * (softbound_const / beta + eps.view({-1}))).ceil_().cumsum_(-1);
    auto result = torch::cat({torch::tensor({0}, opt), eps_}, 0);
    return result;
}


// returns the result of the attention calculation and the intermediate results of QK^T
std::vector<torch::Tensor> attn_v1(
    const torch::Tensor Q,
    const torch::Tensor K,
    const torch::Tensor V,
    const torch::Tensor eps, // eps.shape = [B, H]
    const torch::Tensor shift, // shift.shape = [B, H, L]
    const T beta,
    const T bound) 
{
    int L = Q.size(2);
    auto widths = torch_accessor_impl(eps, beta, bound);

    auto opt = ut::like_tensor(Q);
    int sizes = extract_element<int>(widths.data_ptr<int>() + (widths.size(0) - 1)); // gets the last element

    auto sparse_tensor = torch::empty({sizes * L}, opt);
    auto Y = torch::empty_like(V);

    return {Y, sparse_tensor};
}

int main() {
    auto opt = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto eps = torch::tensor({{4, 3, 4, 4}, {4, 3, 2, 1}}, opt);

    auto acc = torch_accessor_impl(eps, 2, 1e-4);
    ut::print(eps);
    ut::print(acc);
}