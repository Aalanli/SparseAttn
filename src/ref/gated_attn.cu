#define PyExtension
#include <torch/types.h>
#ifdef PyExtension
#include <torch/python.h>
#else
#include <iostream>
#endif
#include <math.h>
#include <vector>
#include "utils.cuh"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;
namespace ut = utils;

using T = float;

// __constant__ T limEps = 0.0001;
// __constant__ T CBound = log(((1 - limEps) / limEps));

__constant__ T CBound = 9.2102403669758; 

template <int N>
using Tensor = torch::PackedTensorAccessor32<T, N, torch::RestrictPtrTraits>;

__device__ T soft_gatev1(T x, T a, T b) {
    return 1 / (1 + __expf(b * (x - a)) + __expf(-b * (x + a)));
}

__device__ T soft_bounds(T a, T b) {
    return CBound / b + a;
}

// D is the number of threads
template <int tile, int D>
__global__ void attn_impl_kernel(
    const T* __restrict__ Q, // Q.shape       = [..., L, D]
    const T* __restrict__ K, // K.shape       = [..., L, D]
    const T* __restrict__ V, // V.shape       = [..., L, D]
    T* __restrict__ Y,       // Y.shape       = [..., L, D]
    T* __restrict__ eps,     // eps.shape     = [..., L, 2] (a, s)
    T* __restrict__ row_max, // row_max.shape = [..., L]
    T* __restrict__ row_sum, // row_sum.shape = [..., L]
    const T beta,
    const int L)
{
    auto gp = cg::this_thread_block();
    auto tgp = cg::tiled_partition<tile>(gp);

    const int mid = blockIdx.y;
    const int mstart = mid * L * D;

    const int qid = blockIdx.x;
    const int tid = threadIdx.x;

    // accessing indicies for the computation of q * k
    const int idx = tgp.thread_rank();
    const int idy = tid / tile;
    const int sta = D / tile;

    __shared__ T q_mem[D]; // q_mem.shape = [D]
    __shared__ T a_mem[D / tile];

    float4* K4 = ((float4*) K) + mstart / 4;
    float4* q_mem4 = ((float4*) q_mem);
    const T norm = sqrtf(D);

    for (int q_row = 0; q_row < L; q_row += gridDim.x) {
        const int idq = q_row + qid;
        if (idq >= L) {break;}
        T alpha = eps[mid * L * 2];
        T shift = eps[mid * L * 2 + 1];
        T center = T(idq) + shift;
        T bounds = soft_bounds(alpha, beta);

        int high = min((int) (center + bounds + 0.5), L);
        int low = max((int) (center - bounds - 0.5), 0);

        T y_mem = 0;
        q_mem[tid] = Q[mstart + idq * D + tid];
        __syncthreads();
        
        T max = -1.0 / 0.0;
        T sum = 0;
    

        for (int r = low; r < high; r += sta) {
            int col_a = r + idy;
            T a_val = 0;
            if (col_a < high) {
                #pragma unroll
                for (int i = 0; i < D / 4; i += tile) {
                    a_val += ut::mul_accum(q_mem4[i + idx], K4[col_a * (D / 4) + i + idx]);
                }
            }
            a_val = ut::reduce_sum_tile_shuffle<tile, T>(tgp, a_val);
            if (idx == 0) {
                a_mem[idy] = a_val * soft_gatev1((T) (col_a) - center, alpha, beta);
            }
            __syncthreads();
            #pragma unroll
            for (int i = 0; i < sta; i++) {
                if (i + r < high) {
                    T a = a_mem[i] / norm;
                    T cur_max = fmaxf(max, a);
                    T scale = expf(max - cur_max);
                    a = expf(a - cur_max);
                    y_mem = y_mem * scale + a * V[mstart + (r + i) * D + tid];   
                    sum = scale * sum + a;
                    max = cur_max;
                }
            }
            __syncthreads();
        }

        Y[mstart + idq * D + tid] = y_mem / sum;
        if (tid == 0)
            row_max[mid * L + idq] = max;
        if (tid == 1)
            row_sum[mid * L + idq] = sum;
    }
}


auto like_tensor(const torch::Tensor &a) {
    auto opt = torch::TensorOptions().device(a.device()).dtype(a.dtype());
    return opt;
}

inline int n_elements(const torch::Tensor &a) {
    return a.size(0) * a.stride(0);
}

std::vector<torch::Tensor> attn(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor epsilion,
    T beta,
    int block_div)
{
    // try to be dimension agnostic
    auto sizes = Q.sizes();
    sizes = sizes.slice(0, sizes.size()-1);
    const int L = Q.size(-2);
    const int D = Q.size(-1);
    const int matrices = n_elements(Q) / (L * D);

    auto opt = like_tensor(Q);
    auto Y = torch::empty_like(Q);
    auto row_max = torch::empty(sizes, opt);
    auto row_sum = torch::empty(sizes, opt);
    

    const dim3 threads(512, 1, 1);
    const dim3 blocks(L / block_div, matrices, 1);
    attn_impl_kernel<4, 512><<<blocks, threads>>>(
        Q.data_ptr<T>(),
        K.data_ptr<T>(),
        V.data_ptr<T>(),
        Y.data_ptr<T>(),
        epsilion.data_ptr<T>(),
        row_max.data_ptr<T>(),
        row_sum.data_ptr<T>(),
        beta,
        L
    );
    return {Y, row_max, row_sum};
}


#ifndef PyExtension

int main() {
    auto epsilion = torch::ones({2, 1024, 2}).cuda();
    auto q = torch::rand({2, 1024, 512}).cuda();
    auto k = torch::rand_like(q);
    auto v = torch::rand_like(q);
    auto y = attn(q, k, v, epsilion, 1, 1);
    ut::cudaLastError();
}

#else

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attn", attn, "(CUDA)");
}
#endif
