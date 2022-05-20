#define PyExtension
#include <torch/types.h>
#ifdef PyExtension
#include <torch/python.h>
#else
#include <iostream>
#endif
#include <vector>
#include "utils.cuh"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

using T = float;

template <int N>
using Tensor = torch::PackedTensorAccessor32<T, N, torch::RestrictPtrTraits>;

__device__ __forceinline__ float mul_accum(const float4 &a, const float4 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}
__device__ __forceinline__ float mul_accum(const float4 a, const float b) {
    return a.x * b + a.y * b + a.z * b + a.w * b;
} 
__device__ __forceinline__ float mul_accum(const float a, const float4 b) {
    return mul_accum(b, a);
}

template <int tile_sz>
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


template <int threads, int tile, int D>
__global__ void simple_attn_v1_t(
    const T* __restrict__ Q, // Q.shape = [L, D]
    const T* __restrict__ K, // K.shape = [L, D]
    const T* __restrict__ V, // V.shape = [L, D]
    T* __restrict__ Y, // Y.shape = [L, D]
    const int L)
{
    auto gp = cg::this_thread_block();
    auto tgp = cg::tiled_partition<tile>(gp);

    const int qid = blockIdx.x;
    const int tid = threadIdx.x;

    // accessing indicies for the computation of q * k
    const int idx = tgp.thread_rank();
    const int idy = tid / tile;
    const int sta = threads / tile;

    const int idc = tid % D;
    const int idr = tid / D;
    const int sty = threads / D;

    __shared__ T q_mems[D]; // q_mem.shape = [D]
    __shared__ T a_mem[threads / tile];
    __shared__ T y_mem[threads];

    if (tid < D) {q_mems[tid] = Q[qid * D + tid];}
    y_mem[tid] = 0;
    __syncthreads();

    float4* K4 = (float4*) K;
    float4* q_mem4 = (float4*) q_mems;
    
    for (int r = 0; r < L; r += sta) {
        int col_a = r + idy;
        T a_val = 0;
        if (col_a < L) {
            #pragma unroll
            for (int i = 0; i < D / 4; i += tile) {
                a_val += mul_accum(q_mem4[i + idx], K4[col_a * (D / 4) + i + idx]);
            }
        }
        a_val = reduce_sum_tile_shuffle<tile>(tgp, a_val);
        if (idx == 0) {
            a_mem[idy] = a_val;
        }
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < sta; i += sty) {
            int ii = i + idr;
            if (ii + r < L)
                y_mem[idr * D + idc] += a_mem[ii] * V[(r + ii) * D + idc];   
        }
        __syncthreads();
    }

    // transpose idc and idr
    const int idct = tid % sty;
    const int idrt = tid / sty;
    
    unsigned mask = __ballot_sync(0xffffffff,(tid >= idrt * sty && tid < (idrt + 1) * sty));
    #pragma unroll
    for (int i = sty / 2; i > 0; i /= 2) {
        y_mem[idct * D + idrt] += __shfl_down_sync(mask, y_mem[idct * D + idrt], i, sty);
    }
    __syncthreads();
    if (tid < D) {Y[qid * D + tid] = y_mem[tid];}
}

// D is the number of threads
template <int tile, int D>
__global__ void simple_attn_v2_t(
    const T* __restrict__ Q, // Q.shape = [L, D]
    const T* __restrict__ K, // K.shape = [L, D]
    const T* __restrict__ V, // V.shape = [L, D]
    T* __restrict__ Y, // Y.shape = [L, D]
    const int L,
    const int halt)
{
    auto gp = cg::this_thread_block();
    auto tgp = cg::tiled_partition<tile>(gp);

    const int qid = blockIdx.x;
    const int tid = threadIdx.x;

    // accessing indicies for the computation of q * k
    const int idx = tgp.thread_rank();
    const int idy = tid / tile;
    const int sta = D / tile;

    __shared__ T q_mem[D]; // q_mem.shape = [D]
    __shared__ T a_mem[D / tile];

    float4* K4 = (float4*) K;
    float4* q_mem4 = (float4*) q_mem;
    const T norm = sqrtf(D);

    for (int q_row = 0; q_row < L; q_row += gridDim.x) {
        const int idq = q_row + qid;
        if (idq >= L) {break;};
        T y_mem = 0;
        q_mem[tid] = Q[idq * D + tid];
        __syncthreads();
        
        T max = -1.0 / 0.0;
        T sum = 0;

        for (int r = 0; r < halt; r += sta) {
            int col_a = r + idy;
            T a_val = 0;
            if (col_a < halt) {
                #pragma unroll
                for (int i = 0; i < D / 4; i += tile) {
                    a_val += mul_accum(q_mem4[i + idx], K4[col_a * (D / 4) + i + idx]);
                }
            }
            a_val = reduce_sum_tile_shuffle<tile>(tgp, a_val);
            if (idx == 0) {
                a_mem[idy] = a_val;
            }
            __syncthreads();
            #pragma unroll
            for (int i = 0; i < sta; i++) {
                if (i + r < halt) {
                    T a = a_mem[i] / norm;
                    T cur_max = fmaxf(max, a);
                    T scale = expf(max - cur_max);
                    a = expf(a - cur_max);
                    y_mem = y_mem * scale + a * V[(r + i) * D + tid];   
                    sum = scale * sum + a;
                    max = cur_max;
                }
            }
            __syncthreads();
        }

        Y[idq * D + tid] = y_mem / sum;
    }
}

// unrolls the last step
// unfortunately, this does not work well for lower halts,
// not enough to justify the tiny performance gain for larger halts
template <int tile, int D>
__global__ void simple_attn_v3_t(
    const T* __restrict__ Q, // Q.shape = [L, D]
    const T* __restrict__ K, // K.shape = [L, D]
    const T* __restrict__ V, // V.shape = [L, D]
    T* __restrict__ Y, // Y.shape = [L, D]
    const int L,
    const int halt)
{
    auto gp = cg::this_thread_block();
    auto tgp = cg::tiled_partition<tile>(gp);

    const int qid = blockIdx.x;
    const int tid = threadIdx.x;

    // accessing indicies for the computation of q * k
    const int idx = tgp.thread_rank();
    const int idy = tid / tile;
    const int sta = D / tile;

    __shared__ T q_mem[D]; // q_mem.shape = [D]
    __shared__ T a_mem[D / tile];

    float4* K4 = (float4*) K;
    float4* q_mem4 = (float4*) q_mem;
    const T norm = sqrtf(D);

    for (int q_row = 0; q_row < L; q_row += gridDim.x) {
        const int idq = q_row + qid;
        if (idq >= L) {break;} // okay to break here since all threads do the same thing
        T y_mem = 0;
        q_mem[tid] = Q[idq * D + tid];
        __syncthreads();
        
        T max = -1.0 / 0.0;
        T sum = 0;

        // unroll to remove bounds checks
        for (int r = 0; r < halt - sta; r += sta) {
            int col_a = r + idy;
            T a_val = 0;
            #pragma unroll
            for (int i = 0; i < D / 4; i += tile) {
                a_val += mul_accum(q_mem4[i + idx], K4[col_a * (D / 4) + i + idx]);
            }
            
            a_val = reduce_sum_tile_shuffle<tile>(tgp, a_val);
            if (idx == 0) {
                a_mem[idy] = a_val;
            }
            __syncthreads();
            #pragma unroll
            for (int i = 0; i < sta; i++) {
                T a = a_mem[i] / norm;
                T cur_max = fmaxf(max, a);
                T scale = expf(max - cur_max);
                a = expf(a - cur_max);
                y_mem = y_mem * scale + a * V[(r + i) * D + tid];   
                sum = scale * sum + a;
                max = cur_max;
            }
            __syncthreads();
        }
        int r = halt - sta;
        int col_a = r + idy;
        T a_val = 0;
        if (col_a < halt) {
            #pragma unroll
            for (int i = 0; i < D / 4; i += tile) {
                a_val += mul_accum(q_mem4[i + idx], K4[col_a * (D / 4) + i + idx]);
            }
        }
        a_val = reduce_sum_tile_shuffle<tile>(tgp, a_val);
        if (idx == 0) {
            a_mem[idy] = a_val;
        }
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < sta; i++) {
            if (i - sta < 0) {
                T a = a_mem[i] / norm;
                T cur_max = fmaxf(max, a);
                T scale = expf(max - cur_max);
                a = expf(a - cur_max);
                y_mem = y_mem * scale + a * V[(r + i) * D + tid];   
                sum = scale * sum + a;
                max = cur_max;
            }
        }
        __syncthreads();

         Y[idq * D + tid] = y_mem / sum;
    }
}

auto like_tensor(const torch::Tensor &a) {
    auto opt = torch::TensorOptions().device(a.device()).dtype(a.dtype());
    return opt;
}


torch::Tensor attnv1_t(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V)
{
    const int L = Q.size(0);
    const int D = Q.size(1);
    auto opt = like_tensor(Q);
    auto Y = torch::empty_like(Q);

    const dim3 threads(1024, 1, 1);
    const dim3 blocks(L, 1, 1);
    simple_attn_v1_t<1024, 4, 512><<<blocks, threads>>>(
        Q.data_ptr<T>(),
        K.data_ptr<T>(),
        V.data_ptr<T>(),
        Y.data_ptr<T>(),
        L
    );
    return Y;
}

torch::Tensor attnv2_t(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    int block_div,
    int halt)
{
    const int L = Q.size(0);
    const int D = Q.size(1);
    auto opt = like_tensor(Q);
    auto Y = torch::empty_like(Q);

    if (halt < 1) {
        halt = L;
    } 

    const dim3 threads(512, 1, 1);
    const dim3 blocks(L / block_div, 1, 1);
    simple_attn_v2_t<4, 512><<<blocks, threads>>>(
        Q.data_ptr<T>(),
        K.data_ptr<T>(),
        V.data_ptr<T>(),
        Y.data_ptr<T>(),
        L, halt
    );
    return Y;
}

torch::Tensor attnv3_t(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    int block_div,
    int halt)
{
    const int L = Q.size(0);
    const int D = Q.size(1);
    auto opt = like_tensor(Q);
    auto Y = torch::empty_like(Q);

    if (halt < 1) {
        halt = L;
    } 

    const dim3 threads(512, 1, 1);
    const dim3 blocks(L / block_div, 1, 1);
    simple_attn_v3_t<4, 512><<<blocks, threads>>>(
        Q.data_ptr<T>(),
        K.data_ptr<T>(),
        V.data_ptr<T>(),
        Y.data_ptr<T>(),
        L, halt
    );
    return Y;
}

#ifndef PyExtension

int main() {
    auto q = torch::rand({1024, 512}).cuda();
    auto k = torch::rand_like(q);
    auto v = torch::rand_like(q);
    auto y = attnv3_t(q, k, v, 2, -1);
    ut::cudaLastError();
}

#else

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attnv1_t", attnv1_t, "(CUDA)");
    m.def("attnv2_t", attnv2_t, "(CUDA)");
    m.def("attnv3_t", attnv3_t, "(CUDA)");
}
#endif
