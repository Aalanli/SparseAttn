#pragma once
#ifdef IS_EXECUTABLE
#include "ops.cuh"
#else
#include "utils/ops.cuh"
#endif

#define FULLMASK 0xffffffff

// version 1 of the kernel accesses global memory four times, twice for computing the dot-product between
// y and eps, and twice for computing the derivative of softmax and gating functions
template <typename T, int threads>
__global__ void d_gated_softmax_v1_kernel(
    const T* __restrict__ grad, // grad.shape = [B, H, L, D]
    const T* __restrict__ y,    // Y.shape = [B, H, L, D]
    const T* __restrict__ eps,  // eps.shape = [B, H, L, 2 (alpha, shift)]
    T* __restrict__ dy_dx,      // out.shape = [B, H, L, D]
    T* __restrict__ dy_de,      // dy_de.shape = [B, H, L, 2 (dy_da, dy_ds)]
    const T beta,
    const int L,
    const int D)
{
    constexpr int warps = threads / 32;
    const int row_id = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    int warp_rank = tid % 32;

    const T alpha = eps[row_id * 2];
    const T shift = eps[row_id * 2 + 1];

    T dot;

    // shift pointers to the correct row, may not be necessarily more efficient
    const T* y_row = y + row_id * D;
    const T* grad_row = grad + row_id * D;
    for (int i = tid; i < L; i += threads) {
        T yi = y_row[i];
        T gi = grad_row[i];
        dot += yi * gi;
    }
    // now perform block-wise reduction of all
    __shared__ T s_pool[warps * 2];
    {
        // first reduce across each warp
        dot = ops::reduce_sum_shfl_down<32>(dot, FULLMASK);
        if (warp_rank == 0) {
            s_pool[warp_rank] = dot;
        }
        __syncthreads();
    
        unsigned mask = __ballot_sync(FULLMASK, tid < warps);
        if (tid < warps) {
            // now reduce across the mem-pool
            dot = s_pool[tid];
            dot = ops::reduce_sum_shfl_down<warps>(dot, mask);
            if (tid == 0) {
                s_pool[0] = dot;
            }
        }
        // now dot-product is accumulated in s_pool[0]
    }
    // broadcast back to all threads the results
    __syncthreads();
    dot = s_pool[0];
    T dy_de_[2] = {};
    // calculate dy_dx and dy_de
    {
        T* dy_dx_row = dy_dx + row_id * D;
        for (int i = tid; i < L; i += threads) {
            T yi = y_row[i];
            T gi = grad_row[i];
            T dy_dx_temp = yi * (gi - dot);
            dy_dx_row[i] = dy_dx_temp;
            T w = ops::softgate(T(i) - shift, alpha, beta);
            T temp = (1 - w) * dy_dx_temp;
            dy_de_[0] += temp;
            dy_de_[1] += temp * tanh(beta * T(i));
        }
    }
    // now accumulate the partial derivatives of dy_de
    {
        ops::reduce_sum_shfl_down_vec<32, T, 2>(dy_de_, FULLMASK);
        if (warp_rank == 0) {
            s_pool[warp_rank] = dy_de_[0];
            s_pool[warps + warp_rank] = dy_de_[1];
        }
        __syncthreads();
    
        unsigned mask = __ballot_sync(FULLMASK, tid < warps);
        if (tid < warps) {
            // now reduce across the mem-pool
            dy_de_[0] = s_pool[tid];
            dy_de_[1] = s_pool[warps + tid];
            ops::reduce_sum_shfl_down_vec<warps, T, 2>(dy_de_, mask);
            if (tid == 0) {
                dy_de[row_id * 2] = dy_de_[0];
                dy_de[row_id * 2 + 1] = dy_de_[1];
            }
        }
    }
}

template <typename T>
struct PartialAccumulations {
    T dot_1;
    T dy_da_1;
    T dy_da_2;
    T dy_ds_1;
    T dy_ds_2;

    __device__ __forceinline__ void zero_() {
        dot_1 = 0;
        dy_da_1 = 0;
        dy_da_2 = 0;
        dy_ds_1 = 0;
        dy_ds_2 = 0;
    }

    template <int warp_sz>
    __device__ __forceinline__ void warp_shfl_down_sum(int mask) {
        #pragma unroll
        for (int i = warp_sz / 2; i > 0; i /= 2) {
            dot_1   += __shfl_down_sync(mask, dot_1,   i, warp_sz);
            dy_da_1 += __shfl_down_sync(mask, dy_da_1, i, warp_sz);
            dy_da_2 += __shfl_down_sync(mask, dy_da_2, i, warp_sz);
            dy_ds_1 += __shfl_down_sync(mask, dy_ds_1, i, warp_sz);
            dy_ds_2 += __shfl_down_sync(mask, dy_ds_2, i, warp_sz);
        }
    }
};

// version 2 performs the dot-product between the row of grad and y to obtain dy_dx,
// while simultaineously calculating the sum-reduction of dy_de 
template <typename T, int threads>
__global__ void d_gated_softmax_v2_kernel(
    const T* __restrict__ grad, // grad.shape = [B, H, L, D]
    const T* __restrict__ y,    // Y.shape = [B, H, L, D]
    const T* __restrict__ eps,  // eps.shape = [B, H, L, 2 (alpha, shift)]
    T* __restrict__ dy_dx,      // out.shape = [B, H, L, D]
    T* __restrict__ dy_de,      // dy_de.shape = [B, H, L, 2 (dy_da, dy_ds)]
    const T beta,
    const int L,
    const int D)
{
    constexpr int warps = threads / 32;
    const int row_id = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int warp_rank = tid % 32;

    const T alpha = eps[row_id * 2];
    const T shift = eps[row_id * 2 + 1];

    PartialAccumulations<T> dot;
    dot.zero_();

    // shift pointers to the correct row, may not be necessarily more efficient
    const T* y_row = y + row_id * D;
    const T* grad_row = grad + row_id * D;
    // calculate the dot product and dy_de simultaineously
    for (int i = tid; i < L; i += threads) {
        T yi = y_row[i];
        T gi = grad_row[i];
        dot.dot_1 += yi * gi;

        T w = ops::softgate(T(i) - shift, alpha, beta);
        T temp = (1 - w) * yi;
        dot.dy_da_1 += temp * gi;
        dot.dy_da_2 += temp;
        temp *= tanh(beta * T(i));
        dot.dy_ds_1 += temp * gi;
        dot.dy_ds_2 += temp;
    }
    // now perform block-wise reduction of elements in dot
    __shared__ PartialAccumulations<T> s_pool[warps];
    __shared__ T dot_product;
    {
        // first reduce across each warp
        dot.warp_shfl_down_sum<32>(FULLMASK);
        if (warp_rank == 0) {
            s_pool[warp_rank] = dot;
        }
        __syncthreads();
    
        unsigned mask = __ballot_sync(FULLMASK, tid < warps);
        if (tid < warps) {
            // now reduce across the mem-pool
            dot = s_pool[tid];
            dot.warp_shfl_down_sum<warps>(mask);
            if (tid == 0) {
                dot_product = dot.dot_1;
            } else if (tid == 1) {
                dy_de[row_id * 2] = dot.dy_da_1 - dot.dy_da_2 * dot.dot_1;
            } else if (tid == 2) {
                dy_de[row_id * 2 + 1] = dot.dy_ds_1 - dot.dy_ds_2 * dot.dot_1;
            }
        }
        // now dot-product is accumulated in dot_product
    }
    __syncthreads();
    // broadcast back to all threads the results
    dot.dot_1 = dot_product;
    // calculate dy_dx and dy_de
    {
        T* dy_dx_row = dy_dx + row_id * D;
        for (int i = tid; i < L; i += threads) {
            T yi = y_row[i];
            T gi = grad_row[i];
            dy_dx_row[i] = yi * (gi - dot.dot_1);
        }
    }
}

// version 3 uses shared memory to accumulate dy_dx, cutting down access to global memory
// by a factor of 2
template <typename T, int threads>
__global__ void d_gated_softmax_v3_kernel(
    const T* __restrict__ grad, // grad.shape = [B, H, L, D]
    const T* __restrict__ y,    // Y.shape = [B, H, L, D]
    const T* __restrict__ eps,  // eps.shape = [B, H, L, 2 (alpha, shift)]
    T* __restrict__ dy_dx,      // out.shape = [B, H, L, D]
    T* __restrict__ dy_de,      // dy_de.shape = [B, H, L, 2 (dy_da, dy_ds)]
    const T beta,
    const int L,
    const int D)
{
    constexpr int warps = threads / 32;
    const int row_id = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int warp_rank = tid % 32;

    const T alpha = eps[row_id * 2];
    const T shift = eps[row_id * 2 + 1];

    PartialAccumulations<T> dot;
    dot.zero_();

    extern __shared__ char temp_mem[]; // smem.shape = [2 * L]
    T* smem = (T*) temp_mem;
    // shift pointers to the correct row, may not be necessarily more efficient
    const T* y_row = y + row_id * D;
    const T* grad_row = grad + row_id * D;
    // calculate the dot product and dy_de simultaineously
    for (int i = tid; i < L; i += threads) {
        T yi = y_row[i];
        T gi = grad_row[i];

        T temp = yi * gi;
        smem[i * 2] = temp;
        smem[i * 2 + 1] = yi;
        dot.dot_1 += temp;

        T w = ops::softgate(T(i) - shift, alpha, beta);
        temp = (1 - w) * yi;
        dot.dy_da_1 += temp * gi;
        dot.dy_da_2 += temp;
        temp *= tanh(beta * T(i));
        dot.dy_ds_1 += temp * gi;
        dot.dy_ds_2 += temp;
    }
    // now perform block-wise reduction of elements in dot
    __shared__ PartialAccumulations<T> s_pool[warps];
    __shared__ T dot_product;
    {
        // first reduce across each warp
        dot.warp_shfl_down_sum<32>(FULLMASK);
        if (warp_rank == 0) {
            s_pool[warp_rank] = dot;
        }
        __syncthreads();
    
        unsigned mask = __ballot_sync(FULLMASK, tid < warps);
        if (tid < warps) {
            // now reduce across the mem-pool
            dot = s_pool[tid];
            dot.warp_shfl_down_sum<warps>(mask);
            if (tid == 0) {
                dot_product = dot.dot_1;
            } else if (tid == 1) {
                dy_de[row_id * 2] = dot.dy_da_1 - dot.dy_da_2 * dot.dot_1;
            } else if (tid == 2) {
                dy_de[row_id * 2 + 1] = dot.dy_ds_1 - dot.dy_ds_2 * dot.dot_1;
            }
        }
        // now dot-product is accumulated in dot_product
    }
    __syncthreads();
    // broadcast back to all threads the results
    dot.dot_1 = dot_product;
    // calculate dy_dx and dy_de
    {
        T* dy_dx_row = dy_dx + row_id * D;
        for (int i = tid; i < L; i += threads) {
            dy_dx_row[i] = smem[i * 2] - smem[i * 2 + i] * dot.dot_1;
        }
    }
}
