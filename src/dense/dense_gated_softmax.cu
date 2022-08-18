/*
Implements the dense version of back propogagation for the gated softmax function
*/
#include <cuda_profiler_api.h>

#include <math.h>
#include <vector>
#include <boost/mpl/list.hpp>
#include <boost/mpl/int.hpp>


#include <torch/types.h>
#ifdef IS_EXECUTABLE
#include <iostream>
#include "func_constructor.hpp"
#include "utils.cuh"
#include "torch_utils.cuh"

#else
#include <torch/python.h>
#include "utils/func_constructor.hpp"
#include "utils/utils.cuh"
#include "utils/torch_utils.cuh"
#endif

#include "kernels.cuh"

namespace mpl = boost::mpl;

using Tensor = torch::Tensor;
struct BackwardArgs {
    torch::Tensor* grad;
    torch::Tensor* y;
    torch::Tensor* eps;
    torch::Tensor* dy_dx;
    torch::Tensor* dy_de; 
    double beta;
};

template <typename T>
struct ParamPack {
    T* grad; 
    T* y;
    T* eps;
    T* dy_dx;
    T* dy_de;
    T beta;
    int L;
    int D;

    dim3 blocks;

    ParamPack(BackwardArgs args) {
        this->beta = (T) args.beta;

        int elems = args.grad->stride(0) * args.grad->size(0);
        D = args.grad->size(-1);
        L = args.grad->size(-2);

        blocks = dim3(elems / D, 1, 1);

        grad  = args.grad->data_ptr<T>();
        y     = args.y->data_ptr<T>();
        eps   = args.eps->data_ptr<T>();
        dy_dx = args.dy_dx->data_ptr<T>();
        dy_de = args.dy_de->data_ptr<T>();
    }
};



template <typename T, typename Threads_t>
struct SoftmaxRefKernel {
    static const int threads = Threads_t::value;
    static void fn(torch::Tensor &x, torch::Tensor &y, int blocks) {
        softmax_ref_kernel<T, threads><<<blocks, threads>>>(
            x.data_ptr<T>(),
            y.data_ptr<T>(),
            x.size(-1)
        );
    }
};


template <typename T, typename Threads_t>
struct DGatedSoftmaxKernelV1 {
    static const int threads = Threads_t::value;
    static void fn(BackwardArgs args) {
        ParamPack<T> v(args);
        cudaProfilerStart();
        d_gated_softmax_v1_kernel<T, threads><<<v.blocks, threads>>>(
            v.grad, v.y, v.eps, v.dy_dx, v.dy_de, v.beta, v.L, v.D);
        cudaProfilerStop();
        
    }
};

template <typename T, typename Threads_t>
struct DGatedSoftmaxKernelV2 {
    static const int threads = Threads_t::value;
    static void fn(BackwardArgs args) {
        ParamPack<T> v(args);
        cudaProfilerStart();
        d_gated_softmax_v2_kernel<T, threads><<<v.blocks, threads>>>(
            v.grad, v.y, v.eps, v.dy_dx, v.dy_de, v.beta, v.L, v.D);
        cudaProfilerStop();
    }
};

template <typename T, typename Threads_t>
struct DGatedSoftmaxKernelV3 {
    static const int threads = Threads_t::value;
    static void fn(BackwardArgs args) {
        ParamPack<T> v(args);
        d_gated_softmax_v3_kernel<T, threads><<<v.blocks, threads, sizeof(T) * v.L * 2>>>(
            v.grad, v.y, v.eps, v.dy_dx, v.dy_de, v.beta, v.L, v.D);
    }
};



using FnParams = mpl::list<
    mpl::list<float , mpl::int_<1024>>,
    mpl::list<float , mpl::int_<512 >>,
    mpl::list<float , mpl::int_<256 >>,
    mpl::list<float , mpl::int_<128 >>,
    mpl::list<float , mpl::int_<64  >>,
    mpl::list<float , mpl::int_<32  >>

>;


auto build_kernels() {
    auto fn_map = fn_builder::FnBuilder<FnParams, DGatedSoftmaxKernelV1>::build_fn(0);
    fn_builder::FnBuilder<FnParams, DGatedSoftmaxKernelV2>::build_fn(fn_map, 1);
    //fn_builder::FnBuilder<FnParams, DGatedSoftmaxKernelV3>::build_fn(fn_map, 2);
    return fn_map;
}
auto d_gated_attn_fn_map = build_kernels();
auto softmax_ref_fn_map = fn_builder::FnBuilder<FnParams, SoftmaxRefKernel>::build_fn(0);

torch::Tensor softmax_ref(torch::Tensor x) {
    auto y = torch::empty_like(x);
    int last_dim = x.size(-1);
    int rows = x.size(0) * x.stride(0) / last_dim;
    int warps = utils::reduction_switcher(last_dim, 1024);

    size_t fn_id = type_repr::construct_runtime_id(0, x.scalar_type(), warps * 32);
    auto fn_iter = softmax_ref_fn_map.find(fn_id);
    if (fn_iter == softmax_ref_fn_map.end()) {
        throw std::runtime_error("No suitable kernel found.");
    } else {
        fn_iter->second(x, y, rows);
    }
    return y;
}

std::vector<torch::Tensor> d_gated_dense_softmax_generic(
    torch::Tensor grad, torch::Tensor y, torch::Tensor eps, double beta, int version) 
{
    auto dy_dx = torch::empty_like(y);
    auto dy_de = torch::empty_like(eps);

    BackwardArgs args = {&grad, &y, &eps, &dy_dx, &dy_de, beta};
    int warps = utils::reduction_switcher(grad.size(-1), 1024);
    size_t fn_id = type_repr::construct_runtime_id(version, y.scalar_type(), warps * 32);
    auto fn_iter = d_gated_attn_fn_map.find(fn_id);
    if (fn_iter == d_gated_attn_fn_map.end()) {
        throw std::runtime_error("No suitable kernel found.");
    } else {
        fn_iter->second(args);
    }

    return {dy_dx, dy_de};
}

#ifdef IS_EXECUTABLE
int main() {
    auto grad = torch::randn({512, 1024}).cuda();
    auto y = torch::randn_like(grad);
    auto eps = torch::randn({512, 2}).cuda();
    auto y1 = d_gated_dense_softmax_generic(grad, y, eps, 2.0, 0);    
    auto y2 = d_gated_dense_softmax_generic(grad, y, eps, 2.0, 1);    
}
#else
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("d_gated_dense_softmax_cuda", &d_gated_dense_softmax_generic, "(CUDA)");
    m.def("softmax_ref", &softmax_ref, "(CUDA)");
}
#endif