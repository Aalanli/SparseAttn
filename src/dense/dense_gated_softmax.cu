/*
Implements the dense version of back propogagation for the gated softmax function
*/

//#define PyExtension
#include <torch/types.h>
#ifdef PyExtension
#include <torch/python.h>
#else
#include <iostream>
#endif
#include <math.h>
#include <vector>
#include <boost/mpl/list.hpp>
#include <boost/mpl/integral_c.hpp>

#include "func_constructor.hpp"
#include "utils.cuh"
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


template <typename T>
size_t hash_arg(size_t base) {
    char id = *typeid(T).name();
    if (sizeof(size_t) == sizeof(long long)) {
        return base | (size_t(id) << 60);
    } else {
        return base | (int(id) << 28);
    }
}


template <typename T, typename Threads_t>
struct DGatedSoftmaxKernelV1 {
    static const int threads = Threads_t::value;
    static void fn(BackwardArgs args) {
        ParamPack<T> v(args);
        d_gated_softmax_v1_kernel<T, threads><<<v.blocks, threads>>>(
            v.grad, v.y, v.eps, v.dy_dx, v.dy_de, v.beta, v.L, v.D);
    }
    static size_t get_id() {
        return hash_arg<T>(threads);
    }
};

template <typename T, typename Threads_t>
struct DGatedSoftmaxKernelV2 {
    static const int threads = Threads_t::value;
    static void fn(BackwardArgs args) {
        ParamPack<T> v(args);
        d_gated_softmax_v2_kernel<T, threads><<<v.blocks, threads>>>(
            v.grad, v.y, v.eps, v.dy_dx, v.dy_de, v.beta, v.L, v.D);
    }
    static size_t get_id() {
        return hash_arg<T>(threads + 1024);
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
    static size_t get_id() {
        return hash_arg<T>(threads + 1024 * 2);
    }
};



using FnParams = mpl::list<
    //mpl::list<float, mpl::integral_c<int, 1024>>,
    //mpl::list<float, mpl::integral_c<int, 512 >>,
    //mpl::list<float, mpl::integral_c<int, 256 >>,
    //mpl::list<float, mpl::integral_c<int, 128 >>,
    //mpl::list<float, mpl::integral_c<int, 64  >>,
    //mpl::list<float, mpl::integral_c<int, 32  >>,
    //mpl::list<double, mpl::integral_c<int, 1024>>,
    //mpl::list<double, mpl::integral_c<int, 512 >>,
    //mpl::list<double, mpl::integral_c<int, 256 >>,
    //mpl::list<double, mpl::integral_c<int, 128 >>,
    //mpl::list<double, mpl::integral_c<int, 64  >>,
    mpl::list<double, mpl::integral_c<int, 32  >>
>;


auto build_kernels() {
    auto fn_map = fn_builder::FnBuilder<FnParams, DGatedSoftmaxKernelV1>::build_fn();
    fn_builder::FnBuilder<FnParams, DGatedSoftmaxKernelV2>::build_fn(fn_map);
    fn_builder::FnBuilder<FnParams, DGatedSoftmaxKernelV3>::build_fn(fn_map);
    return fn_map;
}
auto fn_map = build_kernels();


std::vector<torch::Tensor> d_gated_dense_softmax_generic(
    torch::Tensor grad, torch::Tensor y, torch::Tensor eps, double beta, int version) 
{
    auto dy_dx = torch::empty_like(y);
    auto dy_de = torch::empty_like(eps);

    BackwardArgs args = {&grad, &y, &eps, &dy_dx, &dy_de, beta};


    return {dy_dx, dy_de};
}

int main() {
    auto grad = torch::randn({4, 8, 512, 512}).cuda();
    auto y = torch::randn_like(grad);
    auto eps = torch::randn({4, 8, 512, 2}).cuda();

    d_gated_dense_softmax_generic(grad, y, eps, 1.0, 0);
    utils::cudaLastError();
    utils::print("done.");
}