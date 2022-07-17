//#define PyExtension
#include <torch/types.h>
#ifdef PyExtension
#include <torch/python.h>
#else
#include <iostream>
#endif
#include <math.h>
#include <vector>
#include <iostream>


void print() {
    std::cout << "\n";
}

template <typename T, typename... Args>
void print(T val, Args... args) {
    std::cout << val << " ";
    print(args...);
}


using T = float;

const T limEps = 0.0001;
const T CBound = log(((1 - limEps) / limEps));

template <int N>
using Tensor = torch::TensorAccessor<T, N>;

T soft_gatev1(T x, T a, T b) {
    return 1 / (1 + exp(b * (x - a)) + exp(-b * (x + a)));
}

T soft_bounds(T a, T b) {
    return CBound / b + a;
}

T dot(const T* __restrict_arr a, const T* __restrict_arr b, const int size) {
    T sum = 0;
    for (int i = 0; i < size; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

void soft_attn_cpu_kernel(
    const T* __restrict_arr Q,
    const T* __restrict_arr K,
    const T* __restrict_arr V,
    const T* __restrict_arr eps,
    T* __restrict_arr Y,
    T* __restrict_arr row_max,
    T* __restrict_arr row_sum,
    const float beta,
    const int matrices,
    const int L,
    const int D)
{
    const int mat_stride = L * D;
    for (int m = 0; m < matrices; m++) {
        const int mstart = m * mat_stride;
        for (int idq = 0; idq < L; idq++) {
            T alpha = eps[m * L + idq];
            T shift = eps[m * L + idq + 1];
            T center = T(idq) + shift;
            T bounds = soft_bounds(alpha, beta);

            int high = std::min(int(center + bounds + 0.5), L);
            int low  = std::max(int(center - bounds - 0.5), 0);

            T max = -1.0 / 0.0;
            T sum = 0;

            for (int idk = low; idk < high; idk++) {
                T a = dot(Q + mstart + idq * D, K + mstart + idk * D, D);
                T curmax = std::max(max, a);
                T scale = exp(max - curmax);
                a = soft_gatev1(T(idk) - center, alpha, beta);
                a = exp(a - curmax);
                for (int i = 0; i < D; i++) {
                    Y[mstart + idq * D + i] = Y[mstart + idq * D + i] * scale + a * V[mstart + idk * D + i];
                }
                sum = sum * scale + a;
                max = curmax;
            }
        }
    }
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
    auto Y = torch::empty_like(Q);

    auto row_max = torch::empty(sizes);
    auto row_sum = torch::empty(sizes);
    
    soft_attn_cpu_kernel(
        Q.data_ptr<T>(),
        K.data_ptr<T>(),
        V.data_ptr<T>(),
        epsilion.data_ptr<T>(),
        Y.data_ptr<T>(),
        row_max.data_ptr<T>(),
        row_sum.data_ptr<T>(),
        beta,
        matrices,
        L, D
    );
    return {Y, row_max, row_sum};
}


int main() {
    auto epsilion = torch::ones({2, 1024, 2});
    auto q = torch::rand({2, 1024, 512});
    auto k = torch::rand_like(q);
    auto v = torch::rand_like(q);
    auto y = attn(q, k, v, epsilion, 1, 1);
}