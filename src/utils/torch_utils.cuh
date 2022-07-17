/*
pytorch convenience functions, for separation between functions which includes torch, since
building with it is extremely slow
*/

#pragma once
#include <torch/types.h>

namespace utils {
    auto like_tensor(const torch::Tensor &a) {
        auto opt = torch::TensorOptions().device(a.device()).dtype(a.dtype());
        return opt;
    }

    inline int n_elements(const torch::Tensor &a) {
        return a.size(0) * a.stride(0);
    }

}