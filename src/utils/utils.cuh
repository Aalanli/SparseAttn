/*
General convenience functions
*/

#pragma once
#include <tuple>
#include <string>

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

}