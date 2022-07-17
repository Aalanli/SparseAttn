cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(main LANGUAGES CXX CUDA)

set(CMAKE_CUDA_FLAGS "-arch=sm_75 -lineinfo -Xcompiler -rdynamic")
set(CMAKE_PREFIX_PATH "/home/allan/Documents/C++/libtorch/")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(main "")
add_subdirectory(utils)
add_subdirectory(dense)

set_property(TARGET main PROPERTY CXX_STANDARD 14)
set_property(TARGET main PROPERTY CUDA_SEPARABLE_COMPILATION ON)
