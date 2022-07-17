cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

target_sources(main
    PUBLIC
        func_constructor.hpp
        torch_utils.cuh
        utils.cuh
        ops.cuh
)

find_package(Torch REQUIRED)
target_link_libraries(main PRIVATE ${TORCH_LIBRARIES})
target_include_directories(main PUBLIC ${CMAKE_CURRENT_LIST_DIR})

