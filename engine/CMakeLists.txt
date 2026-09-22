cmake_minimum_required(VERSION 3.16)
project(leaf_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# -O3 + AVX2 for release builds. -mavx2 -mfma enables the intrinsics the
# kernels will use later; -march=native is deliberately avoided so builds
# stay portable across machines that support AVX2 but not whatever the
# build machine specifically has.
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -mavx2 -mfma -DNDEBUG")

add_library(leaf_core
  src/graph_parser.cpp
  src/executor.cpp
)
target_include_directories(leaf_core PUBLIC include)

add_executable(leaf_parse_info
  src/main_parse_info.cpp
)
target_link_libraries(leaf_parse_info PRIVATE leaf_core)

add_executable(leaf_infer
  src/main_infer.cpp
)
target_link_libraries(leaf_infer PRIVATE leaf_core)

add_executable(leaf_bench
  src/main_bench.cpp
)
target_link_libraries(leaf_bench PRIVATE leaf_core)

enable_testing()
add_subdirectory(tests)

target_sources(leaf_core PRIVATE kernels/im2col.cpp kernels/gemm.cpp kernels/conv2d.cpp)
target_include_directories(leaf_core PUBLIC kernels)

# Quantized/fused kernel development library. Kept separate from leaf_core
# until the binary format carries INT8 tensors and calibration parameters.
add_library(leaf_optimized_kernels STATIC
  src/kernels/gemm.cpp
  src/kernels/conv.cpp
)
target_include_directories(leaf_optimized_kernels PUBLIC include)
if(MSVC)
  target_compile_options(leaf_optimized_kernels PRIVATE /O2 /arch:AVX2)
else()
  target_compile_options(leaf_optimized_kernels PRIVATE -O3 -mavx2 -mfma)
endif()

add_executable(leaf_native_tests ../tests/native/test_kernels.cpp)
target_link_libraries(leaf_native_tests PRIVATE leaf_optimized_kernels)
add_test(NAME leaf_native_tests COMMAND leaf_native_tests)

add_executable(leaf_kernel_bench ../benchmark/native_benchmark.cpp)
target_link_libraries(leaf_kernel_bench PRIVATE leaf_optimized_kernels)
