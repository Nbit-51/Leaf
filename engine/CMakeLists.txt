cmake_minimum_required(VERSION 3.16)
project(leaf_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(LEAF_ENABLE_AVX2 "Build AVX2/FMA accelerated kernels" OFF)
option(LEAF_EXPERIMENTAL_VECTOR_GELU "Research-only vector GELU-new in multi-token ordinary FFNs" OFF)
option(LEAF_EXPERIMENTAL_GEMV_PAIR "Research-only paired-row FP32 GEMV" OFF)
option(LEAF_EXPERIMENTAL_ROW_REUSE "Research-only weight-row reuse across FP32 token panels" OFF)
option(LEAF_EXPERIMENTAL_ATTENTION_AVX2 "Research-only runtime-dispatched multi-token AVX2 attention" OFF)
if(NOT MSVC)
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
  if(LEAF_ENABLE_AVX2)
    string(APPEND CMAKE_CXX_FLAGS_RELEASE " -mavx2 -mfma")
  endif()
endif()

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
if(WIN32)
  target_link_libraries(leaf_bench PRIVATE psapi)
endif()

enable_testing()
add_subdirectory(tests)

target_sources(leaf_core PRIVATE kernels/im2col.cpp kernels/gemm.cpp kernels/conv2d.cpp)
target_include_directories(leaf_core PUBLIC kernels)

# Quantized/fused kernels used by the v2 graph executor.
add_library(leaf_optimized_kernels STATIC
  src/kernels/gemm.cpp
  src/kernels/conv.cpp
  src/kernels/transformer.cpp
  src/kv_cache.cpp
)
target_include_directories(leaf_optimized_kernels PUBLIC include)
target_link_libraries(leaf_core PRIVATE leaf_optimized_kernels)
if(MSVC AND LEAF_ENABLE_AVX2)
  target_compile_options(leaf_optimized_kernels PRIVATE /O2 /arch:AVX2)
elseif(NOT MSVC AND LEAF_ENABLE_AVX2)
  target_compile_options(leaf_optimized_kernels PRIVATE -O3 -mavx2 -mfma)
endif()

add_executable(leaf_native_tests ../tests/native/test_kernels.cpp)
target_link_libraries(leaf_native_tests PRIVATE leaf_optimized_kernels)
add_test(NAME leaf_native_tests COMMAND leaf_native_tests)

add_executable(leaf_kernel_bench ../benchmark/native_benchmark.cpp)
target_link_libraries(leaf_kernel_bench PRIVATE leaf_optimized_kernels)

add_executable(leaf_kv_session ../tests/native/test_kv_session.cpp)
target_link_libraries(leaf_kv_session PRIVATE leaf_core)
add_executable(leaf_dataset src/main_dataset.cpp)
target_link_libraries(leaf_dataset PRIVATE leaf_core)

# Portable executable with runtime-dispatched decoder vector kernels.
find_package(Threads REQUIRED)
add_executable(leaf_decoder src/decoder.cpp src/main_decoder.cpp
  src/kv_cache.cpp src/kernels/transformer.cpp)
target_include_directories(leaf_decoder PRIVATE include)
target_link_libraries(leaf_decoder PRIVATE Threads::Threads)
if(LEAF_EXPERIMENTAL_VECTOR_GELU)
  target_compile_definitions(leaf_decoder PRIVATE LEAF_EXPERIMENTAL_VECTOR_GELU=1)
endif()
if(LEAF_EXPERIMENTAL_GEMV_PAIR)
  target_compile_definitions(leaf_decoder PRIVATE LEAF_EXPERIMENTAL_GEMV_PAIR=1)
endif()
if(LEAF_EXPERIMENTAL_ROW_REUSE)
  target_compile_definitions(leaf_decoder PRIVATE LEAF_EXPERIMENTAL_ROW_REUSE=1)
endif()
if(LEAF_EXPERIMENTAL_ATTENTION_AVX2)
  target_compile_definitions(leaf_decoder PRIVATE LEAF_EXPERIMENTAL_ATTENTION_AVX2=1)
endif()
if(NOT MSVC AND LEAF_ENABLE_AVX2)
  target_compile_options(leaf_decoder PRIVATE -mno-avx2 -mno-fma)
endif()
if(WIN32)
  target_link_libraries(leaf_decoder PRIVATE psapi)
endif()

# Header-only token-panel primitive exercises portable scalar execution and
# runtime-dispatched AVX2/FMA without depending on whole-model artifacts.
add_executable(leaf_token_panel_tests ../tests/native/test_token_panel.cpp)
target_include_directories(leaf_token_panel_tests PRIVATE include)
if(NOT MSVC AND LEAF_ENABLE_AVX2)
  target_compile_options(leaf_token_panel_tests PRIVATE -mno-avx2 -mno-fma)
endif()
add_test(NAME leaf_token_panel_tests COMMAND leaf_token_panel_tests)

add_executable(leaf_gelu_tests ../tests/native/test_gelu.cpp)
target_include_directories(leaf_gelu_tests PRIVATE include)
if(NOT MSVC AND LEAF_ENABLE_AVX2)
  target_compile_options(leaf_gelu_tests PRIVATE -mno-avx2 -mno-fma)
endif()
add_test(NAME leaf_gelu_tests COMMAND leaf_gelu_tests)

add_executable(leaf_attention_vector_tests ../tests/native/test_attention_vector.cpp src/kernels/transformer.cpp)
target_include_directories(leaf_attention_vector_tests PRIVATE include)
add_test(NAME leaf_attention_vector_tests COMMAND leaf_attention_vector_tests)
add_executable(leaf_decoder_operator_bench ../benchmark/decoder_operator_benchmark.cpp src/kernels/transformer.cpp)
target_include_directories(leaf_decoder_operator_bench PRIVATE include)
if(NOT MSVC AND LEAF_ENABLE_AVX2)
  target_compile_options(leaf_attention_vector_tests PRIVATE -mno-avx2 -mno-fma)
  target_compile_options(leaf_decoder_operator_bench PRIVATE -mno-avx2 -mno-fma)
endif()

add_executable(leaf_token_panel_bench ../benchmark/token_panel_benchmark.cpp)
target_include_directories(leaf_token_panel_bench PRIVATE include)
if(NOT MSVC AND LEAF_ENABLE_AVX2)
  target_compile_options(leaf_token_panel_bench PRIVATE -mno-avx2 -mno-fma)
endif()

# Link the real decoder dot kernels into research checks via test-only access.
foreach(target leaf_decode_gemv_tests leaf_decode_gemv_bench)
  if(target STREQUAL "leaf_decode_gemv_tests")
    set(gemv_source ../tests/native/test_decode_gemv.cpp)
  else()
    set(gemv_source ../benchmark/decode_gemv_benchmark.cpp)
  endif()
  add_executable(${target} ${gemv_source} src/decoder.cpp src/kv_cache.cpp src/kernels/transformer.cpp)
  target_include_directories(${target} PRIVATE include)
  target_compile_definitions(${target} PRIVATE LEAF_TEST_GEMV=1)
  target_link_libraries(${target} PRIVATE Threads::Threads)
  if(NOT MSVC AND LEAF_ENABLE_AVX2)
    target_compile_options(${target} PRIVATE -mno-avx2 -mno-fma)
  endif()
endforeach()
add_test(NAME leaf_decode_gemv_tests COMMAND leaf_decode_gemv_tests)
