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
)
target_include_directories(leaf_core PUBLIC include)

add_executable(leaf_parse_info
  src/main_parse_info.cpp
)
target_link_libraries(leaf_parse_info PRIVATE leaf_core)

enable_testing()
add_subdirectory(tests)
