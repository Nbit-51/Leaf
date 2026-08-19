add_executable(test_gemm test_gemm.cpp ${CMAKE_SOURCE_DIR}/kernels/gemm.cpp)
target_include_directories(test_gemm PRIVATE ${CMAKE_SOURCE_DIR})
add_test(NAME test_gemm COMMAND test_gemm)
