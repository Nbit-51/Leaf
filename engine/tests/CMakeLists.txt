add_executable(test_gemm test_gemm.cpp ${PROJECT_SOURCE_DIR}/kernels/gemm.cpp)
target_include_directories(test_gemm PRIVATE ${PROJECT_SOURCE_DIR})
add_test(NAME test_gemm COMMAND test_gemm)

add_executable(test_im2col test_im2col.cpp ${PROJECT_SOURCE_DIR}/kernels/im2col.cpp)
target_include_directories(test_im2col PRIVATE ${PROJECT_SOURCE_DIR})
add_test(NAME test_im2col COMMAND test_im2col)

add_executable(verify_conv_vs_python verify_conv_vs_python.cpp)
target_link_libraries(verify_conv_vs_python PRIVATE leaf_core)
target_include_directories(verify_conv_vs_python PRIVATE ${PROJECT_SOURCE_DIR}/include)

add_executable(debug_conv_direct debug_conv_direct.cpp)
target_link_libraries(debug_conv_direct PRIVATE leaf_core)
target_include_directories(debug_conv_direct PRIVATE ${PROJECT_SOURCE_DIR}/include)

add_executable(test_executor test_executor.cpp)
target_link_libraries(test_executor PRIVATE leaf_core)
target_include_directories(test_executor PRIVATE ${PROJECT_SOURCE_DIR}/include)
add_test(NAME test_executor COMMAND test_executor)

add_executable(test_executor_buffer_pool test_executor_buffer_pool.cpp)
target_link_libraries(test_executor_buffer_pool PRIVATE leaf_core)
target_include_directories(test_executor_buffer_pool PRIVATE ${PROJECT_SOURCE_DIR}/include)
add_test(NAME test_executor_buffer_pool COMMAND test_executor_buffer_pool)
