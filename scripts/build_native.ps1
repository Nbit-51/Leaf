param(
    [string]$Compiler = "g++",
    [string]$BuildDirectory = "build"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$output = Join-Path $root $BuildDirectory
New-Item -ItemType Directory -Force -Path $output | Out-Null
$common = @(
    "-std=c++17", "-O3", "-DNDEBUG", "-mavx2", "-mfma",
    "-I", (Join-Path $root "engine/include"),
    (Join-Path $root "engine/src/kernels/gemm.cpp"),
    (Join-Path $root "engine/src/kernels/conv.cpp")
)
& $Compiler @common (Join-Path $root "tests/native/test_kernels.cpp") "-o" (Join-Path $output "leaf_native_tests.exe")
if ($LASTEXITCODE -ne 0) { throw "native test build failed" }
& $Compiler @common (Join-Path $root "benchmark/native_benchmark.cpp") "-o" (Join-Path $output "leaf_kernel_bench.exe")
if ($LASTEXITCODE -ne 0) { throw "native benchmark build failed" }

$runtimeIncludes = @(
    "-I", (Join-Path $root "engine/include"),
    "-I", (Join-Path $root "engine/kernels")
)
$runtimeSources = @(
    (Join-Path $root "engine/src/graph_parser.cpp"),
    (Join-Path $root "engine/src/executor.cpp"),
    (Join-Path $root "engine/kernels/im2col.cpp"),
    (Join-Path $root "engine/kernels/gemm.cpp"),
    (Join-Path $root "engine/kernels/conv2d.cpp")
)
& $Compiler "-std=c++17" "-O3" "-DNDEBUG" "-mavx2" "-mfma" @runtimeIncludes @runtimeSources `
    (Join-Path $root "engine/tests/test_executor.cpp") "-o" (Join-Path $output "test_executor.exe")
if ($LASTEXITCODE -ne 0) { throw "FP32 executor test build failed" }
& $Compiler "-std=c++17" "-O3" "-DNDEBUG" "-mavx2" "-mfma" @runtimeIncludes @runtimeSources `
    (Join-Path $root "engine/tests/test_executor_buffer_pool.cpp") "-o" (Join-Path $output "test_executor_buffer_pool.exe")
if ($LASTEXITCODE -ne 0) { throw "buffer-pool test build failed" }
& $Compiler "-std=c++17" "-O3" "-DNDEBUG" "-mavx2" "-mfma" @runtimeIncludes @runtimeSources `
    (Join-Path $root "engine/src/main_infer.cpp") "-o" (Join-Path $output "leaf_infer.exe")
if ($LASTEXITCODE -ne 0) { throw "FP32 inference runtime build failed" }

& $Compiler "-std=c++17" "-O3" "-DNDEBUG" "-mavx2" "-mfma" `
    (Join-Path $root "engine/kernels/gemm.cpp") (Join-Path $root "engine/tests/test_gemm.cpp") `
    "-o" (Join-Path $output "test_gemm.exe")
if ($LASTEXITCODE -ne 0) { throw "FP32 GEMM test build failed" }
& $Compiler "-std=c++17" "-O3" "-DNDEBUG" "-mavx2" "-mfma" `
    (Join-Path $root "engine/kernels/im2col.cpp") (Join-Path $root "engine/tests/test_im2col.cpp") `
    "-o" (Join-Path $output "test_im2col.exe")
if ($LASTEXITCODE -ne 0) { throw "im2col test build failed" }
