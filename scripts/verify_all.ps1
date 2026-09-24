$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    python -m pytest -q
    if ($LASTEXITCODE -ne 0) { throw "Python test suite failed" }

    & (Join-Path $PSScriptRoot "build_native.ps1")
    & (Join-Path $root "build/leaf_native_tests.exe")
    if ($LASTEXITCODE -ne 0) { throw "native correctness tests failed" }

    foreach ($test in @("test_gemm.exe", "test_im2col.exe", "test_executor.exe", "test_executor_buffer_pool.exe")) {
        & (Join-Path $root "build/$test")
        if ($LASTEXITCODE -ne 0) { throw "$test failed" }
    }

    & (Join-Path $root "build/leaf_kernel_bench.exe") --enforce-speedup --output `
        (Join-Path $root "benchmark/results/native_latest.json")
    if ($LASTEXITCODE -ne 0) { throw "native latency regression gate failed" }

    python -m benchmark.run_baselines
    if ($LASTEXITCODE -ne 0) { throw "PyTorch baseline check failed" }

    python tools/verify_cpp_runtime.py --leaf-infer (Join-Path $root "build/leaf_infer.exe")
    if ($LASTEXITCODE -ne 0) { throw "C++ runtime parity check failed" }

    python tools/verify_quantized_runtime.py --leaf-infer (Join-Path $root "build/leaf_infer.exe")
    if ($LASTEXITCODE -ne 0) { throw "C++ INT8 graph parity check failed" }

    python tools/verify_memory_plan_runtime.py --leaf-infer (Join-Path $root "build/leaf_infer.exe") `
        --leaf-bench (Join-Path $root "build/leaf_graph_bench.exe")
    if ($LASTEXITCODE -ne 0) { throw "C++ memory-plan runtime check failed" }

    python tools/verify_transformer_runtime.py --leaf-infer (Join-Path $root "build/leaf_infer.exe") `
        --leaf-bench (Join-Path $root "build/leaf_graph_bench.exe")
    if ($LASTEXITCODE -ne 0) { throw "C++ RMSNorm parity check failed" }

    python tools/verify_swiglu_runtime.py --leaf-infer (Join-Path $root "build/leaf_infer.exe") `
        --leaf-bench (Join-Path $root "build/leaf_graph_bench.exe") --enforce-no-slowdown
    if ($LASTEXITCODE -ne 0) { throw "C++ SwiGLU parity or speed gate failed" }

    python tools/verify_attention_runtime.py --leaf-infer (Join-Path $root "build/leaf_infer.exe") `
        --leaf-bench (Join-Path $root "build/leaf_graph_bench.exe")
    if ($LASTEXITCODE -ne 0) { throw "C++ attention parity check failed" }
} finally {
    Pop-Location
}
