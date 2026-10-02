$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$wakeState = $null
$locationPushed = $false
if ([Environment]::OSVersion.Platform -eq [PlatformID]::Win32NT) {
    if (-not ("LeafBenchmarkPower" -as [type])) {
        Add-Type -TypeDefinition @'
using System.Runtime.InteropServices;
public static class LeafBenchmarkPower {
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint state);
}
'@
    }
    $wakeState = [LeafBenchmarkPower]::SetThreadExecutionState([uint32]2147483649)
    if ($wakeState -eq 0) { throw "Could not prevent automatic sleep during verification" }
}
try {
    Push-Location $root
    $locationPushed = $true
    python -m pytest -q
    if ($LASTEXITCODE -ne 0) { throw "Python test suite failed" }

    & (Join-Path $PSScriptRoot "build_decoder.ps1")
    python tools/verify_decoder_architectures.py --output benchmark/results/decoder_architectures.json
    if ($LASTEXITCODE -ne 0) { throw "full decoder architecture parity check failed" }

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

    python tools/verify_rope_repeatkv_runtime.py --leaf-infer (Join-Path $root "build/leaf_infer.exe") `
        --leaf-bench (Join-Path $root "build/leaf_graph_bench.exe")
    if ($LASTEXITCODE -ne 0) { throw "C++ RoPE/RepeatKV parity check failed" }

    python tools/verify_kv_cache_runtime.py --session-exe (Join-Path $root "build/leaf_kv_session.exe")
    if ($LASTEXITCODE -ne 0) { throw "C++ dynamic KV-cache parity check failed" }
} finally {
    if ($locationPushed) { Pop-Location }
    if ($null -ne $wakeState) {
        [LeafBenchmarkPower]::SetThreadExecutionState($wakeState) | Out-Null
    }
}
