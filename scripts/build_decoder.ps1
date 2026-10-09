param([string]$Compiler = "g++", [string]$BuildDirectory = "build", [switch]$ExperimentalVectorGelu,
      [switch]$ExperimentalAttentionAvx2, [switch]$ExperimentalRowReuse, [switch]$ExperimentalGemvPair,
      [switch]$ExperimentalMlpPack, [switch]$ConservativeFp32, [switch]$ExperimentalSiluGate)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$output = if ([IO.Path]::IsPathRooted($BuildDirectory)) { $BuildDirectory }
          else { Join-Path $root $BuildDirectory }
New-Item -ItemType Directory -Force -Path $output | Out-Null
# AVX2/FMA functions use runtime dispatch; the executable itself stays portable.
[string[]]$experimentFlags = if ($ExperimentalVectorGelu) { @("-DLEAF_EXPERIMENTAL_VECTOR_GELU=1") } else { @() }
if ($ExperimentalAttentionAvx2) { $experimentFlags += "-DLEAF_EXPERIMENTAL_ATTENTION_AVX2=1" }
if ($ExperimentalRowReuse) { $experimentFlags += "-DLEAF_EXPERIMENTAL_ROW_REUSE=1" }
if ($ExperimentalGemvPair) { $experimentFlags += "-DLEAF_EXPERIMENTAL_GEMV_PAIR=1" }
if ($ExperimentalMlpPack) { $experimentFlags += "-DLEAF_EXPERIMENTAL_MLP_PACK=1" }
if ($ExperimentalSiluGate) { $experimentFlags += "-DLEAF_EXPERIMENTAL_SILU_GATE=1" }
if ($ConservativeFp32) { $experimentFlags += "-DLEAF_OPTIMIZED_FP32=0" }
& $Compiler @experimentFlags "-std=c++17" "-O3" "-DNDEBUG" "-pthread" "-static" "-static-libgcc" "-static-libstdc++" "-I" (Join-Path $root "engine/include") `
    (Join-Path $root "engine/src/decoder.cpp") (Join-Path $root "engine/src/kv_cache.cpp") `
    (Join-Path $root "engine/src/kernels/transformer.cpp") (Join-Path $root "engine/src/main_decoder.cpp") `
    "-o" (Join-Path $output "leaf_decoder.exe") "-lpsapi"
if ($LASTEXITCODE -ne 0) { throw "decoder build failed" }
