param([string]$Compiler = "g++", [string]$BuildDirectory = "build", [switch]$ExperimentalVectorGelu)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$output = Join-Path $root $BuildDirectory
New-Item -ItemType Directory -Force -Path $output | Out-Null
# AVX2/FMA functions use runtime dispatch; the executable itself stays portable.
[string[]]$experimentFlags = if ($ExperimentalVectorGelu) { @("-DLEAF_EXPERIMENTAL_VECTOR_GELU=1") } else { @() }
& $Compiler @experimentFlags "-std=c++17" "-O3" "-DNDEBUG" "-pthread" "-static" "-static-libgcc" "-static-libstdc++" "-I" (Join-Path $root "engine/include") `
    (Join-Path $root "engine/src/decoder.cpp") (Join-Path $root "engine/src/kv_cache.cpp") `
    (Join-Path $root "engine/src/kernels/transformer.cpp") (Join-Path $root "engine/src/main_decoder.cpp") `
    "-o" (Join-Path $output "leaf_decoder.exe") "-lpsapi"
if ($LASTEXITCODE -ne 0) { throw "decoder build failed" }
