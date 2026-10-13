param([string]$BuildDirectory = 'build/attention-span-20261012', [string]$Compiler = 'g++')
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$output = if ([IO.Path]::IsPathRooted($BuildDirectory)) { $BuildDirectory } else { Join-Path $root $BuildDirectory }
if (Test-Path (Join-Path $output 'embedding-benchmark.exe')) { throw 'Use a fresh build directory; preserve measured binaries' }
New-Item -ItemType Directory -Force $output | Out-Null
$flags = @('-std=c++17', '-O3', '-DNDEBUG', '-ffp-contract=off', '-pthread', '-static', '-I', (Join-Path $root 'engine/include'))
& $Compiler @flags (Join-Path $root 'tests/native/test_attention_vector.cpp') `
    (Join-Path $root 'engine/src/kernels/transformer.cpp') '-o' (Join-Path $output 'test.exe')
if ($LASTEXITCODE -ne 0) { throw 'Attention test build failed' }
& (Join-Path $output 'test.exe')
if ($LASTEXITCODE -ne 0) { throw 'Attention correctness failed' }
& $Compiler @flags (Join-Path $root 'benchmark/attention_span_benchmark.cpp') '-o' (Join-Path $output 'micro.exe')
if ($LASTEXITCODE -ne 0) { throw 'Attention microbenchmark build failed' }
& $Compiler @flags (Join-Path $root 'benchmark/embedding_runtime_benchmark.cpp') `
    (Join-Path $root 'engine/src/embedding.cpp') (Join-Path $root 'engine/src/kernels/transformer.cpp') `
    '-o' (Join-Path $output 'embedding-benchmark.exe')
if ($LASTEXITCODE -ne 0) { throw 'Embedding adapter build failed' }
& (Join-Path $root 'scripts/build_embedding.ps1') -Compiler $Compiler -BuildDirectory $output
& (Join-Path $root 'scripts/build_decoder.ps1') -Compiler $Compiler -BuildDirectory (Join-Path $output 'decoder')
