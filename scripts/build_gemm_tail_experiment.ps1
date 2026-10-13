param([string]$BuildDirectory = 'build/gemm-tail-20261011', [string]$Compiler = 'g++')
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$output = if ([IO.Path]::IsPathRooted($BuildDirectory)) { $BuildDirectory } else { Join-Path $root $BuildDirectory }
if (Test-Path (Join-Path $output 'embedding-benchmark.exe')) { throw 'Use a fresh build directory; preserve measured binaries' }
New-Item -ItemType Directory -Force $output | Out-Null
$overlay = Join-Path $output 'include'
New-Item -ItemType Directory -Force (Join-Path $overlay 'leaf/kernels') | Out-Null
$header = [IO.File]::ReadAllText((Join-Path $root 'engine/include/leaf/kernels/row_reuse.h'))
if (-not $header.Contains('NarrowTail = false')) { throw 'Tail experiment anchor changed' }
[IO.File]::WriteAllText((Join-Path $overlay 'leaf/kernels/row_reuse.h'), $header.Replace('NarrowTail = false', 'NarrowTail = true'))
$flags = @('-std=c++17', '-O3', '-DNDEBUG', '-ffp-contract=off', '-pthread', '-static', '-I', $overlay, '-I', (Join-Path $root 'engine/include'))
foreach ($test in @('tests/native/test_token_panel.cpp', 'benchmark/gemm_tail_benchmark.cpp')) {
    & $Compiler @flags (Join-Path $root $test) '-o' (Join-Path $output (([IO.Path]::GetFileNameWithoutExtension($test)) + '.exe'))
    if ($LASTEXITCODE -ne 0) { throw "Build failed: $test" }
}
& (Join-Path $output 'test_token_panel.exe')
if ($LASTEXITCODE -ne 0) { throw 'Kernel correctness failed' }
& $Compiler @flags (Join-Path $root 'benchmark/embedding_runtime_benchmark.cpp') `
    (Join-Path $root 'engine/src/embedding.cpp') (Join-Path $root 'engine/src/kernels/transformer.cpp') `
    '-o' (Join-Path $output 'embedding-benchmark.exe')
if ($LASTEXITCODE -ne 0) { throw 'Embedding adapter build failed' }
& $Compiler @flags (Join-Path $root 'engine/src/embedding.cpp') (Join-Path $root 'engine/src/main_embed.cpp') `
    (Join-Path $root 'engine/src/kernels/transformer.cpp') '-o' (Join-Path $output 'leaf_embed.exe')
if ($LASTEXITCODE -ne 0) { throw 'Tail candidate runtime build failed' }
