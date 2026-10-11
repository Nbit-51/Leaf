param(
    [string]$LlamaDirectory = 'build/competitors/llama-b11540',
    [string]$LeafOutput = 'build/competitors/leaf-embedding-benchmark.exe',
    [string]$Compiler = 'g++',
    [switch]$Diagnostics
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$llamaRoot = if ([IO.Path]::IsPathRooted($LlamaDirectory)) { $LlamaDirectory } else { Join-Path $root $LlamaDirectory }
$leafBinary = if ([IO.Path]::IsPathRooted($LeafOutput)) { $LeafOutput } else { Join-Path $root $LeafOutput }
$source = Join-Path $root 'benchmark/embedding_runtime_benchmark.cpp'
$llamaBinary = Join-Path $llamaRoot 'bin/llama-embedding-benchmark.exe'
if ($Diagnostics) {
    # Never overwrite preserved acceptance binaries with instrumented builds.
    $leafBinary = Join-Path (Split-Path -Parent $leafBinary) 'leaf-embedding-diagnostic.exe'
    $llamaBinary = Join-Path $llamaRoot 'bin/llama-embedding-diagnostic.exe'
}
[string[]]$diagnosticFlags = @()
if ($Diagnostics) { $diagnosticFlags += '-DLEAF_EMBED_DIAGNOSTICS' }
# Use the pinned release's matching headers and official CPU DLLs.
& $Compiler @diagnosticFlags '-std=c++17' '-O3' '-DNDEBUG' '-DLEAF_COMPARE_LLAMA' `
    '-I' (Join-Path $llamaRoot 'source/include') '-I' (Join-Path $llamaRoot 'source/ggml/include') `
    $source (Join-Path $llamaRoot 'bin/llama.dll') (Join-Path $llamaRoot 'bin/ggml.dll') `
    (Join-Path $llamaRoot 'bin/ggml-base.dll') '-static-libgcc' '-static-libstdc++' '-o' $llamaBinary
if ($LASTEXITCODE -ne 0) { throw 'llama.cpp benchmark adapter build failed' }
New-Item -ItemType Directory -Force (Split-Path -Parent $leafBinary) | Out-Null
& $Compiler @diagnosticFlags '-std=c++17' '-O3' '-DNDEBUG' '-ffp-contract=off' '-pthread' '-static' `
    '-I' (Join-Path $root 'engine/include') $source (Join-Path $root 'engine/src/embedding.cpp') `
    (Join-Path $root 'engine/src/kernels/transformer.cpp') '-o' $leafBinary
if ($LASTEXITCODE -ne 0) { throw 'Leaf benchmark adapter build failed' }
Write-Output "Built $leafBinary and $llamaBinary"
