param(
    [string]$LlamaDirectory = 'build/competitors/llama-b11540',
    [string]$Compiler = 'g++'
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$llamaRoot = if ([IO.Path]::IsPathRooted($LlamaDirectory)) { $LlamaDirectory } else { Join-Path $root $LlamaDirectory }
# Use matching headers and official CPU DLLs from the pinned release.
& $Compiler '-std=c++17' '-O3' '-DNDEBUG' `
    '-I' (Join-Path $llamaRoot 'source/include') '-I' (Join-Path $llamaRoot 'source/ggml/include') `
    (Join-Path $root 'benchmark/llama_decoder_benchmark.cpp') `
    (Join-Path $llamaRoot 'bin/llama.dll') (Join-Path $llamaRoot 'bin/ggml.dll') `
    (Join-Path $llamaRoot 'bin/ggml-base.dll') '-static-libgcc' '-static-libstdc++' `
    '-o' (Join-Path $llamaRoot 'bin/llama-decoder-benchmark.exe')
if ($LASTEXITCODE -ne 0) { throw 'llama.cpp decoder benchmark adapter build failed' }
