param([string]$Compiler = "g++", [string]$BuildDirectory = "build/embeddinggemma/native")
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$output = if ([IO.Path]::IsPathRooted($BuildDirectory)) { $BuildDirectory } else { Join-Path $root $BuildDirectory }
New-Item -ItemType Directory -Force $output | Out-Null
& $Compiler '-std=c++17' '-O3' '-DNDEBUG' '-ffp-contract=off' '-pthread' '-static' '-static-libgcc' '-static-libstdc++' `
    '-I' (Join-Path $root 'engine/include') (Join-Path $root 'engine/src/embedding.cpp') `
    (Join-Path $root 'engine/src/main_embed.cpp') (Join-Path $root 'engine/src/kernels/transformer.cpp') `
    '-o' (Join-Path $output 'leaf_embed.exe')
if ($LASTEXITCODE -ne 0) { throw 'Embedding runtime build failed' }
Write-Output "Built $output/leaf_embed.exe"
