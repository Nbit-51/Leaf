param(
    [Parameter(Mandatory=$true)][ValidateSet('Build','Quality','Timing32','Timing8')][string]$Stage,
    [Parameter(Mandatory=$true)][string]$RunDirectory,
    [string]$Compiler = 'g++',
    [string]$ModelPath,
    [string]$WorkDirectory = 'build/tinyllama_windows_fair'
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    $run = if ([IO.Path]::IsPathRooted($RunDirectory)) { [IO.Path]::GetFullPath($RunDirectory) }
           else { [IO.Path]::GetFullPath((Join-Path $root $RunDirectory)) }
    $baseline = Join-Path $run 'baseline/leaf_decoder.exe'
    $candidate = Join-Path $run 'candidate/leaf_decoder.exe'
    $manifestPath = Join-Path $run 'build-manifest.json'
    $quality = Join-Path $run 'quality.json'
    function Invoke-Checked([string]$Program, [string[]]$Arguments) {
        & $Program @Arguments
        if ($LASTEXITCODE -ne 0) { throw "$Program failed (exit $LASTEXITCODE). Stop here and share the log." }
    }
    function Source-Hashes {
        $paths = @('scripts/build_decoder.ps1', 'scripts/test_silu_gate.ps1',
            'tests/native/test_silu_gate.cpp', 'tools/verify_cached_decoder.py',
            'tools/benchmark_default_abba.py', 'tools/validate_decoder.py',
            'tools/decoder_validation.py', 'tools/benchmark_decoder_comparison.py',
            'tools/benchmark_core_scaling.py', 'tools/verify_decoder_architectures.py')
        $paths += @(Get-ChildItem engine/include -Recurse -Filter *.h | ForEach-Object { $_.FullName })
        $paths += @('engine/src/decoder.cpp', 'engine/src/main_decoder.cpp',
            'engine/src/kv_cache.cpp', 'engine/src/kernels/transformer.cpp')
        $result = @{}
        foreach ($path in $paths) {
            $resolved = (Resolve-Path -LiteralPath $path).Path
            $result[$resolved] = (Get-FileHash -LiteralPath $resolved -Algorithm SHA256).Hash
        }
        return $result
    }
    if ($Stage -eq 'Build') {
        if (Test-Path -LiteralPath $run) { throw 'Use a fresh run directory; preserve earlier results.' }
        New-Item -ItemType Directory -Path $run | Out-Null
    } elseif (!(Test-Path -LiteralPath $manifestPath)) {
        throw 'Complete the Build stage first.'
    }
    if ($Stage -eq 'Quality' -and (!$ModelPath -or !(Test-Path -LiteralPath $ModelPath -PathType Container))) {
        throw 'Quality requires -ModelPath pointing to the cached TinyLlama snapshot.'
    }
    $log = Join-Path $run ($Stage.ToLower() + '.log')
    if (Test-Path -LiteralPath $log) { throw "Preserve $log; use a new run after fixing a failed stage." }
    Start-Transcript -Path $log | Out-Null
    try {
        if ($Stage -eq 'Build') {
            $source = Source-Hashes
            & ./scripts/build_decoder.ps1 -Compiler $Compiler -BuildDirectory (Join-Path $run 'baseline') -ConservativeSiluGate
            & ./scripts/build_decoder.ps1 -Compiler $Compiler -BuildDirectory (Join-Path $run 'candidate') -ExperimentalSiluGate
            $test = Join-Path $run 'silu_gate_tests.exe'
            Invoke-Checked $Compiler @('-std=c++17','-O3','-ffp-contract=off','-I','engine/include',
                'tests/native/test_silu_gate.cpp','-o',$test)
            Invoke-Checked $test @()
            Invoke-Checked 'python' @('-m','pytest','tests/unit/test_silu_gate_experiment.py',
                'tests/unit/test_decoder_quality.py','tests/unit/test_packaging.py','-q',
                '--basetemp',(Join-Path $run 'pytest-temp'),'-o',('cache_dir=' + (Join-Path $run 'pytest-cache')))
            $current = Source-Hashes
            foreach ($path in $source.Keys) {
                if ($current[$path] -ne $source[$path]) { throw 'Sources changed during build; start a fresh run.' }
            }
            @{ source_sha256=$source;
               baseline_sha256=(Get-FileHash $baseline -Algorithm SHA256).Hash;
               candidate_sha256=(Get-FileHash $candidate -Algorithm SHA256).Hash;
               compiler=$Compiler } | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 $manifestPath
        } else {
            $saved = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
            $current = Source-Hashes
            foreach ($entry in $saved.source_sha256.PSObject.Properties) {
                if ($current[$entry.Name] -ne $entry.Value) { throw 'Sources changed since Build; use a fresh run.' }
            }
            if ((Get-FileHash $baseline -Algorithm SHA256).Hash -ne $saved.baseline_sha256 -or
                (Get-FileHash $candidate -Algorithm SHA256).Hash -ne $saved.candidate_sha256) {
                throw 'A built executable changed; use a fresh run.'
            }
            if ($Stage -eq 'Quality') {
                Invoke-Checked 'python' @('tools/verify_decoder_architectures.py','--executable',$candidate,
                    '--output',(Join-Path $run 'architectures.json'))
                Invoke-Checked 'python' @('tools/verify_cached_decoder.py','--executable',$candidate,
                    '--model',$ModelPath,'--workdir',$WorkDirectory,'--output',$quality,'--require-silu-gate')
            } else {
                $bits = if ($Stage -eq 'Timing32') { '32' } else { '8' }
                $timing = Join-Path $run ('timing-' + $bits + '.json')
                Invoke-Checked 'python' @('tools/benchmark_default_abba.py','--before',$baseline,'--after',$candidate,
                    '--workdir',$WorkDirectory,'--output',$timing,
                    '--activation-bits',$bits,'--silu-quality',$quality,'--pairs','6','--runs','31','--warmup','10')
                $result = Get-Content -LiteralPath $timing -Raw | ConvertFrom-Json
                if ($result.complete -ne $true -or $result.gate.passed -ne $true) {
                    throw "Timing gate did not pass. Preserve and share $timing; no improvement is qualified."
                }
            }
        }
        Write-Host "PASS $Stage. Results: $run"
    } finally { Stop-Transcript | Out-Null }
} finally { Pop-Location }
