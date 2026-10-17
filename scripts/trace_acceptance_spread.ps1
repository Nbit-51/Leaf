#Requires -RunAsAdministrator
param([string]$RunDirectory = 'build/acceptance-spread-20261015/admin-trace')
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$output = [IO.Path]::GetFullPath((Join-Path $root $RunDirectory))
if (Test-Path -LiteralPath $output) { throw 'Choose a fresh run directory; keep previous evidence.' }
$before = Join-Path $root 'build/acceptance-spread-20261015/before-diag.exe'
$after = Join-Path $root 'build/acceptance-spread-20261015/after-diag.exe'
if (!(Test-Path -LiteralPath $before) -or !(Test-Path -LiteralPath $after)) {
    throw 'Build the two version-4 diagnostic binaries first; see the acceptance spread report.'
}
$status = (& wpr -status | Out-String)
if ($status -notmatch 'WPR is not recording') { throw 'An existing WPR recording may be active; leaving it untouched.' }
New-Item -ItemType Directory -Path $output | Out-Null
$owned = $false
Push-Location $root
try {
    & wpr -start GeneralProfile -filemode -recordtempto $output
    if ($LASTEXITCODE -ne 0) { throw 'Windows did not start the scheduler trace.' }
    $owned = $true
    python tools/diagnose_acceptance_spread.py --before $before --after $after `
        --artifact build/manual-model-cache/models/0ba2eff1d3343234ba895c0d/decoder-32.leaf `
        --tokens build/gelu-erf-20261015/pythia70/tokens.json --cpu 2 --pairs 6 --runs 31 --warmup 10 `
        --output (Join-Path $output 'dispatch.json')
    if ($LASTEXITCODE -ne 0) { throw 'Diagnostic failed; the trace will still be saved.' }
} finally {
    if ($owned) {
        & wpr -stop (Join-Path $output 'scheduler.etl') 'Leaf acceptance timing spread' -skipPdbGen
        if ($LASTEXITCODE -ne 0) { Write-Warning 'WPR stop failed. Run wpr -status and save this owned trace manually.' }
    }
    Pop-Location
}
Write-Host "Local diagnostic saved to $output. ETL contains system activity; do not commit it."
