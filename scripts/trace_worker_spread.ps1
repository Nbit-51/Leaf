[CmdletBinding()]
param(
    [string]$RunDirectory='build/worker-placement-20261022/administrator-trace',
    [switch]$CheckOnly
)
$ErrorActionPreference='Stop'
$workspaceRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$buildBoundary=[IO.Path]::GetFullPath((Join-Path $workspaceRoot 'build')).TrimEnd('\')+'\'
$output=[IO.Path]::GetFullPath($(if ([IO.Path]::IsPathRooted($RunDirectory)) {$RunDirectory} else {Join-Path $workspaceRoot $RunDirectory}))
if (-not $output.StartsWith($buildBoundary,[StringComparison]::OrdinalIgnoreCase)) {throw 'Use a fresh directory inside repository build/'}
if (Test-Path -LiteralPath $output) {throw "Keep existing recording/results; choose another -RunDirectory: $output"}
$runner=Join-Path $workspaceRoot 'benchmark/experiments/worker-placement/replay_unmodified.py'
$python=(Get-Command python -ErrorAction Stop).Source
$wpr=(Get-Command wpr.exe -ErrorAction Stop).Source
$administrator=[Security.Principal.WindowsPrincipal]::new([Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
Push-Location -LiteralPath $workspaceRoot
try {
    & $python $runner --directory $output --check-only
    if ($LASTEXITCODE -ne 0) {throw 'Replay preflight failed; no recording started'}
    $drive=[IO.DriveInfo]::new([IO.Path]::GetPathRoot($output))
    if ($drive.AvailableFreeSpace -lt 4GB) {throw 'Need at least 4 GiB free for the local trace'}
    $profiles=& $wpr -profiles 2>&1
    if ($LASTEXITCODE -ne 0) {throw 'Cannot query installed WPR profiles'}
    foreach ($name in @('CPU','Power','Thermal')) {
        if (($profiles -join "`n") -notmatch "(?m)^\s+$name\s+") {throw "Installed WPR lacks $name profile"}
    }
    if ($CheckOnly) {
        [ordered]@{ready=$true;recording_started=$false;administrator=$administrator;recording_requires_administrator=$true;output=$output;free_gib=[Math]::Round($drive.AvailableFreeSpace/1GB,2)} | ConvertTo-Json
        return
    }
    if (-not $administrator) {throw 'Windows kernel tracing requires Administrator PowerShell. No recording started. Run this script from an elevated PowerShell.'}
    $status=& $wpr -status 2>&1
    if (($status -join "`n") -notmatch 'WPR is not recording') {throw 'An existing or unrecognized WPR session is active; leave it untouched'}
    New-Item -ItemType Directory -Path $output | Out-Null
    $temp=Join-Path $output 'wpr-temp'
    New-Item -ItemType Directory -Path $temp | Out-Null
    $instance='LeafWorkerTiming-'+[Guid]::NewGuid().ToString('N')
    $etl=Join-Path $output 'scheduler-power.etl'
    $record=[ordered]@{instance=$instance;started=$false;stopped=$false;replay_exit_code=$null;stop_exit_code=$null;etl=$etl;profiles=@('CPU.Light','Power.Light','Thermal.Light');scope='Local diagnostic only; includes system process metadata; do not publish the ETL'}
    $recordPath=Join-Path $output 'recording.json'
    $record | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $recordPath -Encoding UTF8
    try {
        & $wpr -start CPU.Light -start Power.Light -start Thermal.Light -filemode -recordtempto $temp -instancename $instance
        if ($LASTEXITCODE -ne 0) {
            # The random instance belongs only to this invocation, even if start partially failed.
            & $wpr -cancel -instancename $instance 2>&1 | Out-Null
            throw 'Could not start the named trace; no benchmark ran'
        }
        $record.started=$true
        $record | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $recordPath -Encoding UTF8
        Write-Output 'Recording about five minutes. Keep this window open and the laptop plugged in and idle.'
        & $python $runner --directory $output
        $record.replay_exit_code=$LASTEXITCODE
        if ($LASTEXITCODE -ne 0) {throw 'Replay stopped; preserve the partial trace/results for diagnosis'}
    } finally {
        if ($record.started) {
            & $wpr -stop $etl 'Leaf unmodified one-core scheduling and power diagnosis' -skipPdbGen -instancename $instance
            $record.stop_exit_code=$LASTEXITCODE
            $record.stopped=($LASTEXITCODE -eq 0)
            if (-not $record.stopped) {
                # Never stop or cancel the default session or another named recording.
                & $wpr -cancel -instancename $instance 2>&1 | Out-Null
            }
        }
        $record | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $recordPath -Encoding UTF8
    }
    if (-not $record.stopped) {throw 'Trace save failed; only this invocation was canceled. See recording.json'}
    Write-Output "Saved locally: $etl"
} finally {
    Pop-Location
}
