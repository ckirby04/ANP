# Launch a training run as an INDEPENDENT process.
#
#   .\scripts\launch_detached.ps1 -Arm oneshot_prune -Seed 0 -Epochs 100 -Gpu 3070ti -Workers 8
#
# Why this exists: a run started from an interactive or agent-driven shell is a
# child of that shell. When the shell or its background job is stopped, the
# whole process tree dies with it, including a multi-hour training run. Twice
# now that has killed a pilot arm mid-flight.
#
# Start-Process creates a process that does not belong to the caller's job
# object, so the run survives the launching shell exiting. Output goes to a log
# file under the run directory rather than to the caller's console.
#
# Check on it with .\scripts\progress.ps1, and stop it with
# Stop-Process -Id <pid> using the PID this prints.

param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("dense", "static_sparse", "oneshot_prune", "rigl")]
    [string]$Arm,

    [int]$Seed = 0,
    [int]$Epochs = 0,
    [string]$ResultsDir = "results",
    [string]$RunId = "",
    [ValidateSet("5060ti", "3070ti")]
    [string]$Gpu = "5060ti",
    [int]$Workers = 0,
    [switch]$NoResume
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

if (-not $RunId) { $RunId = "${Arm}_seed${Seed}" }
$runDir = Join-Path $repo (Join-Path $ResultsDir $RunId)
New-Item -ItemType Directory -Force $runDir | Out-Null

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outLog = Join-Path $runDir "launch_$stamp.out.log"
$errLog = Join-Path $runDir "launch_$stamp.err.log"

$inner = @(
    "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
    (Join-Path $PSScriptRoot "run_arm.ps1"),
    "-Arm", $Arm, "-Seed", $Seed, "-ResultsDir", $ResultsDir,
    "-RunId", $RunId, "-Gpu", $Gpu
)
if ($Epochs -gt 0) { $inner += @("-Epochs", $Epochs) }
if ($Workers -gt 0) { $inner += @("-Workers", $Workers) }
if ($NoResume) { $inner += "-NoResume" }

$proc = Start-Process -FilePath "powershell.exe" -ArgumentList $inner `
    -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $outLog -RedirectStandardError $errLog

Write-Host "launched $RunId detached" -ForegroundColor Green
Write-Host "  pid    : $($proc.Id)"
Write-Host "  gpu    : $Gpu"
Write-Host "  stdout : $outLog"
Write-Host "  stderr : $errLog"
Write-Host "  watch  : .\scripts\progress.ps1"
Write-Host "  stop   : Stop-Process -Id $($proc.Id)"
