# Run a single arm at a single seed.
#
#   .\scripts\run_arm.ps1 -Arm rigl -Seed 0
#   .\scripts\run_arm.ps1 -Arm dense -Seed 1 -Epochs 200
#
# Resumes from results\<run_id>\checkpoint_latest.pt if present. Pass -NoResume
# to start over.

param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("dense", "static_sparse", "oneshot_prune", "rigl")]
    [string]$Arm,

    [int]$Seed = 0,
    [int]$Epochs = 0,
    [string]$ResultsDir = "results",
    [string]$RunId = "",
    [switch]$NoResume
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

# nnU-Net emits warnings on every worker spawn without these. Read-only use;
# nothing in this project writes to G:\BraTS-MEN.
$env:nnUNet_raw = "G:\BraTS-MEN\nnUNet\nnUNet_raw"
$env:nnUNet_preprocessed = "G:\BraTS-MEN\nnUNet\nnUNet_preprocessed"
$env:nnUNet_results = "G:\BraTS-MEN\nnUNet\nnUNet_results"

if (-not $RunId) { $RunId = "${Arm}_seed${Seed}" }

$argsList = @(
    "src\train.py", "configs\$Arm.yaml",
    "--set", "seed=$Seed",
    "--set", "run_id=$RunId",
    "--set", "logging.results_dir=$ResultsDir"
)
if ($Epochs -gt 0) { $argsList += @("--set", "train.epochs=$Epochs") }
if ($NoResume) { $argsList += "--no-resume" }

Write-Host "=== $RunId ===" -ForegroundColor Cyan
$started = Get-Date
& python @argsList
$code = $LASTEXITCODE
$elapsed = (Get-Date) - $started

if ($code -ne 0) {
    Write-Host "$RunId FAILED after $([math]::Round($elapsed.TotalMinutes,1)) min (exit $code)" -ForegroundColor Red
    exit $code
}
Write-Host "$RunId done in $([math]::Round($elapsed.TotalHours,2)) h" -ForegroundColor Green
