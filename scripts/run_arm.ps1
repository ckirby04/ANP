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

# Deliberately Continue, not Stop. PowerShell wraps any stderr output from a
# native executable in a NativeCommandError, and under Stop that aborts the run
# on nothing more than a Python UserWarning. Success is judged by $LASTEXITCODE.
$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

# nnU-Net emits warnings on every worker spawn without these. Read-only use;
# nothing in this project writes to G:\BraTS-MEN.
$env:nnUNet_raw = "G:\BraTS-MEN\nnUNet\nnUNet_raw"
$env:nnUNet_preprocessed = "G:\BraTS-MEN\nnUNet\nnUNet_preprocessed"
$env:nnUNet_results = "G:\BraTS-MEN\nnUNet\nnUNet_results"

# CUDA's default FASTEST_FIRST ordering disagrees with nvidia-smi's. Under it,
# cuda:0 on this machine is the 8 GB RTX 3070 Ti rather than the 16 GB 5060 Ti.
# train.py sets this too; setting it here as well means it is right even when
# python is invoked directly.
$env:CUDA_DEVICE_ORDER = "PCI_BUS_ID"

# fft_conv_pytorch, pulled in by GaussianBlurTransform, emits a deprecation
# UserWarning on every worker spawn. It is benign and it floods the log.
if (-not $env:PYTHONWARNINGS) {
    $env:PYTHONWARNINGS = "ignore::UserWarning:fft_conv_pytorch.fft_conv"
}

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
