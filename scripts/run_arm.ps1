# Run a single arm at a single seed.
#
#   .\scripts\run_arm.ps1 -Arm rigl -Seed 0
#   .\scripts\run_arm.ps1 -Arm dense -Seed 1 -Epochs 200
#
# Resumes from results\<run_id>\checkpoint_latest.pt if present. Pass -NoResume
# to start over.

param(
    # Config basename under configs\ (without .yaml). Validated by file
    # existence rather than a fixed set, so new arms/inits work without editing
    # this script. E.g. dense, static_sparse, sparse_momentum_uniform_init.
    [Parameter(Mandatory = $true)]
    [string]$Arm,

    [int]$Seed = 0,
    [int]$Epochs = 0,
    [string]$ResultsDir = "results",
    [string]$RunId = "",

    # Which physical card to train on. Use this rather than overriding the raw
    # device index: the name match is what decides, so `--set device=cuda:1`
    # on its own is SILENTLY INEFFECTIVE. It resolves straight back to whatever
    # require_device_name points at, warns, and runs on the card you were
    # trying to move off. This parameter sets device, name and VRAM together.
    [ValidateSet("5060ti", "3070ti")]
    [string]$Gpu = "5060ti",

    # Train dataloader workers. Lower this when sharing the machine with
    # another job; the loader is memory-bandwidth bound, so it competes for
    # more than just CPU.
    [int]$Workers = 0,

    [switch]$NoResume
)

# Deliberately Continue, not Stop. PowerShell wraps any stderr output from a
# native executable in a NativeCommandError, and under Stop that aborts the run
# on nothing more than a Python UserWarning. Success is judged by $LASTEXITCODE.
$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

# The dataset is not distributed with this repository. ANP_DATA_ROOT names the
# directory holding it; nothing here is a literal path.
if (-not $env:ANP_DATA_ROOT) {
    throw @"
ANP_DATA_ROOT is not set.

Set it to the directory that contains nnUNet\nnUNet_preprocessed\Dataset002_BraTS_MEN:

    `$env:ANP_DATA_ROOT = 'D:\BraTS-MEN'

BraTS-MEN is not distributed with this repository and must be obtained from its
own source under its own terms.
"@
}
if (-not (Test-Path $env:ANP_DATA_ROOT)) {
    throw "ANP_DATA_ROOT points at '$($env:ANP_DATA_ROOT)', which does not exist."
}

# nnU-Net emits warnings on every worker spawn without these. Read-only use;
# nothing in this project writes to the dataset root.
$nnunet = Join-Path $env:ANP_DATA_ROOT "nnUNet"
$env:nnUNet_raw = Join-Path $nnunet "nnUNet_raw"
$env:nnUNet_preprocessed = Join-Path $nnunet "nnUNet_preprocessed"
$env:nnUNet_results = Join-Path $nnunet "nnUNet_results"

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

$configPath = Join-Path $repo "configs\$Arm.yaml"
if (-not (Test-Path $configPath)) {
    $available = (Get-ChildItem "$repo\configs\*.yaml" | ForEach-Object { $_.BaseName }) -join ", "
    throw "no config configs\$Arm.yaml. Available: $available"
}
if (-not $RunId) { $RunId = "${Arm}_seed${Seed}" }

# Under CUDA_DEVICE_ORDER=PCI_BUS_ID the 5060 Ti is index 0 and the 3070 Ti is
# index 1. The name assertion is what actually decides, so the index here is a
# consistency check rather than the selector.
$gpuSpec = @{
    "5060ti" = @{ Device = "cuda:0"; Name = "RTX 5060 Ti"; MinVram = 15.0 }
    "3070ti" = @{ Device = "cuda:1"; Name = "RTX 3070 Ti"; MinVram = 7.0 }
}[$Gpu]

$argsList = @(
    "src\train.py", "configs\$Arm.yaml",
    "--set", "seed=$Seed",
    "--set", "run_id=$RunId",
    "--set", "logging.results_dir=$ResultsDir",
    "--set", "device=$($gpuSpec.Device)",
    "--set", "require_device_name=$($gpuSpec.Name)",
    "--set", "require_min_vram_gb=$($gpuSpec.MinVram)"
)
if ($Epochs -gt 0) { $argsList += @("--set", "train.epochs=$Epochs") }
if ($Workers -gt 0) {
    $argsList += @("--set", "data.num_workers=$Workers")
}
if ($NoResume) { $argsList += "--no-resume" }

Write-Host "target GPU: $Gpu ($($gpuSpec.Name))" -ForegroundColor DarkGray

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
