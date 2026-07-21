# Run the arm matrix sequentially on one device.
#
# Pilot (1 seed x 4 arms x 100 epochs):
#   .\scripts\run_all.ps1 -Seeds 0 -Epochs 100
#
# Full matrix (3 seeds x 4 arms x 200 epochs):
#   .\scripts\run_all.ps1 -Seeds 0,1,2 -Epochs 200
#
# Arm order is deliberate: dense first so a crash surfaces on the simplest arm,
# rigl second so that a short night still produces the arm the experiment is
# about. A failing arm is retried once, then skipped so one broken arm cannot
# consume the whole run.

param(
    [int[]]$Seeds = @(0),
    [int]$Epochs = 100,
    # Pilot arm set (revised 2026-07-21). dense first so a crash surfaces on the
    # simplest arm; the two sparse_momentum inits are the arms the redesign is
    # about. oneshot_prune is dropped from the pilot (Dice comparison, not
    # allocation) and returns for the full matrix.
    [string[]]$Arms = @("dense", "sparse_momentum_uniform_init",
                        "sparse_momentum_erk_init", "static_sparse"),
    [string]$ResultsDir = "results",

    # All four pilot arms ran on the 3070 Ti. Keep a matrix on ONE card so
    # wall-clock stays comparable across arms, and pick the card the machine's
    # other tenants are not using.
    [ValidateSet("5060ti", "3070ti")]
    [string]$Gpu = "5060ti",

    [int]$Workers = 0,
    [switch]$NoResume
)

$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$summary = @()
$overallStart = Get-Date

foreach ($seed in $Seeds) {
    foreach ($arm in $Arms) {
        $runId = "${arm}_seed${seed}"
        $ok = $false

        foreach ($attempt in 1, 2) {
            $started = Get-Date
            Write-Host ""
            Write-Host "=== $runId (attempt $attempt) ===" -ForegroundColor Cyan

            $p = @{ Arm = $arm; Seed = $seed; Epochs = $Epochs
                    ResultsDir = $ResultsDir; Gpu = $Gpu }
            if ($Workers -gt 0) { $p["Workers"] = $Workers }
            # Only attempt 1 honours -NoResume; a retry always resumes so it
            # picks up from the last checkpoint rather than restarting.
            if ($NoResume -and $attempt -eq 1) { $p["NoResume"] = $true }

            & "$PSScriptRoot\run_arm.ps1" @p
            $code = $LASTEXITCODE
            $mins = [math]::Round(((Get-Date) - $started).TotalMinutes, 1)

            if ($code -eq 0) {
                $summary += [pscustomobject]@{ Run = $runId; Status = "ok"; Minutes = $mins; Attempts = $attempt }
                $ok = $true
                break
            }
            Write-Host "$runId failed (exit $code) after $mins min" -ForegroundColor Yellow
            if ($attempt -eq 2) {
                $summary += [pscustomobject]@{ Run = $runId; Status = "FAILED"; Minutes = $mins; Attempts = 2 }
            }
        }

        if (-not $ok) {
            Write-Host "$runId skipped after 2 attempts; continuing" -ForegroundColor Red
        }
    }
}

Write-Host ""
Write-Host "=== matrix complete in $([math]::Round(((Get-Date) - $overallStart).TotalHours,2)) h ===" -ForegroundColor Cyan
$summary | Format-Table -AutoSize
$summary | Export-Csv -Path (Join-Path $ResultsDir "run_summary.csv") -NoTypeInformation
