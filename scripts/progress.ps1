# Snapshot of every run under a results directory.
#
#   .\scripts\progress.ps1
#   .\scripts\progress.ps1 -ResultsDir results

param([string]$ResultsDir = "results")

$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$rows = @()
foreach ($dir in Get-ChildItem $ResultsDir -Directory -ErrorAction SilentlyContinue) {
    $log = Join-Path $dir.FullName "training_log.jsonl"
    if (-not (Test-Path $log)) { continue }

    # -ReadCount 0 forces an array even for a single line; without it
    # Get-Content returns a string and [-1] indexes the last character.
    $lines = @(Get-Content $log -ReadCount 0)
    if ($lines.Count -eq 0) { continue }
    $last = $lines[-1] | ConvertFrom-Json

    $traj = Join-Path $dir.FullName "trajectory.csv"
    $density = ""
    if (Test-Path $traj) {
        $t = Import-Csv $traj
        if ($t.Count -gt 0) {
            $maxStep = ($t | Measure-Object -Property step -Maximum).Maximum
            $tail = $t | Where-Object { [int]$_.step -eq [int]$maxStep }
            $live = ($tail | Measure-Object -Property n_live -Sum).Sum
            $tot = ($tail | Measure-Object -Property n_weights -Sum).Sum
            if ($tot -gt 0) { $density = [math]::Round($live / $tot, 5) }
        }
    }

    $rows += [pscustomobject]@{
        Run        = $dir.Name
        Epochs     = $lines.Count
        LastEpoch  = $last.epoch
        TrainLoss  = [math]::Round($last.train_loss, 4)
        ValLoss    = [math]::Round($last.val_loss, 4)
        SecPerIter = [math]::Round($last.s_per_iter_mean, 3)
        EpochSec   = [math]::Round($last.epoch_time_s, 1)
        Density    = $density
    }
}

if ($rows.Count -eq 0) {
    Write-Host "no runs with logged epochs under $ResultsDir"
} else {
    $rows | Format-Table -AutoSize
}
