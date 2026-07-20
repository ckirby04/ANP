# Snapshot of every run under a results directory.
#
#   .\scripts\progress.ps1
#   .\scripts\progress.ps1 -ResultsDir results

param([string]$ResultsDir = "results")

$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

# Values can be null or NaN (an epoch with no validation batches, say), and
# [math]::Round refuses anything that is not a double.
function Round2($value, [int]$digits) {
    $d = 0.0
    if ($null -eq $value) { return "" }
    if (-not [double]::TryParse([string]$value, [ref]$d)) { return "" }
    if ([double]::IsNaN($d) -or [double]::IsInfinity($d)) { return "" }
    return [math]::Round($d, $digits)
}

$rows = @()
foreach ($dir in Get-ChildItem $ResultsDir -Directory -ErrorAction SilentlyContinue) {
    $log = Join-Path $dir.FullName "training_log.jsonl"
    if (-not (Test-Path $log)) { continue }

    # The @() wrapper forces an array even when the file has one line. Without
    # it Get-Content returns a bare string and [-1] indexes the last character
    # rather than the last epoch. Note -ReadCount 0 is NOT the fix: it emits a
    # single array object, which @() then wraps into a one-element array.
    $lines = @(Get-Content $log)
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
            if ($tot -gt 0) { $density = Round2 ($live / $tot) 5 }
        }
    }

    $rows += [pscustomobject]@{
        Run        = $dir.Name
        Epochs     = $lines.Count
        LastEpoch  = $last.epoch
        TrainLoss  = Round2 $last.train_loss 4
        ValLoss    = Round2 $last.val_loss 4
        SecPerIter = Round2 $last.s_per_iter_mean 3
        EpochSec   = Round2 $last.epoch_time_s 1
        Density    = $density
    }
}

if ($rows.Count -eq 0) {
    Write-Host "no runs with logged epochs under $ResultsDir"
} else {
    $rows | Format-Table -AutoSize
}
