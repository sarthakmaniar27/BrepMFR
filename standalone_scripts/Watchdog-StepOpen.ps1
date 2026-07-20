<#
.SYNOPSIS
  Watchdog for SolidWorks STEP open hangs.

.DESCRIPTION
  Started by BatchJsonExport.vba BEFORE LoadFile4.
  If in_progress.txt still exists after -TimeoutSec, the open is considered
  stuck: the STEP name is appended to skip_list.txt and SolidWorks is killed.

  Copy this file to: C:\jsons\Watchdog-StepOpen.ps1
  (same path as WATCHDOG_PS1 in the VBA macro)

.NOTES
  VBA cannot cancel LoadFile4 itself — killing SLDWORKS.exe is the only hard stop.
  Re-run the macro after a kill; it will skip the bad file and continue.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [int]$TimeoutSec,

    [Parameter(Mandatory = $true)]
    [string]$InProgressPath,

    [Parameter(Mandatory = $true)]
    [string]$SkipListPath
)

$ErrorActionPreference = 'SilentlyContinue'

Start-Sleep -Seconds $TimeoutSec

if (-not (Test-Path -LiteralPath $InProgressPath)) {
    # Open finished in time — nothing to do.
    exit 0
}

$stuckName = (Get-Content -LiteralPath $InProgressPath -TotalCount 1 -ErrorAction SilentlyContinue)
if (-not [string]::IsNullOrWhiteSpace($stuckName)) {
    $stuckName = $stuckName.Trim()
    $already = $false
    if (Test-Path -LiteralPath $SkipListPath) {
        $already = @(
            Get-Content -LiteralPath $SkipListPath -ErrorAction SilentlyContinue |
            ForEach-Object { $_.Trim() } |
            Where-Object { $_ -ieq $stuckName }
        ).Count -gt 0
    }
    if (-not $already) {
        Add-Content -LiteralPath $SkipListPath -Value $stuckName
    }
}

# Hard stop — only way to abort a stuck LoadFile4.
Get-Process -Name 'SLDWORKS' -ErrorAction SilentlyContinue |
    Stop-Process -Force -ErrorAction SilentlyContinue

exit 0
