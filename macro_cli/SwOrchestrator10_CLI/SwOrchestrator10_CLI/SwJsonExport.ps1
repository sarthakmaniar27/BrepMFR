<#
.SYNOPSIS
  Babysitter for JSON_Generator_2027.swp — does NOT modify SwOrchestrator.Cli.

.DESCRIPTION
  Launches SOLIDWORKS with /m <macro>, watches conversion status + JSON output,
  kills and restarts on stall/crash, blacklists poison files after N failures.

  Your existing thread command keeps using SwOrchestrator.Cli.exe unchanged.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\SwJsonExport.ps1 `
    -SwExe   "C:\images\image_08_03\WinRel64\sldworks.exe" `
    -Macro   "C:\Threads\macro\JSON_Generator_2027.swp" `
    -Parts   "C:\Threads\conversion\sldprts" `
    -JsonOut "C:\Threads\conversion\jsons" `
    -Status  "C:\Threads\conversion\status" `
    -FailureThreshold 3
#>
[CmdletBinding()]
param(
    [string]$SwExe              = "C:\images\image_08_03\WinRel64\sldworks.exe",
    [Parameter(Mandatory = $true)]
    [string]$Macro,
    [string]$Parts              = "C:\Threads\conversion\sldprts",
    [string]$JsonOut            = "C:\Threads\conversion\jsons",
    [string]$Status             = "C:\Threads\conversion\status",
    [int]$StallTimeoutSec       = 900,
    [int]$StartupGraceSec       = 600,
    [int]$PollIntervalSec       = 10,
    [int]$CooldownSec           = 8,
    [int]$FailureThreshold      = 3,
    [int]$MaxRestarts           = 10000
)

$ErrorActionPreference = "Continue"
$HeartbeatFile = Join-Path $Status "heartbeat.txt"
$DoneMarker    = Join-Path $Status "batch_done.marker"
$SkipListFile  = Join-Path $Status "skip_files.txt"

function Write-Log([string]$Level, [string]$Message) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$ts] [$Level] $Message"
    if ($Level -eq "ERROR" -or $Level -eq "WARN") {
        [Console]::Error.WriteLine($line)
    } else {
        Write-Host $line
    }
}

function Ensure-Dir([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
    }
}

function Kill-SolidWorks {
    $names = @("SLDWORKS", "sldworks", "WerFault", "swspmanager", "swshellfileeventserver")
    foreach ($n in $names) {
        Get-Process -Name $n -ErrorAction SilentlyContinue |
            Stop-Process -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds $CooldownSec
}

function Get-JsonDoneCount {
    if (-not (Test-Path -LiteralPath $JsonOut)) { return 0 }
    return @(Get-ChildItem -LiteralPath $JsonOut -Filter "*.json" -File -ErrorAction SilentlyContinue).Count
}

function Get-PartsTotal {
    if (-not (Test-Path -LiteralPath $Parts)) { return 0 }
    return @(Get-ChildItem -LiteralPath $Parts -Filter "*.sldprt" -File -Recurse -ErrorAction SilentlyContinue).Count
}

function Read-Heartbeat {
    if (-not (Test-Path -LiteralPath $HeartbeatFile)) { return $null }
    try {
        $text = (Get-Content -LiteralPath $HeartbeatFile -Raw -ErrorAction Stop).Trim()
        if ([string]::IsNullOrWhiteSpace($text)) { return $null }
        $parts = $text.Split("|", 3)
        if ($parts.Count -lt 3) { return $null }
        return [pscustomobject]@{
            Timestamp = $parts[0].Trim()
            Status    = $parts[1].Trim()
            File      = $parts[2].Trim()
            MtimeUtc  = (Get-Item -LiteralPath $HeartbeatFile).LastWriteTimeUtc
        }
    } catch {
        return $null
    }
}

function Append-Skip([string]$Basename, [string]$Reason) {
    if ([string]::IsNullOrWhiteSpace($Basename)) { return }
    Ensure-Dir $Status
    $existing = @()
    if (Test-Path -LiteralPath $SkipListFile) {
        $existing = Get-Content -LiteralPath $SkipListFile -ErrorAction SilentlyContinue |
            ForEach-Object { $_.Trim() } |
            Where-Object { $_ -and -not $_.StartsWith("#") }
    }
    if ($existing -contains $Basename) { return }
    Add-Content -LiteralPath $SkipListFile -Value "# $(Get-Date -Format 'yyyy-MM-ddTHH:mm:ss')  $Reason"
    Add-Content -LiteralPath $SkipListFile -Value $Basename
    Write-Log "ERROR" "Blacklisted '$Basename' ($Reason)"
}

function Culprit-FromHeartbeat {
    $hb = Read-Heartbeat
    if ($null -eq $hb) { return $null }
    $active = @("file_start", "opening", "exporting")
    if ($active -notcontains $hb.Status -and -not $hb.Status.StartsWith("file_start")) {
        return $null
    }
    if ([string]::IsNullOrWhiteSpace($hb.File)) { return $null }
    return [System.IO.Path]::GetFileName($hb.File)
}

# --- Preflight -------------------------------------------------------------
Write-Log "INFO" "=== SwJsonExport babysitter (sibling to SwOrchestrator.Cli) ==="
Write-Log "INFO" "  SW exe   : $SwExe"
Write-Log "INFO" "  Macro    : $Macro"
Write-Log "INFO" "  Parts    : $Parts"
Write-Log "INFO" "  JSON out : $JsonOut"
Write-Log "INFO" "  Status   : $Status"
Write-Log "INFO" "  Stall    : ${StallTimeoutSec}s  StartupGrace: ${StartupGraceSec}s  FailThreshold: $FailureThreshold"

if (-not (Test-Path -LiteralPath $SwExe))  { Write-Log "ERROR" "SLDWORKS.exe not found: $SwExe"; exit 3 }
if (-not (Test-Path -LiteralPath $Macro))  { Write-Log "ERROR" "Macro .swp not found: $Macro"; exit 3 }
if (-not (Test-Path -LiteralPath $Parts))  { Write-Log "ERROR" "Parts folder not found: $Parts"; exit 3 }

Ensure-Dir $Status
Ensure-Dir $JsonOut

if (Test-Path -LiteralPath $DoneMarker) {
    Remove-Item -LiteralPath $DoneMarker -Force -ErrorAction SilentlyContinue
    Write-Log "INFO" "Cleared stale batch_done.marker"
}

$total = Get-PartsTotal
$doneJson = Get-JsonDoneCount
Write-Log "INFO" "Preflight: $total .sldprt under Parts; $doneJson .json under JsonOut (counts are independent)"

$crashCounts = @{}
$exitCode = 1

for ($attempt = 1; $attempt -le $MaxRestarts; $attempt++) {
    Write-Log "INFO" "=== Attempt $attempt ==="
    Kill-SolidWorks

    $jsonAtStart = Get-JsonDoneCount
    $lastJsonCount = $jsonAtStart
    $lastProgressUtc = [DateTime]::UtcNow
    $lastHbMtime = $null
    $hb0 = Read-Heartbeat
    if ($null -ne $hb0) { $lastHbMtime = $hb0.MtimeUtc }

    Write-Log "INFO" "Launching: `"$SwExe`" /m `"$Macro`""
    $proc = Start-Process -FilePath $SwExe -ArgumentList "/m `"$Macro`"" -PassThru
    if ($null -eq $proc) {
        Write-Log "ERROR" "Failed to start SOLIDWORKS"
        $exitCode = 1
        break
    }

    $outcome = "unknown"
    while ($true) {
        Start-Sleep -Seconds $PollIntervalSec

        if (Test-Path -LiteralPath $DoneMarker) {
            Write-Log "INFO" "batch_done.marker detected"
            $outcome = "done"
            break
        }

        try { $proc.Refresh() } catch {}
        if ($proc.HasExited) {
            Write-Log "WARN" "SOLIDWORKS exited (code=$($proc.ExitCode))"
            $outcome = "exited"
            break
        }

        $now = [DateTime]::UtcNow
        $jsonNow = Get-JsonDoneCount
        $hb = Read-Heartbeat
        $moved = $false

        if ($jsonNow -gt $lastJsonCount) {
            Write-Log "PROG" "JSON files: $jsonNow (+$($jsonNow - $lastJsonCount))"
            $lastJsonCount = $jsonNow
            $moved = $true
        }

        if ($null -ne $hb) {
            if ($null -eq $lastHbMtime -or $hb.MtimeUtc -gt $lastHbMtime) {
                $fname = if ($hb.File) { [System.IO.Path]::GetFileName($hb.File) } else { "(none)" }
                Write-Log "HB" "status=$($hb.Status) file=$fname"
                $lastHbMtime = $hb.MtimeUtc
                $moved = $true
            }
        }

        if ($moved) { $lastProgressUtc = $now }

        $grace = if ($jsonNow -eq $jsonAtStart) { $StartupGraceSec } else { $StallTimeoutSec }
        $idleSec = ($now - $lastProgressUtc).TotalSeconds
        if ($idleSec -gt $grace) {
            Write-Log "WARN" "No progress for $([int]$idleSec)s (grace=${grace}s). Declaring stall."
            $outcome = "stalled"
            break
        }
    }

    Kill-SolidWorks

    if ($outcome -eq "done") {
        Write-Log "INFO" "Batch complete."
        $exitCode = 0
        break
    }

    $culprit = Culprit-FromHeartbeat
    if ($culprit) {
        if (-not $crashCounts.ContainsKey($culprit)) { $crashCounts[$culprit] = 0 }
        $crashCounts[$culprit]++
        $n = $crashCounts[$culprit]
        Write-Log "WARN" "Attributed to '$culprit' (failure #$n)"
        if ($n -ge $FailureThreshold) {
            Append-Skip $culprit "auto-blacklisted after $n consecutive crashes/stalls"
        }
    }
}

Write-Log "INFO" "Exiting with code $exitCode"
exit $exitCode
