# Rewrite approved-list JSON paths from one root folder to another.
# Keeps the filename; only changes the directory prefix.
#
# Example:
#   powershell -ExecutionPolicy Bypass -File scripts\threads\rewrite_approved_json_paths.ps1
#
# Or with custom paths:
#   powershell -ExecutionPolicy Bypass -File scripts\threads\rewrite_approved_json_paths.ps1 `
#     -InputList "D:\thread_and_text\no_confident_thread_or_text.txt" `
#     -OldRoot "C:\jsons" `
#     -NewRoot "D:\thread_and_text\stock_abc_json"

param(
    [string]$InputList = "D:\thread_and_text\no_confident_thread_or_text.txt",
    [string]$OldRoot = "C:\jsons",
    [string]$NewRoot = "D:\thread_and_text\stock_abc_json",
    [string]$OutputList = "",
    [switch]$SkipExistsCheck
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $InputList -PathType Leaf)) {
    throw "Input list not found: $InputList"
}
if (-not (Test-Path -LiteralPath $NewRoot -PathType Container)) {
    throw "New root directory not found: $NewRoot"
}

if ([string]::IsNullOrWhiteSpace($OutputList)) {
    $dir = Split-Path -Parent $InputList
    $name = [IO.Path]::GetFileNameWithoutExtension($InputList)
    $OutputList = Join-Path $dir "${name}_rewritten.txt"
}

$oldRootNorm = $OldRoot.TrimEnd('\', '/')
$newRootNorm = $NewRoot.TrimEnd('\', '/')

$lines = Get-Content -LiteralPath $InputList -Encoding UTF8
$out = New-Object System.Collections.Generic.List[string]
$missing = New-Object System.Collections.Generic.List[string]
$rewritten = 0
$unchanged = 0
$blank = 0

foreach ($raw in $lines) {
    $line = $raw.Trim().Trim('"')
    if ([string]::IsNullOrWhiteSpace($line) -or $line.StartsWith("#")) {
        $out.Add($raw)
        $blank++
        continue
    }

    $fileName = [IO.Path]::GetFileName($line)
    $newPath = Join-Path $newRootNorm $fileName

    # Prefer rewrite when path is under OldRoot OR when only the filename matters.
    $isUnderOld = $line.StartsWith($oldRootNorm, [StringComparison]::OrdinalIgnoreCase)
    if ($isUnderOld -or $fileName.EndsWith(".json", [StringComparison]::OrdinalIgnoreCase)) {
        $out.Add($newPath)
        $rewritten++
        if (-not $SkipExistsCheck -and -not (Test-Path -LiteralPath $newPath -PathType Leaf)) {
            $missing.Add($newPath)
        }
    }
    else {
        $out.Add($line)
        $unchanged++
    }
}

$utf8NoBom = New-Object System.Text.UTF8Encoding $false
[IO.File]::WriteAllLines($OutputList, $out, $utf8NoBom)

Write-Host "Input:      $InputList"
Write-Host "Old root:   $oldRootNorm"
Write-Host "New root:   $newRootNorm"
Write-Host "Output:     $OutputList"
Write-Host "Rewritten:  $rewritten"
Write-Host "Unchanged:  $unchanged"
Write-Host "Blank/#:    $blank"

if ($missing.Count -gt 0) {
    Write-Host ""
    Write-Host "WARNING: $($missing.Count) rewritten path(s) do not exist under NewRoot."
    Write-Host "First 10 missing:"
    $missing | Select-Object -First 10 | ForEach-Object { Write-Host "  $_" }
    throw "Aborting: missing files under $newRootNorm. Fix copy/path or pass -SkipExistsCheck."
}

Write-Host ""
Write-Host "OK. Use this list as -ApprovedList:"
Write-Host "  $OutputList"
