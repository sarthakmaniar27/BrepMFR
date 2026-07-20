<#
.SYNOPSIS
  Local filter: C:\abc_steps -> C:\abc_steps_filtered using an allowlist file.

.DESCRIPTION
  Same logic as the Jenkins stage. Run on one machine, or call from Jenkins.
  Allowlist: one STEP key per line (..._step_NNN), from:
    python standalone_scripts/export_step_allowlist_from_inference.py

.EXAMPLE
  .\Filter-AbcStepsByAllowlist.ps1 `
    -AllowlistPath '\\LP76-RZA2-DSA\jsons\inference\allowed_step_keys.txt'
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$AllowlistPath,

    [string]$SourceDir = 'C:\abc_steps',
    [string]$DestDir = 'C:\abc_steps_filtered',
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $AllowlistPath)) {
    throw "Allowlist not found: $AllowlistPath"
}
if (-not (Test-Path -LiteralPath $SourceDir)) {
    throw "Source not found: $SourceDir"
}

$allow = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
Get-Content -LiteralPath $AllowlistPath |
    ForEach-Object { $_.Trim() } |
    Where-Object { $_ -and -not $_.StartsWith('#') } |
    ForEach-Object {
        $stem = [System.IO.Path]::GetFileNameWithoutExtension($_)
        if ($stem -match '^(?<key>.+?_step_\d+)') {
            [void]$allow.Add($Matches['key'].ToLowerInvariant())
        }
        else {
            [void]$allow.Add($stem.ToLowerInvariant())
        }
    }

Write-Host ("Allowlist keys: {0}" -f $allow.Count)

if (-not $DryRun) {
    if (Test-Path -LiteralPath $DestDir) {
        Get-ChildItem -LiteralPath $DestDir -File -Force -ErrorAction SilentlyContinue |
            Where-Object { $_.Extension -ieq '.step' -or $_.Extension -ieq '.stp' } |
            Remove-Item -Force
    }
    else {
        New-Item -Path $DestDir -ItemType Directory -Force | Out-Null
    }
}

$sourceFiles = @(
    Get-ChildItem -LiteralPath $SourceDir -File |
    Where-Object { $_.Extension -ieq '.step' -or $_.Extension -ieq '.stp' }
)

$matched = 0
foreach ($file in $sourceFiles) {
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($file.Name)
    if ($stem -match '^(?<key>.+?_step_\d+)') {
        $key = $Matches['key'].ToLowerInvariant()
        if ($allow.Contains($key)) {
            $matched++
            if ($DryRun) {
                Write-Host ("WOULD COPY {0}" -f $file.Name)
            }
            else {
                Copy-Item -LiteralPath $file.FullName -Destination (Join-Path $DestDir $file.Name) -Force
            }
        }
    }
}

Write-Host ("Source={0}  Matched={1}  Dest={2}" -f $sourceFiles.Count, $matched, $DestDir)
