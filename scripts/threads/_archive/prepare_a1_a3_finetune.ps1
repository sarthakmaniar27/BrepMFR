param(
    [string]$JsonDir = "Z:\thread_and_text\root_json",
    [string]$AbcJsonDir = "Z:\thread_and_text\abc_jsons",
    [string]$LiteRoot = "Z:\thread_and_text\lite",
    [string]$OutputRoot = "Z:\thread_and_text\no_a2",
    [string]$CondaEnv = "brep_mfr_pyg",
    [int]$FileWorkers = 0,
    [int]$ShortestPathWorkers = 0,
    [int]$ValidationMaxFiles = 0,
    [switch]$FromJson
)

$ErrorActionPreference = "Stop"
$Repo = (Get-Item $PSScriptRoot).Parent.Parent.FullName
Set-Location $Repo

if (-not (Test-Path $LiteRoot -PathType Container)) {
    throw "Existing lite dataset root not found: $LiteRoot"
}
if ([IO.Path]::GetFullPath($LiteRoot).TrimEnd('\') -eq [IO.Path]::GetFullPath($OutputRoot).TrimEnd('\')) {
    throw "OutputRoot must differ from LiteRoot; do not overwrite the trained lite dataset."
}

$PygDir = Join-Path $OutputRoot "pyg"
$LabelDir = Join-Path $OutputRoot "label"
New-Item -ItemType Directory -Force -Path $PygDir, $LabelDir | Out-Null

$LitePyg = Join-Path $LiteRoot "pyg"
$UseFastUpgrade = -not $FromJson -and (Test-Path $LitePyg -PathType Container)

if ($FileWorkers -le 0) {
    $cpu = [Environment]::ProcessorCount
    $FileWorkers = [Math]::Max(1, [Math]::Min(12, $cpu - 2))
}

if ($UseFastUpgrade) {
    Write-Host "Fast path: upgrading lite .pt graphs to no_a2 (A1+A3) under: $OutputRoot"
    Write-Host "FileWorkers=$FileWorkers  (JSON re-conversion is NOT used; pass -FromJson to force it)"
    Write-Host "Existing output .pt files are retained; the upgrader resumes by skipping them."
    $upgradeArgs = @(
        "scripts/threads/upgrade_lite_pt_to_no_a2.py",
        "--lite-root", $LiteRoot,
        "--output-root", $OutputRoot,
        "--file-workers", [string]$FileWorkers,
        "--spatial-pos-max", "32",
        "--max-edge-path-len", "16"
    )
    & conda run --no-capture-output -n $CondaEnv python @upgradeArgs
    if ($LASTEXITCODE -ne 0) {
        throw "A1+A3 lite upgrade failed with exit code $LASTEXITCODE"
    }
} else {
    if (-not (Test-Path $JsonDir -PathType Container)) {
        throw "Primary JSON directory not found: $JsonDir"
    }
    Write-Host "JSON path: converting JSON to no_a2 profile (A1+A3) under: $OutputRoot"
    Write-Host "ShortestPathWorkers=$ShortestPathWorkers (keep 0 unless a single huge graph needs it)"
    Write-Host "Existing output .pt files are retained; the converter resumes by skipping them."
    $convertArgs = @(
        "scripts/inference/json_to_brepmfr_pyg_optimized.py",
        "--json_dir", $JsonDir,
        "--pt_out_dir", $PygDir,
        "--label_out_dir", $LabelDir,
        "--spatial_pos_max", "32",
        "--inference_profile", "no_a2",
        "--max_edge_path_len", "16",
        "--shortest_path_workers", [string]$ShortestPathWorkers
    )
    if ($AbcJsonDir -and (Test-Path $AbcJsonDir -PathType Container)) {
        $convertArgs += @("--abc_json_dir", $AbcJsonDir)
    } elseif ($AbcJsonDir) {
        throw "ABC JSON directory not found: $AbcJsonDir"
    }
    & conda run --no-capture-output -n $CondaEnv python @convertArgs
    if ($LASTEXITCODE -ne 0) {
        throw "A1+A3 conversion failed with exit code $LASTEXITCODE"
    }

    foreach ($name in @("train.txt", "val.txt", "test.txt")) {
        $source = Join-Path $LiteRoot $name
        if (-not (Test-Path $source -PathType Leaf)) {
            throw "Required lite split list not found: $source"
        }
        Copy-Item -Force $source (Join-Path $OutputRoot $name)
    }
    Write-Host "Copied the original train/val/test split lists unchanged."
}

Write-Host "Validating profile tensors, split coverage, labels, and topology..."
$validateArgs = @(
    "scripts/threads/validate_a1_a3_finetune_data.py",
    "--dataset-root", $OutputRoot,
    "--reference-lite-root", $LiteRoot,
    "--report-a3-cap", "768"
)
if ($ValidationMaxFiles -gt 0) {
    $validateArgs += @("--max-files", [string]$ValidationMaxFiles)
}
& conda run --no-capture-output -n $CondaEnv python @validateArgs
if ($LASTEXITCODE -ne 0) {
    throw "A1+A3 dataset validation failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "A1+A3 dataset is ready: $OutputRoot"
Write-Host "Class weights do not need recomputation because labels and split membership are unchanged."
Write-Host "Next: run scripts/threads/train_a1_a3_from_lite.ps1 with your lite checkpoint."
