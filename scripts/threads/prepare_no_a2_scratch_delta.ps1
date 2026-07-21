param(
    [string]$JsonDir = "D:\thread_and_text\root_json",
    [string]$BaseNoA2Root = "D:\thread_and_text\no_a2",
    [string]$OutputRoot = "D:\thread_and_text\no_a2_large",
    [string]$AbcJsonDir = "D:\thread_and_text\abc_jsons",
    [string]$MapJson = "scripts/threads/remap_maps/thread_text_sw_to_brep.json",
    [string]$ClassWeightsOut = "artifacts/class_weights/thread_text/no_a2_large_70k_train_alpha05.json",
    [string]$CondaEnv = "brep_mfr_pyg",
    [int]$ValidationMaxFiles = 0,
    [int]$RemapWorkers = 8,
    [int]$MinFreeGB = 20,
    [ValidateSet("Copy", "HardLink")]
    [string]$SeedMode = "HardLink",
    [switch]$ResetOutput,
    [switch]$ApplyLabelRemap
)

$ErrorActionPreference = "Stop"
$Repo = (Get-Item $PSScriptRoot).Parent.Parent.FullName
Set-Location $Repo

$BasePygDir = Join-Path $BaseNoA2Root "pyg"
$PygDir = Join-Path $OutputRoot "pyg"
if (-not (Test-Path $JsonDir -PathType Container)) {
    throw "JSON directory not found: $JsonDir"
}
if (-not $AbcJsonDir -or -not (Test-Path $AbcJsonDir -PathType Container)) {
    throw "ABC JSON directory not found: $AbcJsonDir"
}
if (-not (Test-Path $BasePygDir -PathType Container)) {
    throw "Base no_a2 PyG directory not found: $BasePygDir"
}
if (-not (Test-Path $MapJson -PathType Leaf)) {
    throw "Label map not found: $MapJson"
}

$baseFull = [IO.Path]::GetFullPath($BaseNoA2Root).TrimEnd('\')
$outputFull = [IO.Path]::GetFullPath($OutputRoot).TrimEnd('\')
$SameRoot = $baseFull -eq $outputFull

$SourceJsonDirs = @($JsonDir)
if ([IO.Path]::GetFullPath($AbcJsonDir).TrimEnd('\') -ne [IO.Path]::GetFullPath($JsonDir).TrimEnd('\')) {
    $SourceJsonDirs += $AbcJsonDir
}

Write-Host "Step 1/7: strict dry-run label audit on JSONs missing from the expanded dataset..."
Write-Host "  protected base: $BaseNoA2Root"
Write-Host "  new output:     $OutputRoot"
Write-Host "  JSON sources:   $($SourceJsonDirs -join ', ')"
foreach ($SourceJsonDir in $SourceJsonDirs) {
    $remapAuditArgs = @(
        "scripts/threads/remap_missing_no_a2_json_labels.py",
        "--json-dir", $SourceJsonDir,
        "--pyg-dir", $BasePygDir,
        "--map-json", $MapJson,
        "--workers", [string]$RemapWorkers,
        "--dry-run"
    )
    & conda run --no-capture-output -n $CondaEnv python @remapAuditArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Delta label audit failed for $SourceJsonDir. Fix unknown labels/read errors before continuing."
    }
}
if (-not $ApplyLabelRemap) {
    Write-Host ""
    Write-Host "Dry run passed. No JSON files were modified."
    Write-Host "Rerun the same command with -ApplyLabelRemap to prepare the complete scratch dataset."
    return
}

if ($ResetOutput) {
    if ($SameRoot) {
        throw "-ResetOutput is refused because BaseNoA2Root and OutputRoot are identical."
    }
    if (Test-Path $OutputRoot) {
        Write-Host "Resetting partial expanded output only: $OutputRoot"
        Remove-Item -LiteralPath $OutputRoot -Recurse -Force
    }
}

Write-Host "Step 2/7: seeding the new output from the existing no_a2 dataset..."
if (-not $SameRoot) {
    New-Item -ItemType Directory -Force -Path $PygDir | Out-Null
    if ($SeedMode -eq "Copy") {
        Write-Host "Copying base .pt files with robocopy; the original directory remains independent."
        & robocopy $BasePygDir $PygDir "*.pt" /E /COPY:DAT /DCOPY:T /R:2 /W:1 /MT:16 /NFL /NDL /NP
        $robocopyCode = $LASTEXITCODE
        if ($robocopyCode -gt 7) {
            throw "robocopy failed with exit code $robocopyCode"
        }
    } else {
        Write-Host "Creating hard links (fast/no duplicate bytes; treat linked .pt files as immutable)."
        foreach ($source in Get-ChildItem $BasePygDir -Filter "*.pt" -File) {
            $destination = Join-Path $PygDir $source.Name
            if (-not (Test-Path $destination -PathType Leaf)) {
                New-Item -ItemType HardLink -Path $destination -Target $source.FullName | Out-Null
            }
        }
    }
} else {
    Write-Warning "BaseNoA2Root and OutputRoot are identical; the base dataset is not isolated."
}

Write-Host "Step 3/7: remapping labels only in missing JSONs..."
foreach ($SourceJsonDir in $SourceJsonDirs) {
    $remapWriteArgs = @(
        "scripts/threads/remap_missing_no_a2_json_labels.py",
        "--json-dir", $SourceJsonDir,
        "--pyg-dir", $PygDir,
        "--map-json", $MapJson,
        "--workers", [string]$RemapWorkers,
        "--yes-write",
        "--skip-prewrite-audit"
    )
    & conda run --no-capture-output -n $CondaEnv python @remapWriteArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Delta label remap failed for $SourceJsonDir."
    }
}

Write-Host "Step 4/7: converting missing JSONs directly to the expanded no_a2 dataset..."
$driveRoot = [IO.Path]::GetPathRoot([IO.Path]::GetFullPath($OutputRoot))
$driveInfo = [IO.DriveInfo]::new($driveRoot)
$freeGB = [Math]::Round($driveInfo.AvailableFreeSpace / 1GB, 2)
Write-Host "Free space on $driveRoot before conversion: $freeGB GB"
if ($freeGB -lt $MinFreeGB) {
    throw "Only $freeGB GB is free on $driveRoot; refusing conversion below MinFreeGB=$MinFreeGB. Use -ResetOutput -SeedMode HardLink or free disk space."
}
$probePath = Join-Path $PygDir (".write_probe_" + [guid]::NewGuid().ToString("N") + ".tmp")
try {
    [IO.File]::WriteAllBytes($probePath, [byte[]](0..255))
} finally {
    Remove-Item $probePath -Force -ErrorAction SilentlyContinue
}
$convertArgs = @(
    "scripts/inference/json_to_brepmfr_pyg_optimized.py",
    "--json_dir", $JsonDir,
    "--abc_json_dir", $AbcJsonDir,
    "--pt_out_dir", $PygDir,
    "--inference_profile", "no_a2",
    "--spatial_pos_max", "32",
    "--max_edge_path_len", "16",
    "--shortest_path_workers", "0"
)
& conda run --no-capture-output -n $CondaEnv python -u @convertArgs
if ($LASTEXITCODE -ne 0) {
    throw "Delta JSON -> no_a2 conversion failed."
}

Write-Host "Checking that every root and ABC JSON now has a no_a2 .pt..."
foreach ($SourceJsonDir in $SourceJsonDirs) {
    $coverageArgs = @(
        "scripts/threads/remap_missing_no_a2_json_labels.py",
        "--json-dir", $SourceJsonDir,
        "--pyg-dir", $PygDir,
        "--map-json", $MapJson,
        "--workers", [string]$RemapWorkers,
        "--dry-run",
        "--require-no-missing"
    )
    & conda run --no-capture-output -n $CondaEnv python @coverageArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Post-conversion coverage check failed for $SourceJsonDir."
    }
}

Write-Host "Step 5/7: backing up old split lists and generating new 70K splits..."
$splitFiles = @("train.txt", "val.txt", "test.txt")
$existingSplits = @($splitFiles | Where-Object { Test-Path (Join-Path $OutputRoot $_) })
if ($existingSplits.Count -gt 0) {
    $backupDir = Join-Path $OutputRoot ("split_backup_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
    New-Item -ItemType Directory -Force -Path $backupDir | Out-Null
    foreach ($name in $existingSplits) {
        Copy-Item -Force (Join-Path $OutputRoot $name) (Join-Path $backupDir $name)
    }
    Write-Host "Old split lists backed up under: $backupDir"
}

$splitArgs = @(
    "scripts/threads/make_random_splits.py",
    "--pyg-dir", $PygDir,
    "--out-dir", $OutputRoot,
    "--seed", "42"
)
if ($AbcJsonDir -and (Test-Path $AbcJsonDir -PathType Container)) {
    $splitArgs += @(
        "--abc-json-dir", $AbcJsonDir,
        "--abc-min-train-frac", "0.8"
    )
}
& conda run --no-capture-output -n $CondaEnv python @splitArgs
if ($LASTEXITCODE -ne 0) {
    throw "Split generation failed."
}

Write-Host "Step 6/7: recomputing class weights from the new train split..."
& conda run --no-capture-output -n $CondaEnv python scripts/training/compute_class_weights.py `
    --dataset_path $OutputRoot `
    --split train `
    --num_classes 3 `
    --alpha 0.5 `
    --num_workers 0 `
    --skip-bad `
    --out $ClassWeightsOut
if ($LASTEXITCODE -ne 0) {
    throw "Class-weight computation failed."
}

Write-Host "Step 7/7: validating the complete expanded no_a2 dataset..."
$validateArgs = @(
    "scripts/threads/validate_a1_a3_finetune_data.py",
    "--dataset-root", $OutputRoot,
    "--report-a3-cap", "768"
)
if ($ValidationMaxFiles -gt 0) {
    $validateArgs += @("--max-files", [string]$ValidationMaxFiles)
} else {
    $validateArgs += "--quarantine-invalid"
}
& conda run --no-capture-output -n $CondaEnv python @validateArgs
if ($LASTEXITCODE -ne 0) {
    throw "Complete no_a2 validation failed."
}

$nJson = (Get-ChildItem $JsonDir -Filter "*.json" -File).Count
$nAbcJson = (Get-ChildItem $AbcJsonDir -Filter "*.json" -File).Count
$nPyg = (Get-ChildItem $PygDir -Filter "*.pt" -File).Count
Write-Host ""
Write-Host "Scratch dataset preparation complete."
Write-Host "  protected base: $BaseNoA2Root"
Write-Host "  expanded root:  $OutputRoot"
Write-Host "  root JSON:      $nJson"
Write-Host "  ABC JSON:       $nAbcJson"
Write-Host "  no_a2 .pt:      $nPyg"
Write-Host "  class weights:  $ClassWeightsOut"
Write-Host "Next: run scripts/threads/train_no_a2_from_scratch.ps1"
