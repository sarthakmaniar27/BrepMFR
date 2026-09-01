param(
    [string]$JsonDir = "Z:\thread_and_text\cadsynth_with_fillets_and_champer\root_json",
    [string]$WorkRoot = "Z:\thread_and_text\cadsynth_with_fillets_and_champer\five_class_a1_a3",
    [string]$MapJson = "scripts/threads/remap_maps/thread_text_sw_to_brep_with_identity.json",
    [string]$ClassWeightsOut = "artifacts/class_weights/thread_text/cadsynth_5class_a1_a3_train_alpha05.json",
    [string]$CondaEnv = "brep_mfr_pyg",
    [int]$RemapWorkers = 12,
    [int]$LiteWorkers = 8,
    [int]$FileWorkers = 0,
    [int]$ValidationMaxFiles = 0,
    [switch]$ApplyLabelRemap
)

$ErrorActionPreference = "Stop"
$Repo = (Get-Item $PSScriptRoot).Parent.Parent.FullName
Set-Location $Repo

$LiteRoot = Join-Path $WorkRoot "lite"
$DatasetRoot = Join-Path $WorkRoot "no_a2"
$LitePygDir = Join-Path $LiteRoot "pyg"
$LiteLabelDir = Join-Path $LiteRoot "label"
$DatasetPygDir = Join-Path $DatasetRoot "pyg"
$EmptyAuditPygDir = Join-Path $WorkRoot "_empty_label_audit_pyg"

if (-not (Test-Path -LiteralPath $JsonDir -PathType Container)) {
    throw "Source JSON directory not found: $JsonDir"
}
if (-not (Test-Path -LiteralPath $MapJson -PathType Leaf)) {
    throw "Label map not found: $MapJson"
}
if ($ValidationMaxFiles -lt 0) {
    throw "ValidationMaxFiles must be non-negative (0 validates the complete dataset)."
}
if ($RemapWorkers -le 0 -or $LiteWorkers -le 0) {
    throw "RemapWorkers and LiteWorkers must be greater than zero."
}
if ($FileWorkers -le 0) {
    $cpu = [Environment]::ProcessorCount
    $FileWorkers = [Math]::Max(1, [Math]::Min(12, $cpu - 2))
}

$jsonFiles = @(Get-ChildItem -LiteralPath $JsonDir -Filter "*.json" -File)
if ($jsonFiles.Count -eq 0) {
    throw "No top-level JSON files found in: $JsonDir"
}

Write-Host "Five-class A1+A3 scratch dataset preparation"
Write-Host "  source JSON:   $JsonDir"
Write-Host "  JSON files:    $($jsonFiles.Count)"
Write-Host "  work root:     $WorkRoot"
Write-Host "  lite staging:  $LiteRoot"
Write-Host "  final dataset: $DatasetRoot"
Write-Host "  ABC data:      disabled"
Write-Host "  label map:     0->0, 70->1, 101->2, 15->3, 24->4"
Write-Host "  remap workers: $RemapWorkers"
Write-Host "  lite workers:  $LiteWorkers"
Write-Host ""

Write-Host "Step 1/7: auditing every source label in parallel (read-only)..."
New-Item -ItemType Directory -Force -Path $EmptyAuditPygDir | Out-Null
if (Get-ChildItem -LiteralPath $EmptyAuditPygDir -Filter "*.pt" -File | Select-Object -First 1) {
    throw "Internal audit directory must contain no .pt files: $EmptyAuditPygDir"
}
$auditArgs = @(
    "scripts/threads/remap_missing_no_a2_json_labels.py",
    "--json-dir", $JsonDir,
    "--pyg-dir", $EmptyAuditPygDir,
    "--map-json", $MapJson,
    "--workers", [string]$RemapWorkers,
    "--dry-run"
)
& conda run --no-capture-output -n $CondaEnv python @auditArgs
if ($LASTEXITCODE -ne 0) {
    throw "Label audit failed. Do not remap or train until every unknown label/read error is fixed."
}

if (-not $ApplyLabelRemap) {
    Write-Host ""
    Write-Host "Dry-run label audit passed. No JSON files were changed."
    Write-Host "Rerun this script with -ApplyLabelRemap to remap in place and build the dataset."
    return
}

Write-Host "Step 2/7: remapping labels in place..."
$remapArgs = @(
    "scripts/threads/remap_missing_no_a2_json_labels.py",
    "--json-dir", $JsonDir,
    "--pyg-dir", $EmptyAuditPygDir,
    "--map-json", $MapJson,
    "--workers", [string]$RemapWorkers,
    "--yes-write",
    "--skip-prewrite-audit"
)
& conda run --no-capture-output -n $CondaEnv python @remapArgs
if ($LASTEXITCODE -ne 0) {
    throw "Label remap failed."
}
Remove-Item -LiteralPath $EmptyAuditPygDir -Force

Write-Host "Step 3/7: converting remapped JSON to small lite graphs..."
New-Item -ItemType Directory -Force -Path $LitePygDir, $LiteLabelDir | Out-Null
$liteArgs = @(
    "scripts/threads/convert_json_to_lite_parallel.py",
    "--json-dir", $JsonDir,
    "--pt-out-dir", $LitePygDir,
    "--label-out-dir", $LiteLabelDir,
    "--workers", [string]$LiteWorkers
)
& conda run --no-capture-output -n $CondaEnv python -u @liteArgs
if ($LASTEXITCODE -ne 0) {
    throw "JSON-to-lite conversion failed."
}

$jsonStems = [Collections.Generic.HashSet[string]]::new(
    [string[]]@($jsonFiles | ForEach-Object { $_.BaseName }),
    [StringComparer]::OrdinalIgnoreCase
)
$liteStems = [Collections.Generic.HashSet[string]]::new(
    [string[]]@(Get-ChildItem -LiteralPath $LitePygDir -Filter "*.pt" -File | ForEach-Object { $_.BaseName }),
    [StringComparer]::OrdinalIgnoreCase
)
$missingLite = [Collections.Generic.List[string]]::new()
foreach ($stem in $jsonStems) {
    if (-not $liteStems.Contains($stem)) {
        $missingLite.Add($stem)
    }
}
$extraLite = [Collections.Generic.List[string]]::new()
foreach ($stem in $liteStems) {
    if (-not $jsonStems.Contains($stem)) {
        $extraLite.Add($stem)
    }
}
if ($missingLite.Count -gt 0 -or $extraLite.Count -gt 0) {
    $missingExample = ($missingLite | Select-Object -First 10) -join ", "
    $extraExample = ($extraLite | Select-Object -First 10) -join ", "
    throw (
        "Lite coverage mismatch: JSON=$($jsonStems.Count), PT=$($liteStems.Count), " +
        "missing=$($missingLite.Count) [$missingExample], extra=$($extraLite.Count) [$extraExample]. " +
        "Use a clean WorkRoot or fix failed JSON conversions."
    )
}
Write-Host "Verified one lite graph for every JSON: $($liteStems.Count)"

Write-Host "Step 4/7: creating STEP-aware 80/10/10 splits (no ABC quota)..."
& conda run --no-capture-output -n $CondaEnv python scripts/threads/make_random_splits.py `
    --pyg-dir $LitePygDir `
    --out-dir $LiteRoot `
    --train-frac 0.8 `
    --val-frac 0.1 `
    --seed 42
if ($LASTEXITCODE -ne 0) {
    throw "Split generation failed."
}

Write-Host "Step 5/7: calculating A1 spatial proximity and A3 edge paths..."
Write-Host "  file workers: $FileWorkers"
$upgradeArgs = @(
    "scripts/threads/upgrade_lite_pt_to_no_a2.py",
    "--lite-root", $LiteRoot,
    "--output-root", $DatasetRoot,
    "--file-workers", [string]$FileWorkers,
    "--spatial-pos-max", "32",
    "--max-edge-path-len", "16"
)
& conda run --no-capture-output -n $CondaEnv python -u @upgradeArgs
if ($LASTEXITCODE -ne 0) {
    throw "A1/A3 generation failed."
}

Write-Host "Step 6/7: validating labels, splits, A1/A3 tensors, and lite parity..."
$validateArgs = @(
    "scripts/threads/validate_a1_a3_finetune_data.py",
    "--dataset-root", $DatasetRoot,
    "--reference-lite-root", $LiteRoot,
    "--num-classes", "5",
    "--report-a3-cap", "768"
)
if ($ValidationMaxFiles -gt 0) {
    $validateArgs += @("--max-files", [string]$ValidationMaxFiles)
}
& conda run --no-capture-output -n $CondaEnv python @validateArgs
if ($LASTEXITCODE -ne 0) {
    throw "A1/A3 dataset validation failed."
}

$finalCount = @(Get-ChildItem -LiteralPath $DatasetPygDir -Filter "*.pt" -File).Count
if ($finalCount -ne $jsonStems.Count) {
    throw "Final graph count mismatch: expected $($jsonStems.Count), found $finalCount in $DatasetPygDir"
}

Write-Host "Step 7/7: computing five-class weights from the train split only..."
& conda run --no-capture-output -n $CondaEnv python scripts/training/compute_class_weights.py `
    --dataset_path $DatasetRoot `
    --split train `
    --num_classes 5 `
    --alpha 0.5 `
    --num_workers 0 `
    --out $ClassWeightsOut
if ($LASTEXITCODE -ne 0) {
    throw "Class-weight computation failed."
}

Write-Host ""
Write-Host "Preparation complete."
Write-Host "  final A1+A3 graphs: $finalCount"
Write-Host "  dataset root:       $DatasetRoot"
Write-Host "  class weights:      $ClassWeightsOut"
Write-Host "Next command:"
Write-Host "  powershell -ExecutionPolicy Bypass -File scripts/threads/train_5class_a1_a3_from_scratch.ps1"
