param(
    [Parameter(Mandatory = $true)]
    [string]$OldNoA2Root,

    [Parameter(Mandatory = $true)]
    [string]$NewLabeledJsonDir,

    [string]$ApprovedList = "C:\jsons\inference\no_confident_thread_or_text.txt",

    [Parameter(Mandatory = $true)]
    [string]$StockJsonDir,

    [Parameter(Mandatory = $true)]
    [string]$CombinedNoA2Root,

    [string]$MapJson = "scripts/threads/remap_maps/thread_text_sw_to_brep.json",
    [string]$ClassWeightsOut = "artifacts/class_weights/thread_text/new_abc_finetune_alpha05.json",
    [string]$CondaEnv = "brep_mfr_pyg",
    [int]$Workers = 8,
    [int]$ValidationMaxFiles = 0,
    [int]$MinFreeGB = 20,
    [ValidateSet("HardLink", "Copy")]
    [string]$SeedMode = "HardLink",
    [switch]$Apply,
    [switch]$OverwriteStockJsons,
    [switch]$ResetCombinedOutput
)

$ErrorActionPreference = "Stop"
$Repo = (Get-Item $PSScriptRoot).Parent.Parent.FullName
Set-Location $Repo

function Assert-Directory([string]$PathValue, [string]$Description) {
    if (-not (Test-Path -LiteralPath $PathValue -PathType Container)) {
        throw "$Description directory not found: $PathValue"
    }
}

Assert-Directory (Join-Path $OldNoA2Root "pyg") "Old no_a2 PyG"
Assert-Directory $NewLabeledJsonDir "New labeled JSON"
if (-not (Test-Path -LiteralPath $ApprovedList -PathType Leaf)) {
    throw "Approved-list file not found: $ApprovedList"
}
if (-not (Test-Path -LiteralPath $MapJson -PathType Leaf)) {
    throw "Label-map JSON not found: $MapJson"
}
if ([IO.Path]::GetFullPath($OldNoA2Root).TrimEnd('\') -eq
    [IO.Path]::GetFullPath($CombinedNoA2Root).TrimEnd('\')) {
    throw "CombinedNoA2Root must differ from OldNoA2Root; the old 72K dataset is protected."
}
if ($Workers -le 0) {
    throw "Workers must be greater than zero."
}

Write-Host "New ABC fine-tuning dataset preparation"
Write-Host "  old no_a2 root:       $OldNoA2Root"
Write-Host "  new labeled JSONs:    $NewLabeledJsonDir"
Write-Host "  approved list:        $ApprovedList"
Write-Host "  Stock JSON output:    $StockJsonDir"
Write-Host "  combined no_a2 root:  $CombinedNoA2Root"
Write-Host "  class weights output: $ClassWeightsOut"
Write-Host "  mode:                 $(if ($Apply) { 'APPLY' } else { 'DRY RUN' })"
Write-Host ""

# The approved-list audit is always performed. In dry-run mode this proves that
# every selected source has a complete face[].label array and contains only
# raw Stock-compatible labels (-10/-1/0).
$stockArgs = @(
    "scripts/threads/prepare_approved_abc_stock_jsons.py",
    "--approved-list", $ApprovedList,
    "--output-dir", $StockJsonDir,
    "--expected-source-labels=-10,-1,0",
    "--stock-label", "0",
    "--workers", [string]$Workers
)
if ($Apply) {
    $stockArgs += "--write"
    if ($OverwriteStockJsons) {
        $stockArgs += "--overwrite"
    }
}
& conda run --no-capture-output -n $CondaEnv python @stockArgs
if ($LASTEXITCODE -ne 0) {
    throw "Approved ABC Stock-label preparation failed with exit code $LASTEXITCODE"
}

if (-not $Apply) {
    Write-Host ""
    Write-Host "Auditing raw labels in the new synthetic ABC JSON folder..."
    & conda run --no-capture-output -n $CondaEnv python `
        scripts/threads/repair_json_face_labels.py `
        --json-dir $NewLabeledJsonDir `
        --map-json $MapJson `
        --dry-run `
        --fail-on-unknown
    if ($LASTEXITCODE -ne 0) {
        throw "New labeled JSON audit failed with exit code $LASTEXITCODE"
    }
    Write-Host ""
    Write-Host "Dry run passed. No JSON or no_a2 files were written."
    Write-Host "Rerun with -Apply to create Stock copies and the combined no_a2 dataset."
    return
}

Assert-Directory $StockJsonDir "Prepared Stock JSON"

# Reuse the repository's complete delta preparer. It:
#   1. protects OldNoA2Root;
#   2. seeds CombinedNoA2Root by hard link or copy;
#   3. remaps only new raw JSON labels;
#   4. converts only missing JSON stems with the no_a2 profile;
#   5. regenerates STEP-family-aware splits;
#   6. recomputes class weights; and
#   7. validates all resulting A1/A3 graphs.
$prepareArgs = @{
    JsonDir = $NewLabeledJsonDir
    BaseNoA2Root = $OldNoA2Root
    OutputRoot = $CombinedNoA2Root
    AbcJsonDir = $StockJsonDir
    MapJson = $MapJson
    ClassWeightsOut = $ClassWeightsOut
    CondaEnv = $CondaEnv
    ValidationMaxFiles = $ValidationMaxFiles
    RemapWorkers = $Workers
    MinFreeGB = $MinFreeGB
    SeedMode = $SeedMode
    ApplyLabelRemap = $true
}
if ($ResetCombinedOutput) {
    $prepareArgs["ResetOutput"] = $true
}

& (Join-Path $PSScriptRoot "prepare_no_a2_scratch_delta.ps1") @prepareArgs
if ($LASTEXITCODE -ne 0) {
    throw "Combined no_a2 preparation failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "Preparation finished."
Write-Host "Fine-tuning dataset: $CombinedNoA2Root"
Write-Host "Class weights:       $ClassWeightsOut"
Write-Host "Next: scripts/threads/train_new_abc_finetune.ps1"
