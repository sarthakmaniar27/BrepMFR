param(
    [string]$ModelARoot = "Z:\thread_and_text\no_a2",
    [string]$ExpandedRoot = "Z:\thread_and_text\no_a2_72k_plus_new_abc",
    [string]$NewAbcJsonDir = "Z:\thread_and_text\new_abc_json_25k",
    [string]$OutputRoot = "Z:\thread_and_text\abc_for_modelA_finetuning",
    [ValidateSet("HardLink", "Copy")]
    [string]$LinkMode = "HardLink",
    [string]$CondaEnv = "brep_mfr_pyg",
    [int]$Seed = 42,
    [int]$ValidationMaxFiles = 0,
    [switch]$SkipValidation,
    [switch]$Apply
)

$ErrorActionPreference = "Stop"
$Repo = (Get-Item $PSScriptRoot).Parent.Parent.FullName
Set-Location $Repo

$PrepareScript = Join-Path $PSScriptRoot "prepare_model_a_unique_abc_dataset.py"
$ValidateScript = Join-Path $PSScriptRoot "validate_a1_a3_finetune_data.py"

foreach ($RequiredDirectory in @(
    $ModelARoot,
    (Join-Path $ModelARoot "pyg"),
    $ExpandedRoot,
    (Join-Path $ExpandedRoot "pyg"),
    $NewAbcJsonDir
)) {
    if (-not (Test-Path -LiteralPath $RequiredDirectory -PathType Container)) {
        throw "Required directory not found: $RequiredDirectory"
    }
}
foreach ($Split in @("train", "val", "test")) {
    $SplitPath = Join-Path $ModelARoot "$Split.txt"
    if (-not (Test-Path -LiteralPath $SplitPath -PathType Leaf)) {
        throw "Model A split file not found: $SplitPath"
    }
}
$StockSplit = Join-Path $ExpandedRoot "stock_only_test.txt"
if (-not (Test-Path -LiteralPath $StockSplit -PathType Leaf)) {
    throw "Expanded stock-only split not found: $StockSplit"
}

$PythonArgs = @(
    $PrepareScript,
    "--model-a-root", $ModelARoot,
    "--expanded-root", $ExpandedRoot,
    "--new-abc-json-dir", $NewAbcJsonDir,
    "--output-root", $OutputRoot,
    "--link-mode", $LinkMode.ToLowerInvariant(),
    "--seed", [string]$Seed
)
if ($Apply) {
    $PythonArgs += @("--apply", "--audit-stock-labels")
}

Write-Host "Model A + unique new ABC dataset preparation"
Write-Host "  Model A root:       $ModelARoot"
Write-Host "  expanded root:      $ExpandedRoot"
Write-Host "  new ABC JSONs:      $NewAbcJsonDir"
Write-Host "  output root:        $OutputRoot"
Write-Host "  graph mode:         $LinkMode"
Write-Host "  operation:          $(if ($Apply) { 'APPLY' } else { 'DRY RUN' })"
Write-Host ""

& conda run --no-capture-output -n $CondaEnv python @PythonArgs
if ($LASTEXITCODE -ne 0) {
    throw "Model A + unique ABC preparation failed with exit code $LASTEXITCODE"
}

if (-not $Apply) {
    Write-Host ""
    Write-Host "Dry run succeeded. Re-run with -Apply to create the dataset."
    exit 0
}

if (-not $SkipValidation) {
    $ValidationArgs = @(
        $ValidateScript,
        "--dataset-root", $OutputRoot,
        "--pt-subdir", "pyg",
        "--num-classes", "3",
        "--report-a3-cap", "768",
        "--quarantine-invalid"
    )
    if ($ValidationMaxFiles -gt 0) {
        $ValidationArgs += @("--max-files", [string]$ValidationMaxFiles)
    }
    Write-Host ""
    Write-Host "Validating prepared A1/A3 graphs and split coverage..."
    & conda run --no-capture-output -n $CondaEnv python @ValidationArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Prepared-dataset validation failed with exit code $LASTEXITCODE"
    }
}

Write-Host ""
Write-Host "Dataset preparation complete: $OutputRoot"
Write-Host "Training launcher:"
Write-Host "  scripts\threads\train_model_a_unique_abc_finetune.ps1"
