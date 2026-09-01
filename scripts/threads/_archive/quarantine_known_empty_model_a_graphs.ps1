param(
    [string]$DatasetRoot = "Z:\thread_and_text\abc_for_modelA_finetuning",
    [string]$ModelARoot = "Z:\thread_and_text\no_a2",
    [string]$ExpandedRoot = "Z:\thread_and_text\no_a2_72k_plus_new_abc",
    [string]$CondaEnv = "brep_mfr_pyg",
    [int]$ExpectedCount = 25,
    [switch]$Apply
)

$ErrorActionPreference = "Stop"
$Repo = (Get-Item $PSScriptRoot).Parent.Parent.FullName
Set-Location $Repo

$PythonArgs = @(
    (Join-Path $PSScriptRoot "quarantine_known_empty_model_a_graphs.py"),
    "--dataset-root", $DatasetRoot,
    "--model-a-root", $ModelARoot,
    "--expanded-root", $ExpandedRoot,
    "--expected-count", [string]$ExpectedCount
)
if ($Apply) {
    $PythonArgs += "--apply"
}

& conda run --no-capture-output -n $CondaEnv python @PythonArgs
if ($LASTEXITCODE -ne 0) {
    throw "Targeted empty-graph quarantine failed with exit code $LASTEXITCODE"
}
