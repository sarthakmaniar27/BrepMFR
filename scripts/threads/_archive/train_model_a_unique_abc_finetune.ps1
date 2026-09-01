param(
    [string]$DatasetRoot = "Z:\thread_and_text\abc_for_modelA_finetuning",
    [string]$Checkpoint = "",
    [string]$ResumeFromCheckpoint = "",
    [string]$CondaEnv = "brep_mfr_pyg",
    [string]$RunName = "thread_text_model_a_unique_abc_finetune_v1",
    [int]$MaxEpochs = 8,
    [double]$LearningRate = 0.00002,
    [int]$OptimizerWarmupSteps = 500,
    [int]$MaxNodesForA3 = 768,
    [int]$BatchSize = 64,
    [int]$BatchNodeSqBudget = 4000000,
    [int]$AccumulateGradBatches = 1,
    [int]$DataLoaderWorkers = 4,
    [int]$PrefetchFactor = 2,
    [bool]$PersistentWorkers = $true,
    [bool]$FusedAdamW = $false,
    [bool]$CudnnBenchmark = $false,
    [int]$LimitTrainBatches = 0
)

$ErrorActionPreference = "Stop"
$Repo = (Get-Item $PSScriptRoot).Parent.Parent.FullName
Set-Location $Repo

if ([string]::IsNullOrWhiteSpace($Checkpoint) -and
    [string]::IsNullOrWhiteSpace($ResumeFromCheckpoint)) {
    $Checkpoint = Join-Path $Repo "model_checkpoints\abc_with_no_a2\last-v1.ckpt"
}
if (-not [string]::IsNullOrWhiteSpace($Checkpoint) -and
    -not [string]::IsNullOrWhiteSpace($ResumeFromCheckpoint)) {
    throw "Specify -Checkpoint for a fresh fine-tune or -ResumeFromCheckpoint for an exact resume, not both."
}

$SummaryPath = Join-Path $DatasetRoot "preparation_summary.json"
if (-not (Test-Path -LiteralPath $SummaryPath -PathType Leaf)) {
    throw "Prepared-dataset summary not found: $SummaryPath. Run prepare_model_a_unique_abc_finetune_data.ps1 -Apply first."
}

$Summary = Get-Content -LiteralPath $SummaryPath -Raw | ConvertFrom-Json
if ($Summary.counts.unique_new_abc_added -le 0) {
    throw "Prepared dataset reports no unique new ABC graphs: $SummaryPath"
}

$GenericLauncher = Join-Path $PSScriptRoot "train_new_abc_finetune.ps1"
$Launch = @{
    DatasetRoot = $DatasetRoot
    CondaEnv = $CondaEnv
    RunName = $RunName
    MaxEpochs = $MaxEpochs
    LearningRate = $LearningRate
    A1A3LearningRate = $LearningRate
    OptimizerWarmupSteps = $OptimizerWarmupSteps
    MaxNodesForA3 = $MaxNodesForA3
    BatchSize = $BatchSize
    BatchNodeSqBudget = $BatchNodeSqBudget
    AccumulateGradBatches = $AccumulateGradBatches
    DataLoaderWorkers = $DataLoaderWorkers
    PrefetchFactor = $PrefetchFactor
    PersistentWorkers = $PersistentWorkers
    FusedAdamW = $FusedAdamW
    CudnnBenchmark = $CudnnBenchmark
    DropInvalidGraphs = $false
    LimitTrainBatches = $LimitTrainBatches
    UseClassWeights = $false
}
if ([string]::IsNullOrWhiteSpace($ResumeFromCheckpoint)) {
    $Launch.Checkpoint = $Checkpoint
} else {
    $Launch.ResumeFromCheckpoint = $ResumeFromCheckpoint
}

Write-Host "Model A -> unique new ABC fine-tuning"
Write-Host "  Model A replay:     $($Summary.counts.model_a_graphs)"
Write-Host "  unique new ABC:     $($Summary.counts.unique_new_abc_added)"
Write-Host "  training graphs:    $($Summary.counts.training_graphs)"
Write-Host "  strict stock eval:  $($Summary.counts.stock_eval_strict) (held out)"
Write-Host "  LR:                 $LearningRate for all branches"
Write-Host "  epochs:             $MaxEpochs"
Write-Host "  class weighting:    disabled"
Write-Host ""

& $GenericLauncher @Launch
if ($LASTEXITCODE -ne 0) {
    throw "Model A unique-ABC fine-tuning failed with exit code $LASTEXITCODE"
}
