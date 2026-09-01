param(
    [string]$Checkpoint = "",
    [string]$ResumeFromCheckpoint = "",

    [Parameter(Mandatory = $true)]
    [string]$DatasetRoot,

    [string]$ClassWeights = "",
    [bool]$UseClassWeights = $false,
    [string]$CondaEnv = "brep_mfr_pyg",
    [string]$RunName = "",
    [int]$MaxEpochs = 15,
    [double]$LearningRate = 0.0001,
    [double]$A1A3LearningRate = 0.0001,
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
    [bool]$DropInvalidGraphs = $true,
    [int]$LimitTrainBatches = 0
)

$ErrorActionPreference = "Stop"
$Repo = (Get-Item $PSScriptRoot).Parent.Parent.FullName
Set-Location $Repo

$HasPreTrain = -not [string]::IsNullOrWhiteSpace($Checkpoint)
$HasResume = -not [string]::IsNullOrWhiteSpace($ResumeFromCheckpoint)
if ($HasPreTrain -eq $HasResume) {
    throw "Specify exactly one of -Checkpoint (new fine-tune) or -ResumeFromCheckpoint (exact resume)."
}
if ($HasPreTrain -and -not (Test-Path -LiteralPath $Checkpoint -PathType Leaf)) {
    throw "Fine-tuning checkpoint not found: $Checkpoint"
}
if ($HasResume -and -not (Test-Path -LiteralPath $ResumeFromCheckpoint -PathType Leaf)) {
    throw "Resume checkpoint not found: $ResumeFromCheckpoint"
}
if (-not (Test-Path -LiteralPath (Join-Path $DatasetRoot "pyg") -PathType Container)) {
    throw "Combined no_a2 graph directory not found: $DatasetRoot\pyg"
}
foreach ($name in @("train.txt", "val.txt", "test.txt")) {
    if (-not (Test-Path -LiteralPath (Join-Path $DatasetRoot $name) -PathType Leaf)) {
        throw "Dataset split is missing: $DatasetRoot\$name"
    }
}
if ($UseClassWeights) {
    if ([string]::IsNullOrWhiteSpace($ClassWeights)) {
        throw "-UseClassWeights true requires -ClassWeights."
    }
    if (-not (Test-Path -LiteralPath $ClassWeights -PathType Leaf)) {
        throw "Class-weights JSON not found: $ClassWeights"
    }
}
if ($LearningRate -le 0 -or $A1A3LearningRate -le 0) {
    throw "LearningRate and A1A3LearningRate must be greater than zero."
}
if ($MaxEpochs -le 0 -or $BatchSize -le 0 -or $BatchNodeSqBudget -le 0) {
    throw "MaxEpochs, BatchSize, and BatchNodeSqBudget must be greater than zero."
}

if ([string]::IsNullOrWhiteSpace($RunName)) {
    if ($HasResume) {
        $RunName = Split-Path (Split-Path $ResumeFromCheckpoint -Parent) -Leaf
    } else {
        $RunName = "thread_text_new_abc_finetune_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
    }
}

$trainArgs = @(
    "segmentation.py", "train",
    "--dataset_path", $DatasetRoot,
    "--pt_subdir", "pyg",
    "--num_classes", "5",
    "--batch_size", [string]$BatchSize,
    "--batch_node_sq_budget", [string]$BatchNodeSqBudget,
    "--accumulate_grad_batches", [string]$AccumulateGradBatches,
    "--precision", "16-mixed",
    "--max_epochs", [string]$MaxEpochs,
    "--num_workers", [string]$DataLoaderWorkers,
    "--pin_memory",
    "--allow_tf32",
    "--log_every_n_steps", "50",
    "--num_sanity_val_steps", "0",
    "--check_val_every_n_epoch", "1",
    "--dropout", "0.3",
    "--attention_dropout", "0.3",
    "--act-dropout", "0.3",
    "--d_model", "512",
    "--dim_node", "256",
    "--n_heads", "32",
    "--n_layers_encode", "8",
    "--warmup_freeze_epochs", "0",
    "--learning_rate", [string]$LearningRate,
    "--a1_a3_learning_rate", [string]$A1A3LearningRate,
    "--optimizer_warmup_steps", [string]$OptimizerWarmupSteps,
    "--a1_a3_ramp_epochs", "0",
    "--max_nodes_for_a3", [string]$MaxNodesForA3,
    "--loss_type", "ce",
    "--length_bucket_batching",
    "--csv_log",
    "--run_name", $RunName
)
if ($DropInvalidGraphs) {
    $trainArgs += "--drop_invalid_graphs"
}
if ($UseClassWeights) {
    $trainArgs += @("--class_weights_path", $ClassWeights)
}
if ($DataLoaderWorkers -gt 0) {
    $trainArgs += @("--dataloader_prefetch_factor", [string]$PrefetchFactor)
    if ($PersistentWorkers) {
        $trainArgs += "--persistent_workers"
    }
}
if ($FusedAdamW) {
    $trainArgs += "--fused_adamw"
}
if ($CudnnBenchmark) {
    $trainArgs += "--cudnn_benchmark"
}
if ($LimitTrainBatches -gt 0) {
    $trainArgs += @("--limit_train_batches", [string]$LimitTrainBatches)
}
if ($HasResume) {
    $trainArgs += @("--resume_from_checkpoint", $ResumeFromCheckpoint)
} else {
    $trainArgs += @("--pre_train", $Checkpoint)
}

Write-Host "New ABC fine-tuning"
Write-Host "  mode:             $(if ($HasResume) { 'exact resume' } else { 'fresh optimizer from model weights' })"
Write-Host "  checkpoint:       $(if ($HasResume) { $ResumeFromCheckpoint } else { $Checkpoint })"
Write-Host "  dataset:          $DatasetRoot"
Write-Host "  run name:         $RunName"
Write-Host "  epochs:           $MaxEpochs"
Write-Host "  learning rates:   base=$LearningRate A1/A3=$A1A3LearningRate"
Write-Host "  A1/A3 scale:      1.0 from epoch 0 (no lite ramp)"
Write-Host "  loss:             CE, class weights=$(if ($UseClassWeights) { $ClassWeights } else { 'disabled' })"
Write-Host "  validation:       every epoch"
Write-Host "  A3 cap:           $MaxNodesForA3"
Write-Host ""

& conda run --no-capture-output -n $CondaEnv python @trainArgs
if ($LASTEXITCODE -ne 0) {
    throw "New ABC fine-tuning failed with exit code $LASTEXITCODE"
}
