param(
    [string]$Checkpoint = "",
    [string]$ResumeFromCheckpoint = "",
    [string]$DatasetRoot = "Z:\thread_and_text\no_a2",
    [string]$ClassWeights = "artifacts/class_weights/thread_text/source_train_alpha05.json",
    [string]$CondaEnv = "brep_mfr_pyg",
    [string]$RunName = "",
    [int]$MaxEpochs = 30,
    [int]$MaxNodesForA3 = 768,
    [int]$BatchSize = 64,
    [int]$BatchNodeSqBudget = 4000000,
    [int]$AccumulateGradBatches = 1,
    [int]$OptimizerWarmupSteps = 1000,
    [int]$CheckValEveryNEpoch = 0,
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

$HasLiteCheckpoint = -not [string]::IsNullOrWhiteSpace($Checkpoint)
$HasResumeCheckpoint = -not [string]::IsNullOrWhiteSpace($ResumeFromCheckpoint)
if ($HasLiteCheckpoint -eq $HasResumeCheckpoint) {
    throw "Specify exactly one of -Checkpoint (fresh fine-tune from lite weights) or -ResumeFromCheckpoint (exact training resume)."
}

if ($CheckValEveryNEpoch -lt 0) {
    throw "CheckValEveryNEpoch must be non-negative (0 selects the safe mode-specific default)."
}
if ($CheckValEveryNEpoch -eq 0) {
    # Legacy fine-tune checkpoints used every_n_epochs=1; matching it restores
    # ModelCheckpoint state as well as the model/optimizer/loop state.
    $CheckValEveryNEpoch = if ($HasResumeCheckpoint) { 1 } else { 2 }
}

if ($HasResumeCheckpoint) {
    if (-not (Test-Path $ResumeFromCheckpoint -PathType Leaf)) {
        throw "Resume checkpoint not found: $ResumeFromCheckpoint"
    }
    if ([string]::IsNullOrWhiteSpace($RunName)) {
        $RunName = Split-Path (Split-Path $ResumeFromCheckpoint -Parent) -Leaf
    }
} else {
    if (-not (Test-Path $Checkpoint -PathType Leaf)) {
        throw "Lite checkpoint not found: $Checkpoint"
    }
    if ([string]::IsNullOrWhiteSpace($RunName)) {
        $RunName = "thread_text_a1_a3_finetune_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
    }
}

if (-not (Test-Path (Join-Path $DatasetRoot "pyg") -PathType Container)) {
    throw "A1+A3 dataset not found under: $DatasetRoot\pyg"
}
foreach ($name in @("train.txt", "val.txt", "test.txt")) {
    if (-not (Test-Path (Join-Path $DatasetRoot $name) -PathType Leaf)) {
        throw "Missing split list: $DatasetRoot\$name"
    }
}
if (-not (Test-Path $ClassWeights -PathType Leaf)) {
    throw "Class-weights file not found: $ClassWeights"
}
if ($BatchSize -le 0 -or $BatchNodeSqBudget -le 0 -or $AccumulateGradBatches -le 0) {
    throw "BatchSize, BatchNodeSqBudget, and AccumulateGradBatches must all be greater than zero."
}
if ($DataLoaderWorkers -lt 0 -or $PrefetchFactor -le 0) {
    throw "DataLoaderWorkers must be non-negative and PrefetchFactor must be greater than zero."
}

$trainArgs = @(
    "segmentation.py", "train",
    "--dataset_path", $DatasetRoot,
    "--pt_subdir", "pyg",
    "--num_classes", "5",
    "--drop_invalid_graphs",
    "--class_weights_path", $ClassWeights,
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
    "--check_val_every_n_epoch", [string]$CheckValEveryNEpoch,
    "--dropout", "0.3",
    "--attention_dropout", "0.3",
    "--act-dropout", "0.3",
    "--d_model", "512",
    "--dim_node", "256",
    "--n_heads", "32",
    "--n_layers_encode", "8",
    "--warmup_freeze_epochs", "0",
    "--learning_rate", "0.0001",
    "--a1_a3_learning_rate", "0.001",
    "--optimizer_warmup_steps", [string]$OptimizerWarmupSteps,
    "--a1_a3_ramp_epochs", "5",
    "--a1_a3_start_scale", "0.1",
    "--max_nodes_for_a3", [string]$MaxNodesForA3,
    "--loss_type", "ce",
    "--length_bucket_batching",
    "--run_name", $RunName
)

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
if ($HasResumeCheckpoint) {
    $trainArgs += @("--resume_from_checkpoint", $ResumeFromCheckpoint)
} else {
    $trainArgs += @("--pre_train", $Checkpoint)
}

if ($HasResumeCheckpoint) {
    Write-Host "Resuming the existing A1+A3 fine-tuning run exactly."
} else {
    Write-Host "Starting a fresh fine-tuning run from lite weights."
}
Write-Host "  checkpoint:      $(if ($HasResumeCheckpoint) { $ResumeFromCheckpoint } else { $Checkpoint })"
Write-Host "  checkpoint mode: $(if ($HasResumeCheckpoint) { 'exact resume (model + optimizer + epoch + global step)' } else { 'lite weights only (fresh optimizer and epoch 0)' })"
Write-Host "  dataset:         $DatasetRoot"
Write-Host "  class weights:   $ClassWeights"
Write-Host "  run name:        $RunName"
Write-Host "  A1/A3 ramp:      0.1 -> 1.0 over 5 epochs"
Write-Host "  learning rates:  backbone=1e-4, A1/A3=1e-3"
Write-Host "  A3 cap:          $MaxNodesForA3 faces (A1 remains active above the cap)"
Write-Host "  batching:        adaptive N^2 budget=$BatchNodeSqBudget, max graphs=$BatchSize"
Write-Host "  accumulation:    $AccumulateGradBatches"
Write-Host "  validation:      every $CheckValEveryNEpoch epoch(s)"
if ($LimitTrainBatches -gt 0) { Write-Host "  benchmark cap:   $LimitTrainBatches train batches/epoch" }
Write-Host "  data loading:    $DataLoaderWorkers worker(s), prefetch=$PrefetchFactor, pinned memory, persistent=$PersistentWorkers"
Write-Host "  CUDA fast path:  fused AdamW=$FusedAdamW, cuDNN benchmark=$CudnnBenchmark, TF32=true"
Write-Host ""

& conda run --no-capture-output -n $CondaEnv python @trainArgs
if ($LASTEXITCODE -ne 0) {
    throw "A1+A3 fine-tuning failed with exit code $LASTEXITCODE"
}
