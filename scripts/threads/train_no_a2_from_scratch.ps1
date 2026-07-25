param(
    [string]$DatasetRoot = "D:\thread_and_text\no_a2_large",
    [string]$ClassWeights = "artifacts/class_weights/thread_text/no_a2_large_70k_train_alpha05.json",
    [string]$CondaEnv = "brep_mfr_pyg",
    [string]$RunName = "thread_text_no_a2_70k_scratch_$(Get-Date -Format 'yyyyMMdd_HHmmss')",
    [int]$MaxEpochs = 100,
    [int]$MaxNodesForA3 = 768,
    [int]$BatchSize = 64,
    [int]$BatchNodeSqBudget = 4000000,
    [int]$AccumulateGradBatches = 1,
    [int]$OptimizerWarmupSteps = 1000,
    [int]$CheckValEveryNEpoch = 2,
    [int]$DataLoaderWorkers = 4,
    [int]$PrefetchFactor = 2,
    [bool]$PersistentWorkers = $true,
    [bool]$FusedAdamW = $false,
    [bool]$CudnnBenchmark = $false,
    [int]$LimitTrainBatches = 0,
    [string]$ResumeFromCheckpoint = "",
    [string]$PreTrain = ""
)

$ErrorActionPreference = "Stop"
$Repo = (Get-Item $PSScriptRoot).Parent.Parent.FullName
Set-Location $Repo

if (-not (Test-Path (Join-Path $DatasetRoot "pyg") -PathType Container)) {
    throw "no_a2 dataset not found: $DatasetRoot\pyg"
}
foreach ($name in @("train.txt", "val.txt", "test.txt")) {
    if (-not (Test-Path (Join-Path $DatasetRoot $name) -PathType Leaf)) {
        throw "Missing split list: $DatasetRoot\$name"
    }
}
if (-not (Test-Path $ClassWeights -PathType Leaf)) {
    throw "Class-weights file not found: $ClassWeights"
}

$trainArgs = @(
    "segmentation.py", "train",
    "--dataset_path", $DatasetRoot,
    "--pt_subdir", "pyg",
    "--num_classes", "3",
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
    "--learning_rate", "0.002",
    "--a1_a3_learning_rate", "0.002",
    "--optimizer_warmup_steps", [string]$OptimizerWarmupSteps,
    "--a1_a3_ramp_epochs", "0",
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
if ($ResumeFromCheckpoint -and $PreTrain) {
    throw "Specify only one of -ResumeFromCheckpoint or -PreTrain."
}
if ($ResumeFromCheckpoint) {
    if (-not (Test-Path $ResumeFromCheckpoint -PathType Leaf)) {
        throw "Resume checkpoint not found: $ResumeFromCheckpoint"
    }
    $trainArgs += @("--resume_from_checkpoint", $ResumeFromCheckpoint)
} elseif ($PreTrain) {
    if (-not (Test-Path $PreTrain -PathType Leaf)) {
        throw "PreTrain checkpoint not found: $PreTrain"
    }
    $trainArgs += @("--pre_train", $PreTrain)
}

if ($ResumeFromCheckpoint) {
    Write-Host "Resuming existing A1+A3 training run (continuing epoch & optimizer state)."
} elseif ($PreTrain) {
    Write-Host "Starting new training run initialized from pre-trained weights (epoch 0)."
} else {
    Write-Host "Starting a completely new A1+A3 training run."
}
Write-Host "  dataset:       $DatasetRoot"
Write-Host "  class weights: $ClassWeights"
Write-Host "  run name:      $RunName"
Write-Host "  A1/A3:         fully enabled from epoch 0"
Write-Host "  batching:      adaptive N^2 budget=$BatchNodeSqBudget, max graphs=$BatchSize"
Write-Host "  accumulation:  $AccumulateGradBatches"
Write-Host "  validation:    every $CheckValEveryNEpoch epoch(s)"
if ($LimitTrainBatches -gt 0) { Write-Host "  benchmark cap: $LimitTrainBatches train batches/epoch" }
Write-Host "  data loading:  $DataLoaderWorkers worker(s), prefetch=$PrefetchFactor, pinned memory, persistent=$PersistentWorkers"
Write-Host "  CUDA fast path: fused AdamW=$FusedAdamW, cuDNN benchmark=$CudnnBenchmark, TF32=true"
Write-Host "  checkpoint:    $(if ($ResumeFromCheckpoint) { "exact resume ($ResumeFromCheckpoint)" } elseif ($PreTrain) { "pre-train weights ($PreTrain)" } else { "none (training from scratch)" })"
Write-Host ""

& conda run --no-capture-output -n $CondaEnv python @trainArgs
if ($LASTEXITCODE -ne 0) {
    throw "From-scratch no_a2 training failed with exit code $LASTEXITCODE"
}
