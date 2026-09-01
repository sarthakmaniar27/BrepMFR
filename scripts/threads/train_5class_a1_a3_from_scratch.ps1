param(
    [string]$DatasetRoot = "Z:\thread_and_text\cadsynth_with_fillets_and_champer\five_class_a1_a3\no_a2",
    [string]$ClassWeights = "artifacts/class_weights/thread_text/cadsynth_5class_a1_a3_train_alpha05.json",
    [string]$CondaEnv = "brep_mfr_pyg",
    [string]$RunName = "",
    [string]$ResumeFromCheckpoint = "",
    [int]$MaxEpochs = 100,
    [int]$Seed = 42,
    [double]$LearningRate = 0.002,
    [int]$OptimizerWarmupSteps = 1000,
    [int]$MaxNodesForA3 = 768,
    [int]$BatchSize = 64,
    [int]$BatchNodeSqBudget = 4000000,
    [int]$AccumulateGradBatches = 1,
    [int]$CheckValEveryNEpoch = 2,
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

if (-not (Test-Path -LiteralPath (Join-Path $DatasetRoot "pyg") -PathType Container)) {
    throw "Five-class A1+A3 graph directory not found: $DatasetRoot\pyg"
}
foreach ($name in @("train.txt", "val.txt", "test.txt")) {
    if (-not (Test-Path -LiteralPath (Join-Path $DatasetRoot $name) -PathType Leaf)) {
        throw "Dataset split is missing: $DatasetRoot\$name"
    }
}
if (-not (Test-Path -LiteralPath $ClassWeights -PathType Leaf)) {
    throw "Five-class weights file not found: $ClassWeights"
}
if ($MaxEpochs -le 0 -or $BatchSize -le 0 -or $BatchNodeSqBudget -le 0) {
    throw "MaxEpochs, BatchSize, and BatchNodeSqBudget must be greater than zero."
}
if ($AccumulateGradBatches -le 0 -or $LearningRate -le 0) {
    throw "AccumulateGradBatches and LearningRate must be greater than zero."
}
if ($DataLoaderWorkers -lt 0 -or $PrefetchFactor -le 0) {
    throw "DataLoaderWorkers must be non-negative and PrefetchFactor must be greater than zero."
}
if ($CheckValEveryNEpoch -le 0 -or $OptimizerWarmupSteps -lt 0) {
    throw "CheckValEveryNEpoch must be positive and OptimizerWarmupSteps must be non-negative."
}
if ($MaxNodesForA3 -lt 0) {
    throw "MaxNodesForA3 must be non-negative (0 disables the cap)."
}

$HasResume = -not [string]::IsNullOrWhiteSpace($ResumeFromCheckpoint)
if ($HasResume -and -not (Test-Path -LiteralPath $ResumeFromCheckpoint -PathType Leaf)) {
    throw "Resume checkpoint not found: $ResumeFromCheckpoint"
}
if ([string]::IsNullOrWhiteSpace($RunName)) {
    if ($HasResume) {
        $RunName = Split-Path (Split-Path $ResumeFromCheckpoint -Parent) -Leaf
    } else {
        $RunName = "stock_thread_text_chamfer_fillet_a1_a3_scratch_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
    }
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
    "--learning_rate", [string]$LearningRate,
    "--a1_a3_learning_rate", [string]$LearningRate,
    "--optimizer_warmup_steps", [string]$OptimizerWarmupSteps,
    "--a1_a3_ramp_epochs", "0",
    "--max_nodes_for_a3", [string]$MaxNodesForA3,
    "--loss_type", "ce",
    "--length_bucket_batching",
    "--csv_log",
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
if ($HasResume) {
    $trainArgs += @("--resume_from_checkpoint", $ResumeFromCheckpoint)
} else {
    $trainArgs += @("--full_a1_a3_from_scratch", "--seed", [string]$Seed)
}

Write-Host "Five-class A1+A3 training"
Write-Host "  mode:             $(if ($HasResume) { 'exact resume of scratch run' } else { 'random initialization from scratch' })"
Write-Host "  pre-trained data: none"
Write-Host "  ABC data:         none"
Write-Host "  dataset:          $DatasetRoot"
Write-Host "  classes:          0=stock, 1=thread, 2=text, 3=chamfer, 4=fillet"
Write-Host "  class weights:    $ClassWeights"
Write-Host "  run name:         $RunName"
Write-Host "  epochs:           $MaxEpochs"
Write-Host "  learning rate:    $LearningRate (backbone and A1/A3)"
Write-Host "  A1/A3 scale:      1.0 from epoch 0"
Write-Host "  A3 cap:           $(if ($MaxNodesForA3 -eq 0) { 'disabled (A3 on every graph; highest VRAM use)' } else { "$MaxNodesForA3 faces (A1 remains enabled above the cap)" })"
Write-Host "  batching:         adaptive N^2 budget=$BatchNodeSqBudget, max graphs=$BatchSize"
Write-Host "  validation:       every $CheckValEveryNEpoch epoch(s)"
Write-Host "  checkpoint:       $(if ($HasResume) { $ResumeFromCheckpoint } else { 'none' })"
Write-Host ""

& conda run --no-capture-output -n $CondaEnv python @trainArgs
if ($LASTEXITCODE -ne 0) {
    throw "Five-class A1+A3 training failed with exit code $LASTEXITCODE"
}
