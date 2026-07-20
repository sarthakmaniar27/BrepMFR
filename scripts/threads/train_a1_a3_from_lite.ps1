param(
    [Parameter(Mandatory = $true)]
    [string]$Checkpoint,
    [string]$DatasetRoot = "Z:\thread_and_text\no_a2",
    [string]$ClassWeights = "artifacts/class_weights/thread_text/source_train_alpha05.json",
    [string]$CondaEnv = "brep_mfr_pyg",
    [string]$RunName = "thread_text_a1_a3_finetune_$(Get-Date -Format 'yyyyMMdd_HHmmss')",
    [int]$MaxEpochs = 30,
    [int]$MaxNodesForA3 = 768
)

$ErrorActionPreference = "Stop"
$Repo = (Get-Item $PSScriptRoot).Parent.Parent.FullName
Set-Location $Repo

if (-not (Test-Path $Checkpoint -PathType Leaf)) {
    throw "Lite checkpoint not found: $Checkpoint"
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

$trainArgs = @(
    "segmentation.py", "train",
    "--dataset_path", $DatasetRoot,
    "--pt_subdir", "pyg",
    "--num_classes", "3",
    "--drop_invalid_graphs",
    "--class_weights_path", $ClassWeights,
    "--batch_size", "16",
    "--accumulate_grad_batches", "4",
    "--precision", "16-mixed",
    "--max_epochs", [string]$MaxEpochs,
    "--num_workers", "0",
    "--log_every_n_steps", "50",
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
    "--optimizer_warmup_steps", "1000",
    "--a1_a3_ramp_epochs", "5",
    "--a1_a3_start_scale", "0.1",
    "--max_nodes_for_a3", [string]$MaxNodesForA3,
    "--length_bucket_batching",
    "--pre_train", $Checkpoint,
    "--run_name", $RunName
)

Write-Host "Starting a fresh fine-tuning run from lite weights."
Write-Host "  checkpoint: $Checkpoint"
Write-Host "  dataset:    $DatasetRoot"
Write-Host "  run name:   $RunName"
Write-Host "  A1/A3 ramp: 0.1 -> 1.0 over 5 epochs"
Write-Host "  learning rates: backbone=1e-4, A1/A3=1e-3"
Write-Host "  A3 cap: $MaxNodesForA3 faces (A1 remains active above the cap)"
Write-Host ""

& conda run -n $CondaEnv python @trainArgs
if ($LASTEXITCODE -ne 0) {
    throw "A1+A3 fine-tuning failed with exit code $LASTEXITCODE"
}
