param(
    [Parameter(Mandatory = $true)]
    [string]$TargetRepo,
    [string]$CondaEnv = "brep_mfr_pyg"
)

$ErrorActionPreference = "Stop"
$SourceRepo = (Get-Item $PSScriptRoot).Parent.Parent.FullName
$TargetRepo = [IO.Path]::GetFullPath($TargetRepo).TrimEnd('\')
if (-not (Test-Path $TargetRepo -PathType Container)) {
    throw "Target repository not found: $TargetRepo"
}
if ([IO.Path]::GetFullPath($SourceRepo).TrimEnd('\') -eq $TargetRepo) {
    throw "Source and target repositories are identical; no synchronization is needed."
}

$relativeFiles = @(
    "segmentation.py",
    "data/collator.py",
    "data/dataset.py",
    "data/utils.py",
    "data/length_bucket_batch_sampler.py",
    "models/brepseg_model.py",
    "models/tensorboard_media.py",
    "scripts/smoke/smoke_training_optimizations.py",
    "models/modules/brep_encoder.py",
    "models/modules/layers/brep_encoder_layer.py",
    "models/modules/layers/multihead_attention.py",
    "scripts/threads/train_no_a2_from_scratch.ps1"
)

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$backupRoot = Join-Path $TargetRepo "a1_a3_sync_backup_$timestamp"
Write-Host "Synchronizing coordinated A1/A3 training files..."
Write-Host "  source: $SourceRepo"
Write-Host "  target: $TargetRepo"
Write-Host "  backup: $backupRoot"

foreach ($relative in $relativeFiles) {
    $source = Join-Path $SourceRepo $relative
    $target = Join-Path $TargetRepo $relative
    if (-not (Test-Path $source -PathType Leaf)) {
        throw "Required source file missing: $source"
    }
    if (Test-Path $target -PathType Leaf) {
        $backup = Join-Path $backupRoot $relative
        New-Item -ItemType Directory -Force -Path (Split-Path $backup -Parent) | Out-Null
        Copy-Item -LiteralPath $target -Destination $backup -Force
    }
    New-Item -ItemType Directory -Force -Path (Split-Path $target -Parent) | Out-Null
    Copy-Item -LiteralPath $source -Destination $target -Force
    Write-Host "  synced: $relative"
}

$pythonFiles = @(
    "segmentation.py",
    "data/collator.py",
    "data/dataset.py",
    "data/utils.py",
    "data/length_bucket_batch_sampler.py",
    "models/brepseg_model.py",
    "models/tensorboard_media.py",
    "scripts/smoke/smoke_training_optimizations.py",
    "models/modules/brep_encoder.py",
    "models/modules/layers/brep_encoder_layer.py",
    "models/modules/layers/multihead_attention.py"
) | ForEach-Object { Join-Path $TargetRepo $_ }

& conda run --no-capture-output -n $CondaEnv python -m py_compile @pythonFiles
if ($LASTEXITCODE -ne 0) {
    throw "Python syntax verification failed in the synchronized target."
}

& conda run --no-capture-output -n $CondaEnv python `
    (Join-Path $TargetRepo "scripts/smoke/smoke_training_optimizations.py")
if ($LASTEXITCODE -ne 0) {
    throw "Training optimization parity smoke failed in the synchronized target."
}

$requiredTokens = @{
    "segmentation.py" = @("--max_nodes_for_a3", "--a1_a3_learning_rate", "--batch_node_sq_budget")
    "data/collator.py" = @("max_nodes_for_a3")
    "data/dataset.py" = @("max_nodes_for_a3")
    "data/utils.py" = @("_CANONICAL_ROTATIONS")
    "data/length_bucket_batch_sampler.py" = @("node_sq_budget", "a3_node_cap")
    "models/brepseg_model.py" = @("a1_a3_learning_rate", "set_a1_a3_scale")
    "models/tensorboard_media.py" = @("log_segmentation_val_confusion")
    "models/modules/brep_encoder.py" = @("max_nodes_for_a3")
    "models/modules/layers/brep_encoder_layer.py" = @("a1_a3_scale", "max_nodes_for_a3")
    "models/modules/layers/multihead_attention.py" = @("scaled_dot_product_attention")
}
foreach ($relative in $requiredTokens.Keys) {
    $content = Get-Content -LiteralPath (Join-Path $TargetRepo $relative) -Raw
    foreach ($token in $requiredTokens[$relative]) {
        if (-not $content.Contains($token)) {
            throw "Compatibility verification failed: $relative does not contain '$token'."
        }
    }
}

Write-Host ""
Write-Host "A1/A3 training-code synchronization passed."
Write-Host "The previous target files are preserved under: $backupRoot"
