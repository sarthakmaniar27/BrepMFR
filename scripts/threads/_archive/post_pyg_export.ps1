# Run AFTER json_to_brepmfr_pyg_optimized.py has finished (all .pt present).
# Uses conda env brep_mfr_pyg. Adjust paths if needed.

$ErrorActionPreference = "Stop"
# $PSScriptRoot = .../BrepMFR_PyG/scripts/threads
$Repo = (Get-Item $PSScriptRoot).Parent.Parent.FullName
if (-not (Test-Path (Join-Path $Repo "segmentation.py"))) {
    $Repo = "C:\Users\D58\Desktop\BrepMFR_PyG"
}
Set-Location $Repo

$JsonDir = "D:\threads\json"
$PygDir = "D:\threads\pyg"
$LabelDir = "D:\threads\label"

$nJson = (Get-ChildItem -Path $JsonDir -Filter "*.json" -File).Count
$nPyg = (Get-ChildItem -Path $PygDir -Filter "*.pt" -File).Count
$nLabel = (Get-ChildItem -Path $LabelDir -Filter "*.json" -File -ErrorAction SilentlyContinue).Count

Write-Host "JSON files: $nJson  |  PYG .pt: $nPyg  |  label JSON: $nLabel"
if ($nPyg -lt $nJson) {
    Write-Error "Conversion not complete ($nPyg < $nJson). Wait for json_to_brepmfr_pyg_optimized.py to finish."
    exit 1
}

conda run -n brep_mfr_pyg python scripts/threads/make_random_splits.py --pyg-dir $PygDir --out-dir $PygDir --seed 42

New-Item -ItemType Directory -Force -Path "artifacts/class_weights/thread" | Out-Null
conda run -n brep_mfr_pyg python scripts/training/compute_class_weights.py `
    --dataset_path $PygDir `
    --split train `
    --num_classes 2 `
    --alpha 0.5 `
    --num_workers 0 `
    --out artifacts/class_weights/thread/source_train_alpha05.json

conda run -n brep_mfr_pyg python scripts/threads/count_thread_label_distribution.py --pyg-dir $PygDir

Write-Host "`nNext: run Stage 1 training (see scripts/threads/README.md section 'Stage 1 command')."
