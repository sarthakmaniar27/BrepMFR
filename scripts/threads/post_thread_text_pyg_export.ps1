# Run AFTER json_to_brepmfr_pyg_optimized.py has finished (all .pt present).
# Thread + text (3-class): splits, class weights, label recount.
# Uses conda env brep_mfr_pyg. Adjust paths if needed.

$ErrorActionPreference = "Stop"
$Repo = (Get-Item $PSScriptRoot).Parent.Parent.FullName
if (-not (Test-Path (Join-Path $Repo "segmentation.py"))) {
    $Repo = "C:\Users\D58\Desktop\BrepMFR_PyG"
}
Set-Location $Repo

$JsonDir = "\\Gr-sw66464\d\thread_and_text\thread_text_merged"
$PygDir = "E:\thread_text_merged\pyg"
$LabelDir = "E:\thread_text_merged\label"
# Split lists (train/val/test.txt) live next to the pyg folder (e.g. .../lite/), not inside pyg/
$DataRoot = Split-Path $PygDir -Parent

$nJson = (Get-ChildItem -Path $JsonDir -Filter "*.json" -File).Count
$nPyg = (Get-ChildItem -Path $PygDir -Filter "*.pt" -File).Count
$nLabel = (Get-ChildItem -Path $LabelDir -Filter "*.json" -File -ErrorAction SilentlyContinue).Count

Write-Host "JSON files: $nJson  |  PYG .pt: $nPyg  |  label JSON: $nLabel"
if ($nPyg -lt $nJson) {
    Write-Error "Conversion not complete ($nPyg < $nJson). Wait for json_to_brepmfr_pyg_optimized.py to finish."
    exit 1
}

conda run -n brep_mfr python scripts/threads/make_random_splits.py --pyg-dir $PygDir --out-dir $DataRoot --seed 42

New-Item -ItemType Directory -Force -Path "artifacts/class_weights/thread_text" | Out-Null
conda run -n brep_mfr python scripts/training/compute_class_weights.py `
    --dataset_path $DataRoot `
    --split train `
    --num_classes 3 `
    --alpha 0.5 `
    --num_workers 0 `
    --out artifacts/class_weights/thread_text/source_train_alpha05.json

conda run -n brep_mfr python scripts/threads/count_thread_label_distribution.py --pyg-dir $PygDir --group "0:stock,1:thread,2:text"

Write-Host "`nNext: run Stage 1 training (see scripts/threads/README_thread_text.md)."
