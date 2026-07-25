# Fine-tuning the 72K no_a2 model with new ABC data

## 1. Goal and current data

The current classifier has three face classes:

| Class ID | Meaning |
|---:|---|
| `0` | Stock |
| `1` | Thread |
| `2` | Text |

The training update described here combines:

1. The existing approximately 72K `no_a2` PyG dataset.
2. A new folder containing approximately 25K ABC-derived JSONs with synthetic
   Thread/Text labels.
3. The 4,834 original ABC JSONs approved by the existing inference filter as
   having no face with Thread or Text confidence above `0.80`. Every face in
   these approved originals is pseudo-labeled `Stock=0`.

The result is one combined `no_a2` dataset with new STEP-family-aware
`train.txt`, `val.txt`, and `test.txt`, new class weights, and validated A1/A3
tensors. The existing 72K dataset is not modified.

### Why the approved originals are included

The synthetic 25K folder teaches the model what inserted Thread and Text look
like on ABC geometry. The untouched approved originals teach it that realistic
ABC geometry without an inserted feature should remain Stock.

These are supervised pseudo-labels. The model is **not** asked to invent labels
during training. The inference filter is the source of the part-level
assumption, and every face in an approved original receives class `0`.

### Important limitation

Only JSON paths listed in:

```text
C:\jsons\inference\no_confident_thread_or_text.txt
```

may be converted to Stock-only samples. Do not label every JSON under
`C:\jsons` as Stock. That root contains both accepted and rejected parts.

At the time this workflow was created:

- `C:\jsons` contained 10,251 root JSON files.
- The approved list contained 4,834 JSON paths.
- The first 100 approved JSONs contained 7,878 faces, all with raw label `-1`.
- Raw `-1` maps to model class `0` (Stock).

## 2. Scripts used by this workflow

| Script | Purpose |
|---|---|
| `scripts/threads/prepare_approved_abc_stock_jsons.py` | Audit the approved path list and create separate JSON copies with every face labeled `0`. It never modifies `C:\jsons`. |
| `scripts/threads/prepare_new_abc_finetune_data.ps1` | Operator wrapper for Stock-copy creation and combined no_a2 preparation. |
| `scripts/threads/prepare_no_a2_scratch_delta.ps1` | Existing underlying pipeline: seed old graphs, remap new labels, convert missing JSONs, rebuild splits, calculate weights, and validate. |
| `scripts/inference/json_to_brepmfr_pyg_optimized.py` | Convert new JSONs to PyG using the `no_a2` profile. Called by the underlying preparer. |
| `scripts/threads/make_random_splits.py` | Generate STEP-key-aware splits. Called by the underlying preparer. |
| `scripts/training/compute_class_weights.py` | Calculate weights from the new train split. Called by the underlying preparer. |
| `scripts/threads/validate_a1_a3_finetune_data.py` | Validate split coverage, labels, A1, and A3 tensors. Called by the underlying preparer. |
| `scripts/threads/train_new_abc_finetune.ps1` | Start a conservative fine-tuning run from an existing no_a2 checkpoint. |
| `scripts/threads/make_stock_only_eval_split.py` | Select untouched Stock-only originals that landed in validation or test. |
| `scripts/threads/run_thread_pyg_inference.py` | Run checkpoint inference and write per-part prediction CSVs and aggregate metrics. |
| `scripts/threads/summarize_stock_only_inference.py` | Report Stock→Text face rate and percentage of Stock-only parts with any false Text prediction. |

## 3. Choose working paths

The examples below use these placeholders. Change them to the actual machine
paths before running:

```powershell
$OldNoA2       = "D:\thread_and_text\no_a2_72k"
$New25KJson    = "D:\thread_and_text\new_abc_25k_json_working"
$StockJson     = "D:\thread_and_text\approved_abc_stock_json"
$CombinedNoA2  = "D:\thread_and_text\no_a2_72k_plus_new_abc"
$ApprovedList  = "C:\jsons\inference\no_confident_thread_or_text.txt"
$ClassWeights  = "artifacts\class_weights\thread_text\new_abc_finetune_alpha05.json"
$IdentityMap   = "scripts\threads\remap_maps\thread_text_sw_to_brep_with_identity.json"
```

Expected directory layout before preparation:

```text
D:\thread_and_text\no_a2_72k\
  pyg\
  train.txt
  val.txt
  test.txt

D:\thread_and_text\new_abc_25k_json_working\
  *.json

C:\jsons\inference\
  no_confident_thread_or_text.txt
```

`$New25KJson` should be a working copy or a folder that may be normalized in
place. The preparation pipeline remaps raw face labels there:

```text
-10, -1, 0 -> 0
70          -> 1
101         -> 2
already-normalized 0, 1, 2 remain unchanged
```

The identity-capable map is used because the supplied 25K folder may contain
either SolidWorks raw labels (`70`, `101`) or already normalized labels
(`1`, `2`).

## 4. Preflight requirements

Run from the repository root:

```powershell
cd C:\Users\RZA2\Desktop\BrepMFR_PyG\BrepMFR_PyG
```

Confirm the conda environment:

```powershell
conda run --no-capture-output -n brep_mfr_pyg python -c "import torch, torch_geometric; print(torch.__version__)"
```

Confirm the major inputs:

```powershell
Test-Path "$OldNoA2\pyg"
Test-Path $New25KJson
Test-Path $ApprovedList

(Get-ChildItem "$OldNoA2\pyg" -Filter "*.pt" -File).Count
(Get-ChildItem $New25KJson -Filter "*.json" -File).Count
(Get-Content $ApprovedList | Where-Object { $_.Trim() }).Count
```

The expected results are approximately 72K old `.pt` graphs, 25K new JSONs,
and 4,834 approved paths. Exact counts may differ after invalid graphs or
duplicate stems are removed.

### Filename/leakage requirement

Generated variants should preserve the base identifier through
`..._step_NNN...`, for example:

```text
00000014_<id>_step_000_101
00000014_<id>_step_000_both_v4_105
```

`make_random_splits.py` groups everything through `_step_NNN` into one atomic
family. The untouched original and every generated version of that STEP
therefore remain in the same split. If the new generator discarded this
identifier, fix the names before splitting; otherwise related geometry can
leak across train and test.

## 5. Dry-run the complete input audit

Do this before writing anything:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\threads\prepare_new_abc_finetune_data.ps1 `
  -OldNoA2Root $OldNoA2 `
  -NewLabeledJsonDir $New25KJson `
  -ApprovedList $ApprovedList `
  -StockJsonDir $StockJson `
  -CombinedNoA2Root $CombinedNoA2 `
  -MapJson $IdentityMap `
  -ClassWeightsOut $ClassWeights
```

The dry run performs two strict checks:

1. Every approved path exists, contains a non-empty top-level `faces` list,
   gives every face an integer label, and uses only `-10`, `-1`, or `0`.
   Labels `70` and `101` are rejected for Stock-only originals.
2. The new 25K folder contains only labels understood by the identity-capable
   remap.

No source JSON, destination JSON, PyG graph, or split list is written in this
mode.

### Stock exporter arguments

The underlying standalone dry run can also be invoked directly:

```powershell
conda run --no-capture-output -n brep_mfr_pyg python `
  scripts\threads\prepare_approved_abc_stock_jsons.py `
  --approved-list $ApprovedList `
  --output-dir $StockJson `
  --expected-source-labels=-10,-1,0 `
  --stock-label 0 `
  --workers 8
```

Relevant arguments:

| Argument | Meaning |
|---|---|
| `--approved-list` | One approved source JSON path per line. |
| `--output-dir` | Separate destination. Source files are never edited. |
| `--expected-source-labels` | Strictly permitted raw labels. Do not add `70` or `101`. |
| `--stock-label` | Output class, fixed to `0` for this project. |
| `--workers` | Parallel JSON audit/write workers. Reduce if host RAM is constrained. |
| `--write` | Apply after the complete audit succeeds. Omit for dry run. |
| `--overwrite` | Deliberately replace existing destination JSONs. |

## 6. Build the combined no_a2 dataset

After the dry run passes:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\threads\prepare_new_abc_finetune_data.ps1 `
  -OldNoA2Root $OldNoA2 `
  -NewLabeledJsonDir $New25KJson `
  -ApprovedList $ApprovedList `
  -StockJsonDir $StockJson `
  -CombinedNoA2Root $CombinedNoA2 `
  -MapJson $IdentityMap `
  -ClassWeightsOut $ClassWeights `
  -Workers 8 `
  -SeedMode HardLink `
  -MinFreeGB 20 `
  -Apply
```

This performs the following sequence:

1. Re-audits the complete approved list.
2. Writes Stock-only JSON copies under `$StockJson`.
3. Writes `$StockJson\stock_label_manifest.csv`, linking every source path to
   its output JSON and face count.
4. Creates `$CombinedNoA2\pyg`.
5. Hard-links existing 72K `.pt` graphs into the combined directory. Old graph
   bytes are not duplicated and the old directory is not modified.
6. Audits and remaps only new JSONs that do not already have matching graphs.
7. Converts the 25K labeled JSONs and 4.8K Stock JSONs with
   `--inference_profile no_a2`, spatial cutoff 32, and A3 path length 16.
8. Checks that every input JSON stem has a corresponding `.pt`.
9. Regenerates 80/10/10 train/validation/test lists over the complete corpus.
10. Keeps `_step_NNN` families atomic.
11. Ensures at least 80% of approved Stock JSON stems, and their related STEP
    variants, are assigned to training.
12. Recomputes class weights from the new training split.
13. Validates all split-listed graphs, A1/A3 flags/tensors, and label range
    `[0,2]`.
14. Quarantines unusable graphs and removes their stems from the split lists.

### HardLink versus Copy

Use `-SeedMode HardLink` when the old and combined directories are on the same
NTFS volume. This is fast and avoids duplicating the dense no_a2 corpus.

Use:

```powershell
-SeedMode Copy
```

when the two roots are on different volumes. Verify enough disk space first.

### Existing Stock outputs

The Stock exporter refuses to silently replace files. When rerunning the exact
same approved export deliberately:

```powershell
-OverwriteStockJsons
```

### Recovering a partial combined output

If a failed attempt left `$CombinedNoA2` incomplete, rerun with:

```powershell
-ResetCombinedOutput
```

This ultimately invokes the existing preparer's `-ResetOutput`. It deletes only
the explicitly supplied combined output directory. It refuses to run if the
combined and old roots are identical. Do not use this switch for an ordinary
resume when preparation already reached validation.

If conversion, splitting, and weight calculation completed but final
validation reported invalid graphs, run the validator in place instead of
starting over:

```powershell
conda run --no-capture-output -n brep_mfr_pyg python `
  scripts\threads\validate_a1_a3_finetune_data.py `
  --dataset-root $CombinedNoA2 `
  --report-a3-cap 768 `
  --quarantine-invalid
```

## 7. Verify the training-ready dataset

Check artifact counts:

```powershell
(Get-ChildItem "$CombinedNoA2\pyg" -Filter "*.pt" -File).Count
(Get-Content "$CombinedNoA2\train.txt" | Where-Object { $_.Trim() }).Count
(Get-Content "$CombinedNoA2\val.txt"   | Where-Object { $_.Trim() }).Count
(Get-Content "$CombinedNoA2\test.txt"  | Where-Object { $_.Trim() }).Count
Get-Content $ClassWeights
```

Recount the final training labels:

```powershell
conda run --no-capture-output -n brep_mfr_pyg python `
  scripts\threads\count_thread_label_distribution.py `
  --pyg-dir "$CombinedNoA2\pyg" `
  --group "0:stock,1:thread,2:text"
```

Run an optional 100-graph validation smoke:

```powershell
conda run --no-capture-output -n brep_mfr_pyg python `
  scripts\threads\validate_a1_a3_finetune_data.py `
  --dataset-root $CombinedNoA2 `
  --max-files 100 `
  --report-a3-cap 768
```

The final full preparation already performs a complete validation. The smoke
command is mainly useful when inspecting a transferred dataset later.

## 8. Create the Stock-only evaluation split

This evaluation is the direct proxy for the reported failure. It contains only
untouched approved ABC originals that landed in the combined test split:

```powershell
conda run --no-capture-output -n brep_mfr_pyg python `
  scripts\threads\make_stock_only_eval_split.py `
  --stock-manifest "$StockJson\stock_label_manifest.csv" `
  --split-file "$CombinedNoA2\test.txt" `
  --out "$CombinedNoA2\stock_only_test.txt"
```

Also create a validation subset if desired:

```powershell
conda run --no-capture-output -n brep_mfr_pyg python `
  scripts\threads\make_stock_only_eval_split.py `
  --stock-manifest "$StockJson\stock_label_manifest.csv" `
  --split-file "$CombinedNoA2\val.txt" `
  --out "$CombinedNoA2\stock_only_val.txt"
```

Do not move test Stock parts into training after examining their results. Keep
this test list fixed across all fine-tuning comparisons.

## 9. Choose the checkpoint

Use the best checkpoint from the 72K model trained natively with `no_a2`, not
the older lite-to-no_a2 recovery checkpoint, unless a direct comparison shows
the latter is clearly better on the fixed Stock-only test set.

Example:

```powershell
$Checkpoint = "C:\path\to\72k_no_a2\best.ckpt"
Test-Path $Checkpoint
```

The new run must use `--pre_train` semantics:

- Load the learned model weights.
- Start epoch 0 with a fresh optimizer.
- Use the new dataset and new schedule.

Do not use exact resume for the first run. `--resume_from_checkpoint` restores
the old optimizer, global step, and loop state and is intended only for
continuing an interrupted run on the same new combined dataset.

## 10. Run a short training smoke

Use a unique run name because even a limited smoke advances epochs and writes
checkpoints:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\threads\train_new_abc_finetune.ps1 `
  -Checkpoint $Checkpoint `
  -DatasetRoot $CombinedNoA2 `
  -RunName "thread_text_new_abc_smoke" `
  -MaxEpochs 1 `
  -LimitTrainBatches 20 `
  -DataLoaderWorkers 2
```

Confirm:

- The checkpoint loads without missing architecture errors.
- Batches contain labels only in `[0,2]`.
- A1/A3 are active.
- Training and validation complete.
- No CUDA or host-memory failure occurs.

Delete or ignore the smoke run afterward; do not resume it as the real run.

## 11. Start the recommended fine-tuning run

First controlled run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\threads\train_new_abc_finetune.ps1 `
  -Checkpoint $Checkpoint `
  -DatasetRoot $CombinedNoA2 `
  -ClassWeights $ClassWeights `
  -UseClassWeights $false `
  -RunName "thread_text_new_abc_finetune_v1" `
  -MaxEpochs 15 `
  -LearningRate 0.0001 `
  -A1A3LearningRate 0.0001 `
  -OptimizerWarmupSteps 500 `
  -MaxNodesForA3 768 `
  -BatchSize 64 `
  -BatchNodeSqBudget 4000000 `
  -DataLoaderWorkers 4
```

The launcher uses:

- `num_classes=3`;
- mixed precision;
- full A1/A3 scale from epoch 0;
- no lite-to-A1/A3 ramp;
- no encoder freeze;
- cross-entropy;
- validation every epoch;
- STEP-aware combined splits;
- CSV and TensorBoard logs;
- adaptive graph-size batching; and
- a 768-face A3 cap while retaining A1 above the cap.

### Why class weights are disabled in run 1

The specific problem is Stock being predicted as Text. Recomputed
inverse-frequency weights reduce the Stock weight as additional Stock faces are
added, partly cancelling the desired signal. The first run therefore uses
unweighted cross-entropy and allows the new Stock-only faces to contribute
normally.

This is a controlled decision, not a permanent claim that weighting is bad.
After run 1, launch an otherwise identical comparison with:

```powershell
-UseClassWeights $true -ClassWeights $ClassWeights
```

Do not change learning rate, split, checkpoint, and weighting simultaneously.

### Why no new loss is enabled in run 1

The repository's existing cross-entropy already penalizes a true Stock face
when its Text logit rises. The new data changes the missing supervision: it
adds realistic, untouched ABC parts whose every face is Stock. Establish this
data-only fine-tuning baseline before adding an asymmetric Stock→Text loss.

If the fixed Stock-only test still shows unacceptable false Text predictions
while true Text recall remains strong, the next experiment should add a
targeted Stock→Text hard-negative term. It should be introduced as one isolated
change and compared on the same splits. Do not add it before measuring the
baseline because it can trade Text recall for Stock precision.

## 12. Monitor training

Run TensorBoard:

```powershell
tensorboard --logdir results\logs\stage1\thread_text_new_abc_finetune_v1
```

Watch:

- `eval_loss`;
- `per_face_accuracy`;
- `per_class_accuracy`;
- `val_class_0_acc` for Stock recall;
- `val_class_1_acc` for Thread recall;
- `val_class_2_acc` for Text recall;
- validation confusion matrix images; and
- learning-rate curves.

The confusion-matrix cell that directly represents the failure is:

```text
true Stock row, predicted Text column
```

Overall per-face accuracy is secondary. A large part can have excellent face
accuracy and still fail because one Stock face is classified as Text.

Checkpoints are written under:

```text
results\stage1\thread_text_new_abc_finetune_v1\
  best.ckpt
  best-v*.ckpt
  last.ckpt
```

The current checkpoint callback retains multiple best candidates by validation
loss. Evaluate the relevant candidates on the fixed Stock-only set instead of
assuming the lowest aggregate loss is operationally best.

## 13. Resume an interrupted fine-tuning run

Only for an interrupted run using the same dataset and run name:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\threads\train_new_abc_finetune.ps1 `
  -ResumeFromCheckpoint "results\stage1\thread_text_new_abc_finetune_v1\last.ckpt" `
  -DatasetRoot $CombinedNoA2 `
  -RunName "thread_text_new_abc_finetune_v1" `
  -MaxEpochs 15 `
  -LearningRate 0.0001 `
  -A1A3LearningRate 0.0001
```

Use `-Checkpoint` for a new experiment and `-ResumeFromCheckpoint` for an exact
continuation. Never provide both.

## 14. Evaluate Stock→Text false positives

After training:

```powershell
$FineTunedCheckpoint = "results\stage1\thread_text_new_abc_finetune_v1\best.ckpt"
$StockInference = "$CombinedNoA2\inference_stock_only_test_v1"
$StockMetrics = "$CombinedNoA2\metrics_stock_only_test_v1"
```

Run inference on exact Stock-only test stems:

```powershell
conda run --no-capture-output -n brep_mfr_pyg python `
  scripts\threads\run_thread_pyg_inference.py `
  --checkpoint $FineTunedCheckpoint `
  --dataset_path $CombinedNoA2 `
  --pyg_dir "$CombinedNoA2\pyg" `
  --split_file "$CombinedNoA2\stock_only_test.txt" `
  --inference_dir $StockInference `
  --metrics_dir $StockMetrics `
  --batch_size 1 `
  --max_nodes_for_a3 768
```

Summarize the operational false positives:

```powershell
conda run --no-capture-output -n brep_mfr_pyg python `
  scripts\threads\summarize_stock_only_inference.py `
  --inference-dir $StockInference
```

This reports:

- Stock→Thread face rate;
- Stock→Text face rate;
- percentage of Stock-only parts with any false Thread;
- percentage of Stock-only parts with any false Text; and
- percentage of Stock-only parts with any false feature.

Run the same commands with the original 72K checkpoint and a separate output
directory. The fine-tuned model should reduce both Stock→Text face rate and,
more importantly, the fraction of Stock-only parts with any Text prediction.

## 15. Evaluate the full held-out test split

Do not accept a Stock-only improvement that destroys Thread/Text detection:

```powershell
conda run --no-capture-output -n brep_mfr_pyg python segmentation.py test `
  --dataset_path $CombinedNoA2 `
  --pt_subdir pyg `
  --num_classes 3 `
  --drop_invalid_graphs `
  --batch_size 4 `
  --num_workers 0 `
  --max_nodes_for_a3 768 `
  --checkpoint $FineTunedCheckpoint
```

Compare:

- Stock precision and recall;
- Thread precision and recall;
- Text precision and recall;
- full confusion matrix;
- macro precision/recall;
- Stock-only part false-Text rate; and
- results on any separate GrabCAD inference set.

## 16. Model-selection rule

Prefer the checkpoint that:

1. materially reduces the percentage of Stock-only parts with any false Text;
2. reduces the Stock→Text confusion-matrix cell;
3. retains acceptable synthetic Text recall;
4. retains the already strong Thread precision and recall; and
5. does not regress badly on the full combined test split.

Do not select solely by train accuracy, aggregate validation accuracy, or
validation loss.

## 17. Common failure cases

### Approved source contains label 70 or 101

The Stock exporter stops. Do not expand `--expected-source-labels` to make the
error disappear. That part is not safe to assign entirely to Stock. Regenerate
the approved list or investigate why the wrong path entered it.

### New 25K folder contains an unknown label

The dry-run remap stops and prints the unknown counts. Update the remap only
after confirming the SolidWorks meaning. Do not use `--allow-unmapped`.

### Output filenames collide

The Stock exporter stops before writing. Two different approved sources would
otherwise overwrite one another. Resolve naming at the source or create a
deliberate unique naming convention that preserves `_step_NNN`.

### Hard links fail

The old and combined roots are probably on different volumes or the filesystem
does not support hard links. Use `-SeedMode Copy` after checking free space.

### Disk fills during conversion

The preparer refuses conversion below `-MinFreeGB`. Use hard links, free disk
space, or move the combined output. Do not lower the safety threshold without
checking the expected no_a2 size.

### A3 memory failure

Lower:

```powershell
-MaxNodesForA3 512
```

A1 remains active above the cap. Do not set the cap to zero unless host and GPU
memory can support dense A3 for the largest graph.

### Fine-tuning immediately worsens Text recall

Stop the run, verify the new label histogram, check that synthetic labels were
normalized correctly, and confirm that the train split contains Thread/Text
examples. Then compare class-weighted versus unweighted CE without changing
other settings.

## 18. Minimal command sequence

After setting the path variables, the essential sequence is:

```powershell
# 1. Strict no-write audit
powershell -ExecutionPolicy Bypass -File scripts\threads\prepare_new_abc_finetune_data.ps1 `
  -OldNoA2Root $OldNoA2 -NewLabeledJsonDir $New25KJson `
  -ApprovedList $ApprovedList -StockJsonDir $StockJson `
  -CombinedNoA2Root $CombinedNoA2 -MapJson $IdentityMap `
  -ClassWeightsOut $ClassWeights

# 2. Build Stock copies + combined no_a2 dataset + splits + weights + validation
powershell -ExecutionPolicy Bypass -File scripts\threads\prepare_new_abc_finetune_data.ps1 `
  -OldNoA2Root $OldNoA2 -NewLabeledJsonDir $New25KJson `
  -ApprovedList $ApprovedList -StockJsonDir $StockJson `
  -CombinedNoA2Root $CombinedNoA2 -MapJson $IdentityMap `
  -ClassWeightsOut $ClassWeights -Apply

# 3. Freeze the Stock-only test subset
conda run --no-capture-output -n brep_mfr_pyg python `
  scripts\threads\make_stock_only_eval_split.py `
  --stock-manifest "$StockJson\stock_label_manifest.csv" `
  --split-file "$CombinedNoA2\test.txt" `
  --out "$CombinedNoA2\stock_only_test.txt"

# 4. Smoke training
powershell -ExecutionPolicy Bypass -File scripts\threads\train_new_abc_finetune.ps1 `
  -Checkpoint $Checkpoint -DatasetRoot $CombinedNoA2 `
  -RunName "thread_text_new_abc_smoke" -MaxEpochs 1 `
  -LimitTrainBatches 20 -DataLoaderWorkers 2

# 5. Full fine-tuning
powershell -ExecutionPolicy Bypass -File scripts\threads\train_new_abc_finetune.ps1 `
  -Checkpoint $Checkpoint -DatasetRoot $CombinedNoA2 `
  -ClassWeights $ClassWeights -UseClassWeights $false `
  -RunName "thread_text_new_abc_finetune_v1" -MaxEpochs 15
```

