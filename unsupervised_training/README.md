# Unlabeled ABC masked-geometry experiment

## Objective

Use approximately 20K additional raw ABC parts without asserting that any face
is Stock, Thread, or Text. The experiment starts from the frozen Model A
champion and must preserve its supervised behavior while learning more general
ABC geometry.

This folder does not modify `segmentation.py`, `data/dataset.py`, or the existing
Stage-1/ONNX workflow.

## Are label-free SolidWorks JSONs sufficient?

Yes, provided they contain exactly the same geometric/topological fields as an
inference JSON:

- non-empty `faces` and an `edges` list;
- every face has `id`, `uv`, `z`, `y`, `l`, and `a`;
- every edge has `nf` and `pt`;
- optional A2 `face_pairs` are not needed because this experiment uses the
  `no_a2` profile.

The JSONs do not need a `label` field. If a label happens to be present it is
ignored. `prepare_unlabeled.py` writes a new PyG graph with every
`label_feature=-100`. That value is an ignore/safety sentinel, not Stock. The
trainer aborts if an unlabeled batch contains anything else.

Do **not** run the old Model-A inference filter before this experiment. That
filter removes the unfamiliar geometry we want masked modeling to learn. The
preparer excludes exact stems and STEP families already present in the labeled
train/validation/test lists instead.

## Method

Each optimizer step contains one labeled and one unlabeled batch.

1. **Supervised three-class CE:** unchanged labeled CADSynth + ABC data, using
   Model A's mild embedded class weights.
2. **Source behavioral anchor:** soft-logit distillation from frozen Model A on
   labeled faces.
3. **Masked geometry:** 15% of each unlabeled part's faces lose their local UV,
   area, type, loop, and degree inputs. The student reconstructs continuous
   surface descriptors and intrinsic categorical attributes using neighboring
   faces, edges, A1, and A3.
4. **Unlabeled behavioral anchor:** on unmasked faces, the student stays near
   Model A's soft logits and fused embeddings. These are not hard pseudo-labels
   and the weight is deliberately small.
5. **Frozen BatchNorm statistics:** prevents the domain-statistics failure seen
   in Model D. Distillation addresses the encoder/head drift that remained in
   Model E.

The unlabeled losses ramp to full strength over the first three epochs. Only the
student encoder/classifier is exported; the teacher and reconstruction head are
training-only.

## Folder contents

| File | Purpose |
|---|---|
| `prepare_unlabeled.py` | Strict JSON audit, overlap exclusion, no-A2 conversion, sentinel labels, family-safe train/val split |
| `audit_prepared.py` | Reload every generated graph and prove sentinel/profile invariants |
| `train.py` | Joint labeled + unlabeled Lightning training |
| `semi_model.py` | Masked reconstruction, fixed-teacher distillation, validation regression accounting |
| `export_for_solidworks.py` | Extract standard Stage-1 student checkpoint and invoke the existing A1+A3 ONNX exporter |
| `evaluate_against_champion.py` | Per-face/per-part candidate versus Model A report and SolidWorks review queue |
| `configs/abc_masked_geometry_v1.json` | Versioned conservative experiment configuration |
| `tests/test_core.py` | Unit tests for masking, targets, distillation, and label-free JSON schema |

## Stage 0: fixed inputs

Recommended paths on the training machine:

```powershell
$Repo = "C:\Users\RZA2\thread_project\BrepMFR"
$Labeled = "D:\thread_and_text\abc_for_modelA_finetuning"
$RawUnlabeled = "D:\thread_and_text\unlabeled_abc_json_20k"
$Unlabeled = "D:\thread_and_text\unlabeled_abc_no_a2"
$Champion = "$Repo\model_checkpoints\abc_with_no_a2\last-v1.ckpt"
Set-Location $Repo
```

Keep the existing labeled `train.txt`, `val.txt`, and `test.txt` unchanged.

## Stage 1: conversion smoke test

First convert only 100 raw JSONs into a disposable output root:

```powershell
python -m unsupervised_training.prepare_unlabeled `
  --json-dir $RawUnlabeled `
  --output-root "D:\thread_and_text\unlabeled_abc_no_a2_smoke" `
  --labeled-dataset-root $Labeled `
  --workers 8 `
  --limit 100

python -m unsupervised_training.audit_prepared `
  --dataset-root "D:\thread_and_text\unlabeled_abc_no_a2_smoke"
```

Expected result: non-zero train and validation counts, profile `no_a2`, and
`sentinel_only=true`. Nothing under the raw JSON folder is modified.

## Stage 2: full unlabeled preparation

```powershell
python -m unsupervised_training.prepare_unlabeled `
  --json-dir $RawUnlabeled `
  --output-root $Unlabeled `
  --labeled-dataset-root $Labeled `
  --workers 8

python -m unsupervised_training.audit_prepared `
  --dataset-root $Unlabeled
```

If conversion is interrupted, rerun the same command. Existing output graphs
are skipped. Use `--overwrite` only when the source JSONs or conversion code
actually changed.

## Stage 3: one-batch integration smoke

This loads the real champion, labeled graphs, unlabeled graphs, all four forward
passes, validation, checkpointing, and mixed precision but limits execution to
two train and two validation batches:

```powershell
python -m unsupervised_training.train `
  --config unsupervised_training\configs\abc_masked_geometry_v1.json `
  --champion-checkpoint $Champion `
  --labeled-dataset-root $Labeled `
  --unlabeled-dataset-root $Unlabeled `
  --smoke
```

Do not start the full run unless every loss is finite and
`val/regression_rate`, `val/macro_iou`, and `val/guarded_score` are logged.

## Stage 4: full experiment U1

```powershell
python -m unsupervised_training.train `
  --config unsupervised_training\configs\abc_masked_geometry_v1.json `
  --champion-checkpoint $Champion `
  --labeled-dataset-root $Labeled `
  --unlabeled-dataset-root $Unlabeled `
  --run-name abc_masked_geometry_model_a_v1
```

The default is 12 epochs. This is intentionally a short adaptation run, not a
new 100-epoch search. Results are written under:

```text
results/unsupervised/abc_masked_geometry_model_a_v1/
results/logs/unsupervised/abc_masked_geometry_model_a_v1/
```

Exact resume after interruption:

```powershell
python -m unsupervised_training.train `
  --config unsupervised_training\configs\abc_masked_geometry_v1.json `
  --champion-checkpoint $Champion `
  --labeled-dataset-root $Labeled `
  --unlabeled-dataset-root $Unlabeled `
  --run-name abc_masked_geometry_model_a_v1 `
  --resume-from-checkpoint "$Repo\results\unsupervised\abc_masked_geometry_model_a_v1\last.ckpt"
```

## Stage 5: export each serious candidate

Do not assume the final epoch is best. Export the top guarded checkpoints and
compare them separately.

```powershell
$Joint = "$Repo\results\unsupervised\abc_masked_geometry_model_a_v1\candidate-epochXX-stepYYYY.ckpt"
$Export = "$Repo\artifacts\unsupervised_training\candidate_epochXX"

python -m unsupervised_training.export_for_solidworks `
  --joint-checkpoint $Joint `
  --champion-checkpoint $Champion `
  --output-dir $Export
```

Expected files include `student_stage1.ckpt`, `brepmfr_no_a2.onnx`,
`model_config.json`, and `label_map.json`. The existing exporter performs
PyTorch/ONNX numerical and class-decision parity checks.

## Stage 6: automated tests before SolidWorks UI

Run candidate-versus-champion evaluation on the unchanged labeled validation
and test splits:

```powershell
python -m unsupervised_training.evaluate_against_champion `
  --champion-checkpoint $Champion `
  --candidate-checkpoint "$Export\student_stage1.ckpt" `
  --dataset-root $Labeled `
  --split val `
  --output-dir "$Export\eval_val"

python -m unsupervised_training.evaluate_against_champion `
  --champion-checkpoint $Champion `
  --candidate-checkpoint "$Export\student_stage1.ckpt" `
  --dataset-root $Labeled `
  --split test `
  --output-dir "$Export\eval_test"
```

Also run it on any retained Stock-only split by passing:

```powershell
--split-file "D:\thread_and_text\abc_for_modelA_finetuning\stock_only_test.txt"
```

Artifacts:

- `summary.json`: metrics and aggregate regression/improvement counts;
- `parts.csv`: prediction-count changes for every part;
- `face_disagreements.csv`: exact changed face IDs and both confidences;
- `solidworks_ui_review_queue.csv`: parts ranked for UI inspection.

## Promotion gates before manual UI testing

A candidate reaches SolidWorks only if all of these pass:

1. PyTorch and ONNX have 100% class-decision parity in exporter validation.
2. No class recall on the fixed labeled test falls by more than 0.5 percentage
   points versus Model A.
3. Macro IoU falls by no more than 0.2 percentage points.
4. Stock-only parts with any predicted Thread/Text do not increase.
5. There is no systemic Thread-removal pattern like Models D/E.
6. There is no systemic Thread-expansion pattern detectable in the disagreement
   queue.

These are pre-UI gates, not proof of real-world improvement.

## Fixed SolidWorks UI review

Use the same saved collection for Model A and every candidate:

- all previously observed Model A failure parts;
- Klavuz;
- confirmed Stock-only GrabCAD parts;
- native Thread parts of multiple constructions;
- engraved/embossed Text parts;
- large parts above 768 faces;
- the top disagreement parts from `solidworks_ui_review_queue.csv`.

For every reviewed component record: part, face indices, Model A class and
confidence, candidate class and confidence, visual verdict, and notes. Keep the
review file outside train/validation data. A candidate replaces Model A only if
it fixes meaningful failures without introducing a critical complete feature
miss.

## Planned experiment sequence

Only U1 should run initially:

| ID | Change | When to run |
|---|---|---|
| U0 | Frozen Model A | Existing benchmark; no training |
| U1 | Default masked geometry + conservative distillation | Run now |
| U2 | Reduce unlabeled distillation from 0.20 to 0.10 | Only if U1 exactly copies Model A and learns useful reconstruction |
| U3 | Increase source distillation from 0.50 to 1.00 | Only if U1 regresses on labeled/UI gates |
| U4 | Circular edge-angle sine/cosine architecture change | Separate future experiment after U1; requires rebuilt compatibility path |

Do not change loss weights, source data, architecture, and initialization in the
same run. If U1 fails, its disagreement report should determine whether U2 or U3
is justified; otherwise stop rather than launching an open-ended sweep.

