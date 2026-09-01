# Thread/Text Training Experiment History

Last updated: 2026-07-27

## 1. Purpose and scope

This document consolidates the Thread/Text Stage-1 training experiments discussed
throughout the ABC-data investigation. It records:

- the data used by each run;
- whether the model started from random weights or another checkpoint;
- the inference/proximity profile;
- loss and class weighting;
- learning rates, warmup, epochs, and saved checkpoint state;
- in-distribution test results when available;
- ONNX Klavuz results;
- the failure mode and conclusion from each experiment.

The main lineage is named consistently here as **Model A through Model F**. Some
earlier conversation turns used model letters inconsistently, so checkpoint paths
and run names are the authoritative identifiers.

This document does not attempt to catalogue unrelated MFCAD/Stage-2 experiments
under `results/`. It covers the Stock/Thread/Text campaign based on CADSynth, ABC,
the 72K no-a2 expansion, unique ABC replay, frozen BatchNorm, and full A1/A3
training from scratch.

## 2. Task definition and common model contract

### Classes

| Class ID | Class |
|---:|---|
| 0 | Stock |
| 1 | Thread |
| 2 | Text |

### Geometry profiles

| Profile | Inputs |
|---|---|
| `lite` | Face/edge geometry without A1, A2, or A3 proximity tensors |
| `no_a2` / A1+A3 | A1 shortest-path spatial proximity plus A3 edge-path proximity; A2 distance/angle histograms excluded |

For all later A1+A3 runs, `max_nodes_for_a3=768`. Graphs above 768 faces retain
A1 but skip dense A3 during training. ONNX export currently forces full A3.

### Common architecture for the A–F comparisons

| Parameter | Value |
|---|---:|
| Classes | 3 |
| `dim_node` | 256 |
| `d_model` | 512 |
| Heads | 32 |
| Encoder layers | 8 |
| Dropout | 0.3 |
| Attention dropout | 0.3 |
| Activation dropout | 0.3 |
| Precision | 16-bit mixed precision |

## 3. How labels and ABC data were created

No native face labels were available for Stock, Thread, or Text.

### CADSynth

1. Begin with synthetic CADSynth parts.
2. Use SolidWorks to insert:
   - engraved/debossed Text;
   - cylindrical/drilled Thread constructions.
3. Export JSON labels for inserted feature faces.
4. Treat every remaining face as Stock.

### ABC

Native ABC parts may already contain real text or threads. Labeling all original
faces as Stock would therefore teach contradictory labels.

The filtering procedure was:

1. Train a sufficiently good CADSynth model.
2. Run it on candidate ABC parts.
3. Approve only parts for which no face is predicted as Thread or Text above the
   chosen confidence threshold (approximately 0.80).
4. Treat approved original faces as Stock.
5. Insert synthetic Text and Thread features with SolidWorks and create labels.

The approved Stock-only preparation produced:

- 5,128 JSON parts;
- 651,953 faces;
- all source labels audited as class 0;
- original source JSONs left unchanged.

This process reduces label noise but cannot guarantee that every approved ABC part
is truly free of native Thread/Text.

## 4. Canonical datasets

### 4.1 Model A small/replay dataset

Paths:

```text
Z:\thread_and_text\lite
Z:\thread_and_text\no_a2
```

Composition reported during the investigation:

- approximately 30K CADSynth;
- approximately 10K ABC;
- retained graph store: 48,455 graphs.

The original narrative often called this 40K or 45K. The prepared graph store and
split audit report 48,455, so 48,455 is used when discussing the actual retained
dataset.

Observed no-a2 split loading:

| Split | Listed/loaded | Valid after scan |
|---|---:|---:|
| Train | 38,766 | 38,745 |
| Validation | 4,851 | 4,851 |
| Test/remainder | approximately 4,838 | not separately re-audited in the retained log |

### 4.2 Model B 72K no-a2 dataset

Path:

```text
D:\thread_and_text\no_a2_large
```

Composition:

- Model A-era samples;
- approximately 20K older CADSynth parts not present in the Model A set;
- the extra set was described as Text and Text+Thread generation, without a
  dedicated Thread-only subset.

Final validated graph count:

```text
72,197
```

Splits:

| Split | Graphs |
|---|---:|
| Train | 57,760 |
| Validation | 7,227 |
| Test | 7,210 |
| Total | 72,197 |

### 4.3 Model C expanded 72K plus new ABC dataset

Path:

```text
D:\thread_and_text\no_a2_72k_plus_new_abc
```

Inputs:

- protected 72,197-graph no-a2 base;
- `new_abc_json_25k`, which contained 32,040 JSON files after invalid JSONs were
  removed;
- 18,432 labeled JSON stems not already present in the 72K graph store;
- 5,128 approved Stock JSONs;
- 19 filenames overlapped between the labeled and Stock JSON sources.

Final graph count and splits:

| Split | Graphs |
|---|---:|
| Train | 76,452 |
| Validation | 9,664 |
| Test | 9,622 |
| Total | 95,738 |

The original discussion referred to “20K new ABC + 10K previous ABC” or “about
30K ABC.” The exact expanded graph count is more reliable than those rounded
descriptions.

### 4.4 Model A replay plus unique ABC dataset

Logical paths used on different machines:

```text
Z:\thread_and_text\abc_for_modelA_finetuning
D:\thread_and_text\abc_for_modelA_finetuning
```

Preparation goals:

- preserve Model A split membership;
- append only full-stem-unique new ABC graphs;
- prevent exact-stem and STEP-family leakage;
- exclude the strict Stock-only evaluation set from train/validation/test.

Reported preparation counts:

| Item | Count |
|---|---:|
| Model A replay candidates | 48,455 |
| Unique new ABC candidates | 18,434 |
| Final train/val/test graphs | 66,864 |
| Strict Stock-only holdout | 407 |
| Total `.pt` files scanned | 67,271 |

Final splits:

| Split | Graphs |
|---|---:|
| Train | 53,779 |
| Validation | 6,841 |
| Test | 6,244 |
| Train+validation+test | 66,864 |
| Strict Stock-only holdout | 407 |

The candidate counts sum to 66,889, while the final leakage-safe split contains
66,864. The retained preparation summary reported both values; the final split
files are authoritative for training.

## 5. Canonical experiment lineage

```text
Model A-lite
  random initialization
  small 48K dataset, lite profile
        |
        v
Model A
  fine-tune A-lite on the same data with A1+A3/no-a2

Model B
  random initialization on the expanded 72K no-a2 dataset
        |
        v
Model C
  fresh-optimizer fine-tune of B on 95,738 graphs (72K + new ABC)

Model A
   |-----------------------------|
   v                             v
Model D                       Model E
unweighted CE                exact Model A weights
BN updates                   weighted CE
8 epochs                     BatchNorm frozen
                             5 epochs

Model F
  random initialization
  full A1+A3 from epoch 0
  same 66,864 replay+unique-ABC split
```

## 6. Loss and optimization summary

| Model | Initialization | Loss | Class weights actually enabled | BN policy | Base LR | A1/A3 LR | Epoch state |
|---|---|---|---|---|---:|---:|---|
| A-lite | Random | CE | `[0.93349, 1.07052, 0.99599]` | Update | legacy/default | n/a | retained checkpoint epoch 32 |
| A | A-lite | CE | `[0.93349, 1.07052, 0.99599]` | Update | `1e-4` | `1e-3` | retained checkpoint epoch 25; configured 30 |
| B | Random | CE | `[0.98405, 1.14453, 0.87141]` | Update | `2e-3` | `2e-3` | epoch 49, step 80,600 |
| C | B | CE | Disabled | Update | `1e-4` | `1e-4` | best-v9 epoch 9, step 21,250; configured 15 |
| D | A | CE | Disabled | Update | `2e-5` | `2e-5` | best-v7 epoch 7, step 11,944 |
| E | A | CE | exact embedded Model A weights | `freeze_all` | `1e-5` | `1e-5` | best-v3 epoch 3/step 5,972; last epoch 4/step 7,465 |
| F | Random | CE | Disabled (`[1,1,1]`) | Update | `2e-3` | `2e-3` | last epoch 47, step 71,664; configured 50 |

Important: a non-unit `class_weights` tensor may remain stored inside a checkpoint
even when the fine-tuning run says class weighting was disabled. The run log and
the explicit `reuse_checkpoint_class_weights` flag determine whether weights were
actually applied.

## 7. Experiment details

## 7.1 Related older 53K lite baseline

Checkpoint:

```text
model_checkpoints\53k_thread_text\last.ckpt
```

Run:

```text
thread_text_53k_good_balance
```

Configuration:

| Item | Value |
|---|---|
| Data | `Z:\thread_and_text\lite` |
| Profile | lite |
| Initialization | random |
| Epoch/step | 99 / 850,000 |
| Configured epochs | 100 |
| Loss | weighted CE |
| Weights | `[1.01875, 0.96246, 1.01879]` |
| Batch | 8, gradient accumulation 2 |
| Precision | 16-mixed |
| Dropout | 0.2 |

This is retained as an older baseline but is not the source checkpoint of the
canonical Model A A1+A3 lineage. No Klavuz evaluation was attached to this
checkpoint in the current campaign.

## 7.2 Model A-lite: small mixed data, lite profile

Checkpoint:

```text
model_checkpoints\abc_included_48k\last.ckpt
```

Run:

```text
thread_text_lite_abc_jsons
```

Configuration:

| Item | Value |
|---|---|
| Data | approximately 30K CADSynth + 10K ABC; retained store 48,455 |
| Profile | lite, no A1/A2/A3 |
| Initialization | random |
| Loss | weighted CE |
| Class weights | `[0.93349, 1.07052, 0.99599]` |
| Batch | 8 |
| Gradient accumulation | 4 |
| Precision | 16-mixed |
| Retained checkpoint | epoch 32, step 105,897 |
| Configured max epochs | 100 |

The original narrative described this phase as approximately 40 epochs. The
retained source checkpoint used for the next stage contains epoch 32 metadata.

## 7.3 Model A: lite checkpoint fine-tuned with A1+A3

Checkpoint:

```text
model_checkpoints\abc_with_no_a2\last-v1.ckpt
```

Run:

```text
thread_text_a1_a3_finetune_20260720_182409
```

Configuration:

| Item | Value |
|---|---|
| Source | `thread_text_lite_abc_jsons\last.ckpt` |
| Data | same Model A data under `Z:\thread_and_text\no_a2` |
| Profile | no-a2: A1+A3 |
| Loss | weighted CE |
| Class weights | `[0.93349, 1.07052, 0.99599]` |
| Backbone LR | `1e-4` |
| A1/A3 LR | `1e-3` |
| Optimizer warmup | 1,000 steps |
| A1/A3 activation | scale 0.1 → 1.0 across first 5 epochs |
| A3 cap | 768 |
| Configured epochs | 30 |
| Retained checkpoint | epoch 25, step 35,451 |

The run was resumed at least once. The retained `last-v1.ckpt` is the source used
for Models D and E.

### Model A Klavuz result

Using the corrected 502-face reference:

| Metric | Value |
|---|---:|
| Accuracy | 97.41% |
| mIoU | 94.70% |
| Prediction counts | Stock 230, Thread 166, Text 106 |
| Stock recall | 94.65% |
| Thread precision | 96.99% |
| Thread recall | 100% |
| Text recall | 100% |
| Total errors | 13 |

Known errors:

- Stock→Text: faces 99, 100, 102, 103, 104, 107, 498, 500.
- Stock→Thread: faces 366, 367, 368, 371, 372.

This remains the best Klavuz model in the campaign.

## 7.4 Model B: 72K A1+A3 model trained from scratch

Checkpoint:

```text
model_checkpoints\abc_with_no_a2_no_finetuning\last.ckpt
```

Run:

```text
thread_text_no_a2_70k_optimized_20260720_235522
```

Configuration:

| Item | Value |
|---|---|
| Source | random initialization |
| Data | 72,197 no-a2 graphs |
| Profile | full A1+A3 from epoch 0 |
| Loss | weighted CE |
| Class weights | `[0.98405, 1.14453, 0.87141]` |
| Base/A1/A3 LR | `2e-3` |
| Warmup | 1,000 steps |
| Batch | max 64 with `4,000,000` node² budget |
| Precision | 16-mixed |
| A3 cap | 768 |
| Retained checkpoint | epoch 49, step 80,600 |
| Configured max | 100, but retained run stopped at 50 completed epochs |

### Model B Klavuz result

| Metric | Value |
|---|---:|
| Accuracy | 76.49% |
| mIoU | 68.28% |
| Prediction counts | Stock 139, Thread 272, Text 91 |
| Stock recall | 54.32% |
| Thread precision | 59.19% |
| Thread recall | 100% |
| Text recall | 92.86% |
| Errors | 118 |

Failure mode: severe Stock→Thread overprediction. All native Threads were found,
but 111 Stock faces were classified as Thread.

## 7.5 Model C: Model B fine-tuned on 72K plus new ABC

Checkpoint retained for ONNX/Klavuz comparison:

```text
model_checkpoints\30k_abc_finetuning\best-v9.ckpt
```

Run:

```text
thread_text_new_abc_finetune_v1
```

Configuration:

| Item | Value |
|---|---|
| Source | Model B `thread_text_no_a2_70k_optimized...\last.ckpt` |
| Optimizer | fresh optimizer from pretrained weights |
| Data | 95,738 graphs under `no_a2_72k_plus_new_abc` |
| Loss | unweighted CE; training log says class weights disabled |
| Base/A1/A3 LR | `1e-4` |
| Warmup | 500 steps |
| A1/A3 scale | 1.0 from epoch 0 |
| Batch | max 64, `4,000,000` node² budget |
| A3 cap | 768 |
| Configured epochs | 15 |
| best-v9 | epoch 9, step 21,250 |

### Model C best-v5 full test

The earlier `best-v5.ckpt` candidate was evaluated on 9,622 test graphs:

| Metric | Value |
|---|---:|
| Per-face accuracy | 99.116% |
| mIoU | 98.395% |
| Stock precision / recall | 98.11% / 99.63% |
| Thread precision / recall | 99.70% / 99.66% |
| Text precision / recall | 99.74% / 98.33% |

Strict Stock holdout:

| Item | Value |
|---|---:|
| Parts | 409 |
| Faces | 48,830 |
| Stock→Thread | 0 |
| Stock→Text | 2 faces |
| Parts with a false feature | 2/409 |

The corrected table contained 48,817 Stock faces and 13 Text faces. Two Stock
faces were predicted as Text.

### Model C best-v9 Klavuz result

| Metric | Value |
|---|---:|
| Accuracy | 81.08% |
| mIoU | 74.39% |
| Prediction counts | Stock 150, Thread 255, Text 97 |
| Stock recall | 61.32% |
| Thread precision | 63.14% |
| Thread recall | 100% |
| Text recall | 98.98% |
| Errors | 95 |

Failure mode: the new ABC fine-tune improved Model B but retained a broad Thread
decision region, producing 94 Stock→Thread errors.

## 7.6 Model D: Model A fine-tuned on replay plus unique ABC, unweighted

Checkpoint:

```text
model_checkpoints\abc_unique_prev_lite_and_noa2_finetuning\best-v7.ckpt
```

Run:

```text
thread_text_model_a_unique_abc_finetune_v1
```

Configuration:

| Item | Value |
|---|---|
| Source | Model A `last-v1.ckpt` |
| Data | 66,864 Model A replay + unique ABC train/val/test graphs |
| Strict holdout | 407 Stock-only parts |
| Loss | unweighted CE |
| Base/A1/A3 LR | `2e-5` |
| Warmup | 500 steps |
| A1/A3 scale | 1.0 from epoch 0 |
| BN | normal training/update |
| Batch | max 64, `4,000,000` node² budget |
| Epochs | 8: epochs 0–7 |
| best-v7 | epoch 7, step 11,944 |

The run initially started, was interrupted, and then resumed exactly from
`last.ckpt`; epoch 7 completed and `max_epochs=8` stopped training normally.

### Model D full test

| Metric | Value |
|---|---:|
| Per-face accuracy | 98.634% |
| mIoU | 97.426% |
| Stock precision / recall | 98.54% / 98.18% |
| Thread precision / recall | 99.64% / 98.77% |
| Text precision / recall | 97.99% / 99.08% |

### Model D Klavuz result

| Metric | Value |
|---|---:|
| Accuracy | 67.53% |
| mIoU | 52.58% |
| Prediction counts | Stock 404, Thread 0, Text 98 |
| Thread recall | 0% |
| Thread misses | 161/161 |
| Errors | 163 |

Failure mode: complete native-Thread collapse on Klavuz despite excellent
in-distribution metrics.

Diagnostic BatchNorm hybrids showed:

- Model A: 161/161 Klavuz Threads;
- Model D: 0/161;
- Model D weights with Model A BN buffers: 153/161;
- Model A weights with Model D BN buffers: 0/161.

Conclusion: BatchNorm running-stat drift was sufficient to explain most of this
run's collapse.

## 7.7 Model E: Model A fine-tuned with frozen BatchNorm and source weights

Checkpoints:

```text
model_checkpoints\abc_finetune_forzen_bc\best-v3.ckpt
model_checkpoints\abc_finetune_forzen_bc\last.ckpt
```

Run:

```text
thread_text_model_a_unique_abc_bn_frozen_weighted_v1
```

Configuration:

| Item | Value |
|---|---|
| Source | Model A `last-v1.ckpt` |
| Data | same 66,864 replay+unique-ABC split as Model D |
| Loss | weighted CE |
| Weights | exact checkpoint-embedded `[0.93349, 1.07052, 0.99599]` |
| BatchNorm | `freeze_all`: running statistics and affine parameters frozen |
| Frozen BN modules | 23 |
| Frozen BN affine tensors | 46 |
| Base/A1/A3 LR | `1e-5` |
| Warmup | 500 steps |
| Seed | 42 |
| Configured epochs | 5 |
| best-v3 | epoch 3, step 5,972 |
| last | epoch 4, step 7,465 |

Only approximately 5.5K BN parameters were non-trainable; approximately 5.4M
encoder/classifier parameters remained trainable.

### Model E Klavuz results

Both checkpoints produced identical class decisions on all 502 faces:

| Checkpoint | Stock | Thread | Text | Mean confidence |
|---|---:|---:|---:|---:|
| best-v3 | 405 | 0 | 97 | 96.29% |
| last | 405 | 0 | 97 | 94.69% |

Corrected-reference metrics:

| Metric | Value |
|---|---:|
| Accuracy | 67.73% |
| mIoU | 52.99% |
| Stock recall | 100% |
| Thread recall | 0% |
| Text recall | 98.98% |
| Errors | 162 |

Freezing BatchNorm did not solve the native-Thread collapse.

Module swaps showed:

- replacing only Model A's small attention fusion with Model E attention retained
  160/161 Threads;
- Model E encoder with Model A head retained only 8/161;
- Model A encoder with Model E head retained 0/161.

Interpretation: once BN drift was removed, the encoder and classifier still
co-adapted away from Model A's native-thread representation. Four completed
epochs represented 5,972 optimizer updates, not four small changes.

## 7.8 Model F: full A1+A3 training from scratch on the combined dataset

Checkpoint:

```text
model_checkpoints\thread_text_full_a1_a3_scratch_abc70k_training\last.ckpt
```

Run:

```text
thread_text_full_a1_a3_scratch_abc70k_v1
```

Configuration:

| Item | Value |
|---|---|
| Initialization | random |
| Data | logical `abc_for_modelA_finetuning` dataset |
| Approximate narrative composition | about 30–40K CADSynth and about 30K ABC |
| Authoritative split size | 66,864 train/val/test graphs + 407 strict Stock |
| Profile | full A1+A3 from epoch 0 |
| Loss | unweighted CE |
| Class weights | `[1,1,1]` |
| Base/A1/A3 LR | `2e-3` |
| Warmup | 1,000 steps |
| A1/A3 ramp | none |
| BN | normal update |
| Seed | 42 |
| Batch | max 64, `4,000,000` node² budget |
| Configured epochs | 50 |
| Supplied last checkpoint | epoch 47, step 71,664 |

The supplied `last.ckpt` did not reach epochs 48–49.

### Model F Klavuz result

| Metric | Value |
|---|---:|
| Accuracy | 80.48% |
| mIoU | 73.74% |
| Prediction counts | Stock 147, Thread 258, Text 97 |
| Stock precision / recall | 99.32% / 60.08% |
| Thread precision / recall | 62.40% / 100% |
| Text precision / recall | 100% / 98.98% |
| Errors | 98 |
| Mean confidence | 98.85% |
| Mean confidence on wrong faces | 96.97% |

Confusion matrix:

| True \ Predicted | Stock | Thread | Text |
|---|---:|---:|---:|
| Stock | 146 | 97 | 0 |
| Thread | 0 | 161 | 0 |
| Text | 1 | 0 | 97 |

Model F corrected all 13 known Model A errors and retained all 161 native
Threads, but introduced 97 other Stock→Thread errors.

It differed from Model C on only 9/502 face decisions. This shows that random
initialization on the combined distribution converges to almost the same broad
Thread concept as Model C.

Confidence-threshold calibration was not enough:

- best tested Thread threshold was approximately 0.958;
- accuracy improved only to approximately 83.7%;
- 160/161 Threads remained;
- 80 Stock→Thread errors still remained.

## 8. Klavuz comparison across all main models

Ground truth for this table is the original Model A CSV with the 13 known Model A
errors corrected to Stock. All other Model A decisions are treated as correct,
following the agreed evaluation method.

Reference distribution:

```text
Stock:  243
Thread: 161
Text:    98
Total:  502
```

| Model | Initialization/training path | Accuracy | mIoU | Stock recall | Thread precision | Thread recall | Predictions S/T/Tx |
|---|---|---:|---:|---:|---:|---:|---|
| A | lite small-data → no-a2 same data | 97.41% | 94.70% | 94.65% | 96.99% | 100% | 230 / 166 / 106 |
| B | scratch 72K no-a2 | 76.49% | 68.28% | 54.32% | 59.19% | 100% | 139 / 272 / 91 |
| C | B → 72K + new ABC | 81.08% | 74.39% | 61.32% | 63.14% | 100% | 150 / 255 / 97 |
| D | A → unique ABC, unweighted | 67.53% | 52.58% | 99.59% | n/a | 0% | 404 / 0 / 98 |
| E | A → unique ABC, BN frozen, weighted | 67.73% | 52.99% | 100% | n/a | 0% | 405 / 0 / 97 |
| F | scratch A1+A3 on A replay + unique ABC | 80.48% | 73.74% | 60.08% | 62.40% | 100% | 147 / 258 / 97 |

## 9. What the campaign established

### 9.1 In-distribution test metrics were not sufficient

Models C and D achieved approximately 98–99% test accuracy and class metrics, yet
failed strongly on Klavuz in opposite directions.

### 9.2 More ABC did not automatically define the correct boundary

ABC backgrounds improved real-world diversity, but inserted Thread features were
still synthetic. The training distribution lacked enough native real-world Thread
constructions and hard Stock negatives near that geometry.

### 9.3 Optimization path strongly affected the failure direction

- Model A fine-tuning on unique ABC moved native Threads into Stock.
- Random training on the same combined distribution moved many Stock faces into
  Thread.
- Model B/C/F independently converged toward broad Thread prediction.

### 9.4 BatchNorm was important but not the whole cause

BatchNorm drift explained most of Model D's collapse. Model E proved that encoder
and classifier updates can cause the same collapse even when BN is frozen.

### 9.5 Model A remains the best current real-part model

On Klavuz, Model A has only the 13 known false positives and retains all native
Threads and Text. Every later model either:

- overpredicts Thread on a large Stock region; or
- removes native Thread prediction entirely.

## 10. Aborted or non-comparable engineering runs

These runs did not produce model candidates and should not be compared as trained
models:

1. Early no-a2 scratch speed smoke tests failed before useful training because an
   in-place operation on a split autograd view (`q *= scaling`) was incompatible
   with that PyTorch path.
2. The first frozen-BN launch stopped before data loading because the copied
   `segmentation.py` expected a newer dataset constructor argument.
3. The next frozen-BN launch stopped on its first backward pass due GPU OOM. The
   filename-based node-count fallback packed about 64 large graphs per batch.
   Re-enabling the graph scan supplied actual node counts and fixed batching.
4. Missing-checkpoint and exact-resume launch errors for Model D occurred before
   training and did not create separate model variants.

## 11. Evidence and artifact index

### Checkpoints

```text
model_checkpoints\abc_included_48k\last.ckpt
model_checkpoints\abc_with_no_a2\last-v1.ckpt
model_checkpoints\abc_with_no_a2_no_finetuning\last.ckpt
model_checkpoints\30k_abc_finetuning\best-v9.ckpt
model_checkpoints\abc_unique_prev_lite_and_noa2_finetuning\best-v7.ckpt
model_checkpoints\abc_finetune_forzen_bc\best-v3.ckpt
model_checkpoints\abc_finetune_forzen_bc\last.ckpt
model_checkpoints\thread_text_full_a1_a3_scratch_abc70k_training\last.ckpt
```

### Klavuz comparisons

```text
artifacts\klavuz_onnx_comparison\all_faces_onnx_comparison_abcd.csv
artifacts\klavuz_onnx_comparison\comparison_summary_abcd.json
artifacts\klavuz_full_a1_a3_scratch_abc70k_last_onnx\all_face_comparison.csv
artifacts\klavuz_full_a1_a3_scratch_abc70k_last_onnx\analysis_summary.json
```

### ONNX packages

```text
migration_to_c++\migration_to_c\exported_a1_a3
migration_to_c++\migration_to_c\no_a2_72k_epoch50_onnx
migration_to_c++\migration_to_c\30k_abc_best-v9_onnx
migration_to_c++\migration_to_c\model_d_abc_unique_best-v7_onnx
migration_to_c++\migration_to_c\abc_finetune_froze_bn_onnx
migration_to_c++\migration_to_c\abc_finetune_frozen_bn_best-v3_onnx
migration_to_c++\migration_to_c\thread_text_full_a1_a3_scratch_abc70k_last_onnx
```

## 12. Known uncertainties and naming cautions

1. The user narrative described Model A as 40K–45K and approximately 40 lite
   epochs. The retained graph store contains 48,455 graphs and the source lite
   checkpoint reports epoch 32.
2. `best-vN.ckpt` normally means the Nth versioned best-checkpoint filename, not
   necessarily epoch N. Epoch values in this document come from checkpoint
   metadata. For Model E, `best-v3` happens to contain epoch 3.
3. Model C's full-test results were produced with `best-v5`; its Klavuz ONNX
   comparison used retained `best-v9`.
4. Klavuz “ground truth” is not an independently labeled face file. It is Model
   A's prediction CSV with 13 verified Model A mistakes corrected to Stock.
5. Counts such as “30K CADSynth + 30K ABC” are rounded descriptions. Final split
   files and checkpoint dataset paths are the authoritative training inputs.

