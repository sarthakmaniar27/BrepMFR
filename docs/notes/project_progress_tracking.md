# BrepMFR_PyG — Project Progress Tracking

This file is a living analysis of the files involved in the **ONNX PyG inference (Thread + Text, 3-class)** workflow, how they interact, and the main goals of the system. It is updated whenever the inference scripts change.

## 0. Latest update (2026-07-27) — Sample of C:\jsons NOT on no-Thread/Text allowlist

Allowlist files (`standalone_scripts/allowed_step_keys*.txt`, ~5128 unique STEP
keys) are the Stage-2 filter output: parts where inference found **no** Thread
or Text above confidence. JSON filenames in `C:\jsons` whose STEP key is
**absent** from those lists are the ones that kept Thread and/or Text.

Any 50 such files (sorted by name):

```text
00000001_1ffb81a71e5b402e966b9341_step_000_101.json
00000015_b08aa818955948c690fd9b6d_step_000_101.json
00000026_ad34a3f60c4a4caa99646600_step_006_101.json
00000030_ad34a3f60c4a4caa99646600_step_010_101.json
00000033_ad34a3f60c4a4caa99646600_step_013_101.json
00000041_c7d977f326364e35bb5b5d27_step_004_101.json
00000045_c7d977f326364e35bb5b5d27_step_008_101.json
00000064_767e4372b5f94a88a7a17d90_step_005_101.json
00000066_767e4372b5f94a88a7a17d90_step_007_101.json
00000067_767e4372b5f94a88a7a17d90_step_008_101.json
00000083_9ea934d3400e45fb9d190941_step_000_101.json
00000096_fcd027a9411a4ae7ac146826_step_000_101.json
00000098_72cf0f9ca5324822a4d41a80_step_000_101.json
00000100_5ed74bccca6f4e89829bcb5e_step_000_101.json
00000122_43c44d81ee2f44428314b518_step_001_101.json
00000129_e18c7feacfb54f3880659b43_step_004_101.json
00000130_e18c7feacfb54f3880659b43_step_005_101.json
00000148_d9a2aa6d24764b809c265460_step_001_101.json
00000192_222c0df7c7474deb967b5baa_step_001_101.json
00000197_826494e894454504b0971fd1_step_000_107.json
00000198_826494e894454504b0971fd1_step_001_101.json
00000199_826494e894454504b0971fd1_step_002_101.json
00000203_89aafdfc86b04301be0478c6_step_001_101.json
00000209_33bd159d563f438fbbebd9fa_step_001_101.json
00000214_5324c79b172f41e8a49afc6f_step_001_102.json
00000218_5324c79b172f41e8a49afc6f_step_005_103.json
00000220_5324c79b172f41e8a49afc6f_step_007_101.json
00000221_5324c79b172f41e8a49afc6f_step_008_101.json
00000223_97b354a907114b7183faa9c4_step_001_101.json
00000227_000ace8ff6634150be81fe86_step_001_101.json
00000229_682d529d5e8c4387bc59b3a1_step_000_101.json
00000231_f2f1249e743349808fca42a3_step_000_101.json
00000244_4829bc0645d6414d9b5cec31_step_007_101.json
00000247_c78d3dcf76d64ec9b46e7cef_step_002_101.json
00000252_a367cf6202d34e9ca31daf52_step_000_109.json
00000277_c740d53bc77c4e1b871b89cf_step_000_101.json
00000286_09d7ee9962c8427aac9ba7cb_step_000_101.json
00000297_52a153d2417747b5926ff9df_step_002_101.json
00000300_52a153d2417747b5926ff9df_step_005_101.json
00000305_3062bccff48e47a2b9de05e3_step_000_101.json
00000306_3062bccff48e47a2b9de05e3_step_001_101.json
00000307_3062bccff48e47a2b9de05e3_step_002_101.json
00000308_3062bccff48e47a2b9de05e3_step_003_101.json
00000309_3062bccff48e47a2b9de05e3_step_004_101.json
00000310_3062bccff48e47a2b9de05e3_step_005_101.json
00000323_3062bccff48e47a2b9de05e3_step_018_101.json
00000324_3062bccff48e47a2b9de05e3_step_019_101.json
00000334_f5a32b1cc9c1466caf094d49_step_000_101.json
00000336_1a821c3883f7459a9615fca4_step_000_101.json
00000337_153efbd85ca64c4fb6b7ed5f_step_000_101.json
```

Related: `match_allowlist_vs_jsons_folder.py`, `run_onnx_json_batch_inference.py`
(`stage2_filter`), `filter_abc_steps_by_allowlist.py`.

---

## 0b. Prior update (2026-07-27) — Interpreted orphan TensorBoard events under `model_checkpoints/`

### File

`model_checkpoints/events.out.tfevents.1784928125.GR-SW66464.18940.0` (~403 KB)

### What it is

Binary TensorBoard event log written by PyTorch Lightning’s `TensorBoardLogger`
during Stage-1 training. Filename encodes start Unix time `1784928125`
(2026-07-24 21:22 UTC), hostname `GR-SW66464`, and process id `18940`.

It sits at the **root** of `model_checkpoints/` (misplaced vs the canonical layout
`results/logs/stage1/<run_name>/tensorboard/version_0/`). It is the only
`events.out.tfevents*` under `model_checkpoints/`.

### Which run

Embedded `meta/hparams_json` identifies:

| Field | Value |
|---|---|
| `run_name` / `experiment_name` | `thread_text_new_abc_finetune_v1` |
| Stage | stage1 |
| Git SHA | `a2ad0a2` |
| Dataset | `D:\thread_and_text\no_a2_72k_plus_new_abc` |
| Pretrain ckpt | Model B `thread_text_no_a2_70k_optimized_20260720_235522\last.ckpt` |
| Classes / loss | 3-class CE, no class weights |
| Epochs / batch | 15 epochs, batch ≤64, node² budget 4e6 |
| LR | AdamW 1e-4 → 5e-5 after ~epoch 7; A1/A3 scale fixed at 1.0 |
| Trainable params | ~5.39M |

This is **Model C** in `thread_text_training_experiment_history.md`
(`model_checkpoints/30k_abc_finetuning/best-v9.ckpt`).

### Training curve (validation, 15 epochs)

| Metric | Epoch 0 (best early) | Epoch 9 | Epoch 14 (last) | Worst epoch |
|---|---:|---:|---:|---|
| `per_face_accuracy` | 0.9951 | 0.9926 | 0.9833 | 0.9709 @ ep12 |
| `IoU` | 0.9913 | 0.9869 | 0.9708 | 0.9495 @ ep12 |
| `eval_loss` | 0.0143 | 0.0259 | 0.0541 | 0.0963 @ ep12 |
| `train_loss_epoch` | 0.0170 | 0.0077 | 0.0074 | — (generally ↓) |
| Stock IoU `c00` | 0.988 | 0.983 | 0.963 | 0.937 @ ep12 |
| Thread IoU `c01` | 0.997 | 0.996 | 0.993 | 0.987 @ ep12 |
| Text IoU `c02` | 0.988 | 0.981 | 0.956 | 0.924 @ ep12 |

Also logged: `val/confusion_matrix` images, `val/max_pred_prob_batch0`
histogram, LR param-group scalars, and model param text dumps.

### Interpretation

- In-distribution val stayed very high, but **did not improve monotonically**;
  epoch 12 spiked `eval_loss` and hurt Text (`c02`) most.
- Train loss kept falling while val metrics softened → mild overfitting /
  unstable val after mid-run; history notes `best-v9` at epoch 9 / step 21250,
  consistent with a mid-run checkpoint being preferred over the last epoch.
- Thread (`c01`) stayed near-perfect; Stock/Text IoU drifted more — matches
  later Klavuz failure mode (broad Thread region / Stock→Thread errors).

### Related system files

| File | Role vs this events log |
|---|---|
| `segmentation.py` | Stage-1 entry; creates TB logger under run logs |
| `callbacks/training_logging.py` | Builds `TensorBoardLogger`, pins `version_0` |
| `models/tensorboard_media.py` | Writes confusion matrix / per-class IoU-recall scalars |
| `models/brepseg_model.py` | Logs train/val loss, accuracy, IoU, A1/A3 LR/scale |
| `scripts/diagnostics/dump_tb_scalars.py` | CLI to dump scalars from a TB logdir |
| `model_checkpoints/30k_abc_finetuning/best-v9.ckpt` | Checkpoint artifact from this same run |

---

## 0b. Prior update (2026-07-27) — Klavuz evaluation reference is invalid; model ranking inverts

### Goal

Diagnose why every model with 98–99% in-distribution test accuracy fails on new
real parts, using `Klavuz_101` as the probe case, and audit the evaluation
reference itself.

### Headline finding

`Klavuz_101` is a **tap with three flutes**. Its helical thread is therefore cut
into three angular lands. The evaluation reference (Model A predictions plus 13
manual corrections) labels only **two** of the three lands as Thread. Faces
**274–361 (88 faces)** are the third land and are labelled Stock in the reference.

Verified geometrically against the source JSON, land 3 vs the two accepted lands:

| Property | Accepted Thread lands | Disputed band 274–361 |
|---|---|---|
| `face_type` | 100% type 6 | 100% type 6 |
| `face_loop` | 100% loop 1 | 100% loop 1 |
| `face_area` median | 0.0011 | 0.0011 |
| radial distance | 0.075–0.111 | 0.078–0.109 |
| axial span | 0.007–0.601 | 0.005–0.603 |
| angular sector | 90–202°, 315–90° | 180–315° |

The three lands tile the full 360° at identical radius and axial span, and every
axial slice from z=0.00 to z=0.60 contains faces from all three. This is one
helical thread surface interrupted by flutes, not a thread plus a stock region.

### Corrected scoring

Re-scoring `artifacts/klavuz_full_a1_a3_scratch_abc70k_last_onnx/all_face_comparison.csv`
after relabelling faces 274–361 as Thread:

| Model | Accuracy (reference as-is) | Accuracy (land 3 corrected) | Errors (corrected) |
|---|---:|---:|---:|
| F scratch, replay+uniqueABC | 80.48% | **97.21%** | 14 |
| C B → +new ABC | 81.08% | **97.01%** | 15 |
| B scratch 72K | 76.49% | **94.02%** | 30 |
| A lite → A1+A3 | 97.41% | **79.88%** | 101 |
| E A → finetune, BN frozen | 67.73% | **50.20%** | 250 |
| D A → finetune, BN update | 67.53% | **50.00%** | 251 |

The ranking inverts. Models B, C and F were never over-predicting Thread on this
part; they were finding a thread land that Model A misses. Model A's median
confidence on land 3 is 0.579, i.e. it never learned this geometry and sat on the
decision boundary.

Model F's 14 remaining errors under the corrected reference are the flute/gash
faces (462, 465, 469, 473, 474), thread runout faces (110–112, 192, 271, 272),
two land-end faces (355, 361) and one Text face (105).

### Hypotheses tested and ruled out

| Hypothesis | Verdict | Evidence |
|---|---|---|
| Absolute scale / no normalization | **Ruled out** | The SolidWorks macro normalizes to a ±1 box. Klavuz `coord_absmax`=1.0, `bbox_diag`=1.85 vs training p5–p95 of 1.66–2.91. |
| Approved ABC Stock parts contain native thread labelled Stock | **Ruled out** | 0/426 approved parts contain a Stock-labelled cluster with the real-thread signature. |
| Klavuz thread region too large / out of range | **Ruled out** | 249 thread faces = 89th percentile; thread `edge_len` median 0.103 = 85th percentile. Inside distribution. |
| Class prior shift caused the D/E collapse | **Weak** | Stock:Thread went 1.33:1 → 1.51:1 only. |

### Real coverage gap

Scanning 500 training parts for the Klavuz real-thread signature (type-6, loop-1
clusters ≥15 joined by tangent-continuous, smooth edges):

- **13.2%** of parts contain it labelled **Thread**;
- **0.0%** of parts contain it labelled **Stock**.

The mapping "thread-like helical band → Thread" is perfectly consistent in
training and has **no counterexample anywhere in 67K parts**. The class boundary
for this geometry family is therefore undefined by the data, which is why it
drifts arbitrarily between training runs.

### Fine-tuning diagnosis

Model A's decision on land 3 is marginal (0.579). The data supplies no supervision
for that boundary. Fine-tuning from A ran 11,944 (D) and 5,972 (E) optimizer
steps, so a marginal, unsupervised decision moved — and nothing in the setup
controls the direction. D and E moved it to Stock; F, trained from scratch with no
marginal initialization, moved it to Thread. The apparent "native-thread collapse"
is an unsupervised boundary drifting, amplified by a metric anchored to Model A.

### Secondary defect: circular `edge_ang` fed as a raw scalar

`edge_ang` is wrapped by `(a + π) % 2π − π` in
`scripts/inference/json_to_brepmfr_pyg.py` and then fed as a raw scalar through
`NonLinear(1, num_heads)`. All 165 of Klavuz's thread-thread edges have raw
dihedral exactly π; floating-point noise sends 152 of them to −π and 13 to +π.
Geometrically identical edges therefore receive scalar inputs 6.28 apart. Training
data shows the same split (19,868 negative vs 5,322 positive), so this is a
robustness defect rather than the root cause. Encoding `(sin θ, cos θ)` removes it.

### Diagnostic scripts added

| File | Role |
|---|---|
| `standalone_scripts/diag_klavuz_geometry.py` | Face-property comparison of the disputed band vs accepted thread |
| `standalone_scripts/diag_klavuz_angular.py` | Angular/axial decomposition proving the three-land structure |
| `standalone_scripts/diag_klavuz_plot.py` | Renders `artifacts/klavuz_thread_lands.png` |
| `standalone_scripts/diag_rescore_klavuz.py` | Re-scores all six models under corrected references |
| `standalone_scripts/diag_scale_shift.py` | Training vs Klavuz absolute-scale comparison |
| `standalone_scripts/diag_thread_signature.py` | Synthetic vs real thread feature signature |
| `standalone_scripts/diag_edge_angle_wrap.py` | Quantifies the ±π wrap discontinuity |
| `standalone_scripts/diag_stock_label_poison.py` | Searches for thread geometry labelled Stock |
| `standalone_scripts/diag_thread_extent.py` | Thread-region size percentiles |
| `standalone_scripts/diag_class_prior.py` | Face-level class priors of both datasets |

### Consequence for the campaign

Sections 8 and 9.5 of `thread_text_training_experiment_history.md` ("Model A
remains the best current real-part model") are artifacts of the circular
reference. Model selection should be redone against independently labelled real
parts before any further training runs.

## 1. Prior update (2026-07-26) — abc_finetune_forzen_bc/last.ckpt → ONNX `_froze_bn` → Klavuz_101

### Goal

Convert Lightning checkpoint
`model_checkpoints/abc_finetune_forzen_bc/last.ckpt` (frozen BatchNorm ABC
finetune) to A1+A3 (`no_a2`) ONNX named with `_froze_bn`, then run ONNX Runtime
inference on `\\Gr-sw66464\d\Demo\grab_cad_brepmfr_testing\jsons\Klavuz_101.json`
and write per-face CSV under `standalone_scripts/`.

### Pipeline (files and interactions)

```text
abc_finetune_forzen_bc/last.ckpt
    │
    ▼  migration_to_c++/migration_to_c/export_a1_a3_onnx.py
    │    loads models.brepseg_model.BrepSeg via Lightning state_dict
    │    wraps encoder+attention+classifier as BrepMFRONNXWrapper
    │    exports 16 flat tensors → probabilities [N, 3]
    │    (renamed export to brepmfr_a1_a3_froze_bn.onnx)
    ▼
abc_finetune_froze_bn_onnx/
    brepmfr_a1_a3_froze_bn.onnx
    model_config.json   (profile=no_a2, 3 classes)
    label_map.json      (0=Stock, 1=Thread, 2=Text)
    │
Klavuz_101.json ──► scripts/inference/json_to_brepmfr_pyg.build_pyg_from_json_path
    │                  inference_profile="no_a2"  (builds spatial_pos + edge_path)
    ▼
PyG Data (502 faces, 2902 edges)
    │
    ▼  standalone_scripts/run_onnx_a1_a3_inference.py helpers
    │    ensure_a1_a3_graph → make_a1_a3_batch → batch_to_ort_feed
    │    → onnxruntime.InferenceSession.run(["probabilities"], …)
    ▼
standalone_scripts/Klavuz_101_froze_bn_predictions.csv
```

### Result (2026-07-26)

| Item | Value |
|------|-------|
| ONNX | `migration_to_c++/migration_to_c/abc_finetune_froze_bn_onnx/brepmfr_a1_a3_froze_bn.onnx` (~20.7 MB, opset 17) |
| Export validation | PyTorch↔ORT max_diff ≤ 1.2e-7 on dummy graphs; all labels match |
| Faces predicted | 502 |
| Class counts | Stock=405, Text=97, Thread=0 |
| Mean confidence | 0.94686562 |
| Output CSV | `standalone_scripts/Klavuz_101_froze_bn_predictions.csv` |

CSV columns: `face_index`, `predicted_class_id`, `predicted_label`, `confidence`,
`prob_Stock`, `prob_Thread`, `prob_Text`.

### Key file roles in this slice

| File | Role |
|------|------|
| `export_a1_a3_onnx.py` | CKPT → ONNX + config/label_map; A1+A3 wrapper (no A2) |
| `json_to_brepmfr_pyg.py` | Raw B-rep JSON → PyG tensors; `no_a2` keeps A1/A3 |
| `run_onnx_a1_a3_inference.py` | Collate single graph to 16 ORT inputs; write prediction CSVs |
| `models/brepseg_model.py` | BrepSeg Lightning module (incl. frozen BN training path) loaded from CKPT |
| `data/collator.py` | Training collator mirrored by `make_a1_a3_batch` for ORT feed |

### System goal (unchanged)

Per-face manufacturing feature recognition on B-rep graphs: Stock / Thread / Text
for the 3-class Thread+Text line; ONNX enables C++ deployment without PyTorch.

## 0. Prior update (2026-07-25) — best-v7.ckpt → ONNX → Klavuz_101 JSON inference

### Goal

Convert Lightning checkpoint
`model_checkpoints/abc_unique_prev_lite_and_noa2_finetuning/best-v7.ckpt`
to A1+A3 (`no_a2`) ONNX, then run ONNX Runtime inference on
`\\Gr-sw66464\d\Demo\grab_cad_brepmfr_testing\jsons\Klavuz_101.json` and write
per-face CSV under `standalone_scripts/`.

### Pipeline (files and interactions)

```text
best-v7.ckpt
    │
    ▼  migration_to_c++/migration_to_c/export_a1_a3_onnx.py
    │    loads models.brepseg_model.BrepSeg via Lightning state_dict
    │    wraps encoder+attention+classifier as BrepMFRONNXWrapper
    │    exports 16 flat tensors → probabilities [N, 3]
    ▼
model_d_abc_unique_best-v7_onnx/
    brepmfr_a1_a3.onnx
    model_config.json   (profile=no_a2, 3 classes, dim_node=256, d_model=512, …)
    label_map.json      (0=Stock, 1=Thread, 2=Text)
    │
Klavuz_101.json ──► scripts/inference/json_to_brepmfr_pyg.build_pyg_from_json_path
    │                  inference_profile="no_a2"  (builds spatial_pos + edge_path)
    ▼
PyG Data (502 faces, 2902 edges)
    │
    ▼  standalone_scripts/run_onnx_a1_a3_inference.py helpers
    │    ensure_a1_a3_graph → make_a1_a3_batch → batch_to_ort_feed
    │    → onnxruntime.InferenceSession.run(["probabilities"], …)
    ▼
standalone_scripts/Klavuz_101_predictions.csv
```

### Result (2026-07-25)

| Item | Value |
|------|-------|
| ONNX | `migration_to_c++/migration_to_c/model_d_abc_unique_best-v7_onnx/brepmfr_a1_a3.onnx` (~20.7 MB, opset 17) |
| Export validation | PyTorch↔ORT max_diff ≤ 1.2e-7 on dummy graphs; all labels match |
| Faces predicted | 502 |
| Class counts | Stock=404, Text=98, Thread=0 |
| Mean confidence | 0.96972960 |
| Output CSV | `standalone_scripts/Klavuz_101_predictions.csv` |

CSV columns: `face_index`, `predicted_class_id`, `predicted_label`, `confidence`,
`prob_Stock`, `prob_Thread`, `prob_Text`.

### Key file roles in this slice

| File | Role |
|------|------|
| `export_a1_a3_onnx.py` | CKPT → ONNX + config/label_map; A1+A3 wrapper (no A2) |
| `json_to_brepmfr_pyg.py` | Raw B-rep JSON → PyG tensors; `no_a2` keeps A1/A3 |
| `run_onnx_a1_a3_inference.py` | Collate single graph to 16 ORT inputs; write prediction CSVs |
| `models/brepseg_model.py` | BrepSeg Lightning module loaded from CKPT |
| `data/collator.py` | Training collator mirrored by `make_a1_a3_batch` for ORT feed |

### System goal (unchanged)

Per-face manufacturing feature recognition on B-rep graphs: Stock / Thread / Text
for the 3-class Thread+Text line; ONNX enables C++ deployment without PyTorch.

## 0. Prior update (2026-07-25) — Stock summarize CSV name mismatch

### Bug

`run_thread_pyg_inference.py` wrote `{stem}.csv`, while
`summarize_stock_only_inference.py` only globbed `*_predictions.csv` → false
"no csv found" even when per-graph CSVs existed.

### Fix

- Summarizer accepts `*_predictions.csv` or plain `{stem}.csv` (skips metrics).
- Inference writer now emits `{stem}_predictions.csv` for new runs.

## 0. Prior update (2026-07-24) — Rewrite approved-list JSON roots

### Problem

Approved list paths pointed at `C:\jsons\*.json` but files live under
`D:\thread_and_text\stock_abc_json`.

### Fix

`scripts/threads/rewrite_approved_json_paths.ps1` — keeps filenames, swaps root,
writes `*_rewritten.txt`, verifies files exist unless `-SkipExistsCheck`.

## 0. Prior update (2026-07-24) — prepare_new_abc_finetune_data.ps1 -Apply

### What `-Apply` does

1. Writes Stock JSON copies from `$ApprovedList` → `$StockJsonDir` (needs
   `-OverwriteStockJsons` if Option B already filled that dir).
2. Calls `prepare_no_a2_scratch_delta.ps1` to build `$CombinedNoA2Root`:
   seed old `.pt`, remap+convert new/Stock JSONs, splits, class weights, validate.

Dry-run (no `-Apply`) only audits Stock list + remappable labels in the 25K folder.

## 0. Prior update (2026-07-24) — Path variable origins for finetune prep

| Variable | How you get it |
|----------|----------------|
| `$ApprovedList` | Already have: `C:\jsons\inference\no_confident_thread_or_text.txt` |
| `$IdentityMap` | Already in repo: `scripts/threads/remap_maps/thread_text_sw_to_brep_with_identity.json` |
| `$CombinedNoA2` | **Output dir** created by `prepare_new_abc_finetune_data.ps1 -Apply` |
| `$ClassWeights` | **Output file** written by same prep (via `compute_class_weights.py`) |

Do not create Combined/class-weights by hand; run the preparer.

## 0. Prior update (2026-07-24) — After STEP allowlist export → Stock JSON Option B

### Status

User ran `standalone_scripts/export_step_allowlist_from_inference.py`, which wrote
STEP-key allowlists (`allowed_step_keys*.txt`, ~5128 keys). That is for Stage-2
STEP filtering / distribute — **not** Stock JSON creation.

Approved JSON path list already exists:
`C:\jsons\inference\no_confident_thread_or_text.txt` (~5128 absolute `.json` paths).

### Next (Option B)

1. Dry-run `prepare_approved_abc_stock_jsons.py` (no `--write`).
2. Apply with `--write` → `$StockJson` + `stock_label_manifest.csv`.
3. Then continue combined-dataset prep (`prepare_new_abc_finetune_data.ps1`).

## 0. Prior update (2026-07-24) — StockJson is generated, not pre-existing

### Clarification

`$StockJson` (`approved_abc_stock_json`) is an **output** of
`prepare_approved_abc_stock_jsons.py`. Inputs are only:

1. `C:\jsons\*.json` (or whatever paths the approved list points at)
2. `C:\jsons\inference\no_confident_thread_or_text.txt`

The script copies each approved path to `$StockJson` and sets every face label
to `0`. It never edits `C:\jsons`. Do not manually filter the whole JSON root
into Stock.

## 0. Prior update (2026-07-24) — New ABC fine-tune data preparation

### Main goal

Combine three sources into one training-ready `no_a2` dataset for fine-tuning the
72K Thread/Text classifier (classes `0=Stock`, `1=Thread`, `2=Text`):

1. Existing ~72K `no_a2` PyG graphs (seeded, not modified).
2. ~25K synthetic ABC JSONs with Thread/Text labels.
3. ~4,834 approved original ABC JSONs (no confident Thread/Text) pseudo-labeled
   entirely as Stock=`0`.

Operator guide: `training_on_new_abc_document.md`.

### File roles and interactions

```text
ApprovedList (no_confident_thread_or_text.txt)
        |
        v
prepare_approved_abc_stock_jsons.py  -->  StockJson/*.json + stock_label_manifest.csv
        |
New25KJson + StockJson + OldNoA2/pyg
        |
        v
prepare_new_abc_finetune_data.ps1  (wrapper)
        |
        v
prepare_no_a2_scratch_delta.ps1
  -> seed old .pt (HardLink/Copy)
  -> remap new labels (identity map)
  -> json_to_brepmfr_pyg_optimized.py (no_a2)
  -> make_random_splits.py (_step_NNN families)
  -> compute_class_weights.py
  -> validate_a1_a3_finetune_data.py
        |
        v
CombinedNoA2/{pyg, train.txt, val.txt, test.txt}
        |
        v
make_stock_only_eval_split.py  -->  stock_only_test.txt (fixed eval)
```

| File | Role |
|------|------|
| `training_on_new_abc_document.md` | End-to-end prep + fine-tune + Stock→Text eval. |
| `scripts/threads/prepare_approved_abc_stock_jsons.py` | Audit approved list; write Stock-only JSON copies (never edits `C:\jsons`). |
| `scripts/threads/prepare_new_abc_finetune_data.ps1` | Operator wrapper: dry-run or `-Apply` full build. |
| `scripts/threads/prepare_no_a2_scratch_delta.ps1` | Underlying seed/remap/convert/split/weights/validate. |
| `scripts/inference/json_to_brepmfr_pyg_optimized.py` | JSON → PyG with `--inference_profile no_a2`. |
| `scripts/threads/make_random_splits.py` | STEP-key-aware 80/10/10 splits. |
| `scripts/training/compute_class_weights.py` | Weights from new train split. |
| `scripts/threads/validate_a1_a3_finetune_data.py` | Labels `[0,2]`, A1/A3 tensors, quarantine. |
| `scripts/threads/make_stock_only_eval_split.py` | Untouched Stock originals in val/test. |
| `scripts/threads/remap_maps/thread_text_sw_to_brep_with_identity.json` | Remap map for 25K labels. |

### Critical constraints

- Only paths in `C:\jsons\inference\no_confident_thread_or_text.txt` become Stock.
- Filenames must keep `_step_NNN` so families stay atomic across splits.
- Dry-run before `-Apply`; use HardLink when old+combined share an NTFS volume.

## 0. Prior update (2026-07-23) — Repair requeue after bad distribute commit

### Problem

First distribute run committed all pending keys (`distributed=4339`, `pending=0`)
even when most agents had empty/missing local STEP sources. Next distribute then
had nothing to ship.

### Fix files

| File | Role |
|------|------|
| `pipeline_dedup/requeue_distributed_keys.py` | Move distributed → pending; clear distributed. |
| `Jenkinsfile.pipeline_repair_requeue` | Sync `*.py` to share + run requeue. |
| `pipeline_dedup/commit_successful_keys.py` | Must exist on share before success-only commit. |

### Operator steps

1. Run `Pipeline-Repair-Requeue-Distributed` once (syncs scripts + requeues).
2. Re-run distribute with shared UNC source (`\\GR-SW65551\abc_steps`).

## 0. Prior update (2026-07-23) — Shared STEP source + success-only commit

### Main goal

Fix distribute for agents that have **no local** `C:\abc_steps`: copy from a
shared UNC STEP pool into each agent's `C:\abc_steps_filtered`, and only mark
keys distributed when copy/already-present succeeded.

### Files

| File | Role |
|------|------|
| `Jenkinsfile.pipeline_distribute_only` | `SOURCE_DIR` = shared UNC; 45m timeout; success stash per node. |
| `pipeline_dedup/commit_successful_keys.py` | Commit only `success_*.txt` keys; leave missing in pending. |
| `CONTINUOUS_PIPELINE.md` | Documents “empty C:\\abc_steps” anti-pattern. |

## 0. Prior update (2026-07-23) — Bootstrap state machine via Jenkins

### Main goal

Allow the one-time GR-SW66464 setup (folders, copy `pipeline_dedup` scripts, seed
`stage2_done_keys.txt` from `D:\thread_and_text\abc_json`) to run through Jenkins
instead of a manual RDP session.

### File

| File | Role |
|------|------|
| `standalone_scripts/Jenkinsfile.pipeline_bootstrap_state` | Idempotent bootstrap job on `GR-SW66464`. |
| `standalone_scripts/CONTINUOUS_PIPELINE.md` | Documents Option A (Jenkins) vs Option B (manual). |

### Interaction

Job checks out SCM (or uses `SCRIPT_SOURCE`) → copies `*.py` to
`D:\thread_and_text\pipeline_scripts` → runs `seed_stage2_done_keys.py` → creates
empty sibling ledgers under `pipeline_state`. Other continuous jobs then assume
those paths exist.

## 0. Prior update (2026-07-23) — Continuous Stage-1/Stage-2 Jenkins pipeline (dedup)

### Main goal

Run Stage-1 (filter no Thread/Text STEPs) and Stage-2 (synthetic thread/text on
remote agents) **in parallel**, without reprocessing keys already filtered,
distributed, or completed — including the ~3k STEP keys already present as
~10k JSONs under `D:\thread_and_text\abc_json` on GR-SW66464.

### Architecture

```text
Stage-1 job (hourly)          Distribute job (~15 min)       Cleanup job (hourly)
infer + enqueue NEW keys  -->  append STEPs to agents'   --> harvest C:\Threads\jsons
                               C:\abc_steps_filtered         prune finished STEPs
                               (NO CLI)                      merge → stage2_done

Central ledgers on GR-SW66464:
  D:\thread_and_text\pipeline_state\
    pending_keys.txt / stage1_seen_keys.txt
    stage2_distributed_keys.txt / stage2_done_keys.txt
```

### Files added / roles

| File | Role |
|------|------|
| `standalone_scripts/CONTINUOUS_PIPELINE.md` | Operator guide for continuous parallel flow. |
| `standalone_scripts/pipeline_dedup/key_utils.py` | Shared STEP-key parse + ledger IO. |
| `standalone_scripts/pipeline_dedup/seed_stage2_done_keys.py` | Seed `stage2_done` from `abc_json`. |
| `standalone_scripts/pipeline_dedup/enqueue_filtered_keys.py` | Allowlist → pending (dedup). |
| `standalone_scripts/pipeline_dedup/plan_distribute_chunks.py` | Pending → per-node chunks + commit. |
| `standalone_scripts/pipeline_dedup/append_steps_from_allowlist.py` | Local append-copy helper. |
| `standalone_scripts/pipeline_dedup/cleanup_agent_filtered.py` | Agent harvest/prune helper. |
| `standalone_scripts/pipeline_dedup/merge_harvested_done_keys.py` | Merge harvests into done ledger. |
| `standalone_scripts/Jenkinsfile.pipeline_stage1_enqueue` | Stage-1 infer + enqueue job. |
| `standalone_scripts/Jenkinsfile.pipeline_distribute_only` | Distribute-only (no CLI, no wipe). |
| `standalone_scripts/Jenkinsfile.pipeline_cleanup_stage2` | Periodic prune + done-ledger merge. |

### How they interact

1. Seed once: `seed_stage2_done_keys.py` reads `D:\thread_and_text\abc_json` → `stage2_done_keys.txt`.
2. Stage-1 uses existing `run_onnx_json_batch_inference.py` + `export_step_allowlist_from_inference.py`, then `enqueue_filtered_keys.py` so only keys not in pending/distributed/done enter the queue.
3. Distribute plans chunks on GR-SW66464, stash/unstash to 10 agents, **append**-copies from each agent's `C:\abc_steps` → `C:\abc_steps_filtered`, then commits keys to `stage2_distributed`.
4. Local Stage-2 CLI (outside Jenkins) consumes filtered STEPs → `C:\Threads\jsons`.
5. Cleanup harvests JSON keys per agent, deletes finished STEPs from filtered folders, merges into `stage2_done` and prunes pending/distributed.

### System goals

1. Parallel Stage-1 production and Stage-2 consumption (queue + cron jobs).
2. Idempotent ledgers so no STEP key is inferred/distributed/synthesized twice.
3. Jenkins owns distribution + cleanup only; CLI stays off the Jenkins critical path.

## 0. Prior update (2026-07-22) — End-to-end ABC JSON filter pipeline doc

### Main goal

Document the operator pipeline that turns ABC STEPs into filtered (no confident
Thread/Text) allowlisted STEPs on each Jenkins agent.

### Flow and file interactions

```text
BatchJsonExport.vba (+ Watchdog-StepOpen.ps1)
  -> C:\jsons
  -> delete_duplicate_jsons.py
  -> check_and_delete_covered_steps.py  (shrink C:\abc_steps_not_in_allowlist)
  -> run_onnx_json_batch_inference.py --skip-existing
       Stage-1 CSVs + Stage-2 no_confident_thread_or_text.*
  -> export_step_allowlist_from_inference.py -> allowed_step_keys.txt
  -> _gen_filter_jenkinsfile.py -> Jenkinsfile.filter_abc_steps_no_thread_text
  -> Jenkins: C:\abc_steps -> C:\abc_steps_filtered (per VM)
```

| File | Role in this pipeline |
|------|------------------------|
| `standalone_scripts/END_TO_END_PIPELINE.md` | Full checklist for steps 1–6 above. |
| `standalone_scripts/WORKFLOW_GUIDE.md` | Smaller task-set CLI details; links to end-to-end doc. |
| `standalone_scripts/BatchJsonExport.vba` | SolidWorks STEP→JSON export into `C:\jsons`. |
| `standalone_scripts/run_onnx_json_batch_inference.py` | Lite ONNX + Stage-2 filter. |
| `standalone_scripts/export_step_allowlist_from_inference.py` | Stage-2 list → STEP-key allowlist. |
| `standalone_scripts/Jenkinsfile.filter_abc_steps_no_thread_text` | Multi-node copy of allowlisted STEPs. |

### System goals

1. Recover cleanly when VBA export is interrupted (skip list + covered-STEP delete).
2. Infer only new JSONs; flag parts without confident Thread/Text.
3. Ship those keys as STEPs into `C:\abc_steps_filtered` via Jenkins.

## 0. Prior update (2026-07-22) — Standalone ops workflow guide

### Main goal

Document the three recurring housekeeping + inference loops operators run from
`standalone_scripts/`, so the correct script is used for each folder pair and
safety flag (`DRY_RUN` / `--skip-existing`).

### New / updated files

| File | Role |
|------|------|
| `standalone_scripts/WORKFLOW_GUIDE.md` | Task-set instructions derived from script defaults and CLI. |
| `standalone_scripts/README.md` | Points to the workflow guide. |

### Task-set → script map

| Task | Primary script | Interaction |
|------|----------------|-------------|
| 1. Clean duplicate JSONs in `C:\jsons` | `delete_duplicate_jsons.py` | Groups by `..._step_NNN`; keeps first JSON; deletes extras + `*.SLDPRT`. |
| 2. Delete STEPs already covered by JSONs | `delete_step_files.py` (or `delete_step_files_from_abc_jsons.py` / `check_and_delete_covered_steps.py`) | Matches JSON keys to `.step`/`.stp` in the STEP root; deletes covered STEPs only. |
| 3. Infer only new JSONs | `run_onnx_json_batch_inference.py --skip-existing` | Skips JSONs that already have `<stem>_predictions.csv` under `<json-dir>/inference/`. |

Optional report helper before deletes: `count_unique_json_vs_steps.py`.
A1+A3 path remains JSON → `json_to_brepmfr_pyg_optimized.py --inference_profile no_a2` →
`run_onnx_a1_a3_inference.py` (documented as Option B in the guide).

### System goals these scripts support

1. Keep the JSON export folder free of multi-body duplicates and SolidWorks temps.
2. Shrink the STEP backlog to only parts still needing JSON export.
3. Avoid re-running ONNX on JSONs that already have prediction CSVs.

## 0. Prior update (2026-07-22) — `run_onnx_a1_a3_inference.py` inference flow

### Main system goal (this script)

Run the exported A1+A3 / `no_a2` ONNX model on PyG `.pt` graphs and emit per-face
`Stock` / `Thread` / `Text` predictions (plus optional accuracy metrics when GT exists).

### File role and interactions

| File / artifact | Role |
|-----------------|------|
| `standalone_scripts/run_onnx_a1_a3_inference.py` | End-to-end runner: resolve graphs → validate A1+A3 tensors → build 16 ORT inputs → `session.run` → write CSVs/metrics. |
| `migration_to_c++/migration_to_c/no_a2_72k_epoch50_onnx/brepmfr_a1_a3.onnx` | Default ONNX graph; expects exactly the 16 A1+A3 inputs; single output `probabilities[N,3]`. |
| `*.pt` under `<dataset>/pyg` (or `--input`) | Stored PyG graphs produced with `--inference_profile no_a2` (must include `spatial_pos` + `edge_path`). |
| optional `label/<stem>.json` or `graph.label_feature` | Ground truth for metrics columns and aggregate confusion/IoU. |
| `--output-dir` CSVs | Always written: `<stem>_predictions.csv` + `onnx_inference_summary.csv`. |
| `--metrics-dir` (default = output-dir) | Written only when GT faces exist: `confusion_matrix.csv`, `per_class.csv`, `summary.md`. |

### Inference path (per graph)

```text
resolve_graphs (--dataset-path | --input)
  -> torch.load(.pt)
  -> ensure_a1_a3_graph (require spatial_pos + edge_path)
  -> make_a1_a3_batch (16 tensors; spatial_pos+1; edge_path pad/trunc to K=16)
  -> batch_to_ort_feed (float32 / int64 / bool NumPy)
  -> onnxruntime.InferenceSession.run -> probabilities[N,3]
  -> argmax -> predicted_class_id; max -> confidence
  -> write_predictions CSV; accumulate GT metrics if available
```

ONNX output is treated as probabilities already (no second softmax). Class map default:
`0=Stock`, `1=Thread`, `2=Text`.

### Outputs produced

1. **`<stem>_predictions.csv`** — one row per face: `face_index`, `predicted_class_id`,
   `predicted_label`, `confidence`, `prob_Stock`, `prob_Thread`, `prob_Text`, and when GT
   exists also `ground_truth_*` + `correct_top1`.
2. **`onnx_inference_summary.csv`** — one row per graph: path, CSV path, face count,
   mean confidence, class counts, `has_gt`, PASS/SKIP/FAIL, error text.
3. **Metrics (GT only)** — face-level confusion matrix, per-class precision/recall/IoU,
   and `summary.md` with accuracy and mIoU.

## 0. Prior update (2026-07-22) — GrabCAD no_a2 inference and confidence audit

The 28 raw B-rep JSONs under
`\\Gr-sw66464\d\Demo\grab_cad_brepmfr_testing\jsons` were converted and inferred
with the A1+A3/no_a2 ONNX package:

```text
jsons/*.json
  -> scripts/inference/json_to_brepmfr_pyg_optimized.py --inference_profile no_a2
  -> pyg_no_a2/pyg/*.pt
  -> standalone_scripts/run_onnx_a1_a3_inference.py
  -> migration_to_c++/migration_to_c/exported_a1_a3/brepmfr_a1_a3.onnx
  -> inference_csvs_no_a2/*_predictions.csv
```

Files and generated directories involved:

| Item | Role and result |
|------|-----------------|
| `scripts/inference/json_to_brepmfr_pyg_optimized.py` | Converted all 28 JSONs to A1+A3 graphs with `spatial_pos` and 16-hop `edge_path`; 28 passed and 0 failed. |
| `\\Gr-sw66464\d\Demo\grab_cad_brepmfr_testing\pyg_no_a2\pyg` | Reusable intermediate no_a2 graphs, one `.pt` per source JSON. |
| `standalone_scripts/run_onnx_a1_a3_inference.py` | Wrote face index, predicted class, confidence, and all three class probabilities to each per-part CSV. |
| `migration_to_c++/migration_to_c/exported_a1_a3/*` | Exact ONNX model, three-class label map (`Stock=0`, `Thread=1`, `Text=2`), and no_a2 model contract used for the run. |
| `\\Gr-sw66464\d\Demo\grab_cad_brepmfr_testing\inference_csvs_no_a2` | Final output: 28 prediction CSVs, 8,690 face rows, an aggregate `onnx_inference_summary.csv`, and `inference_model_manifest.csv`. |

Full A1+A3 inference passed for 26 graphs. Two assembly instances
(`grab_cad_4_assembly_final_104` and `_111`) each contain 3,182 faces. The
exported model attempted a 20,736,253,952-byte A3 gather and exhausted memory.
Those two were inferred with an A1-only copy of the same ONNX graph by bypassing
and pruning the A3 bias branch. This matches the training policy more closely
than full A3 because training skipped A3 above `max_nodes_for_a3=768`. The
temporary fallback ONNX copies were removed after inference; the manifest marks
the two affected outputs. Both large graphs predicted all 3,182 faces as Stock.

Aggregate predictions: 7,586 Stock, 220 Thread, and 884 Text. There is no ground
truth in these JSONs, so these numbers do not measure correctness. Confidence
does not by itself support a safe "low confidence -> Stock" rule: predicted Text
has median confidence 0.9901; only 21.15% of Text predictions are below 0.80,
while 49.77% are below 0.99. If the manually observed stock-as-Text errors are
among the high-confidence half, thresholding cannot remove them.

Recommended validation path: label a representative real-world calibration set,
join ground truth to these per-face probabilities, and sweep class-specific
acceptance thresholds for Thread and Text. Report precision/recall and the
number of whole parts with zero face errors at every threshold. Add connected
component/topology and CAD-geometry checks as a second-stage validator, and
abstain for manual review when a strict guarantee is required. A forced
three-class prediction cannot guarantee 100% on unseen domain-shifted parts.

## 0. Latest update (2026-07-21) — A1+A3/no_a2 ONNX inference audit

The new fine-tuned model is not compatible with the old seven-input lite inference
contract. Production inference must use `no_a2` PyG graphs and the A1+A3 ONNX runner:

```text
raw B-rep JSON
  -> json_to_brepmfr_pyg_optimized.py --inference_profile no_a2
  -> PyG graph with base node/edge tensors + spatial_pos (A1) + edge_path (A3)
  -> run_onnx_a1_a3_inference.py
  -> brepmfr_a1_a3.onnx (16 inputs; no d2_distance/angle_distance)
  -> per-face Stock/Thread/Text probabilities and CSVs
```

Files audited and their interactions:

| File | Role and audit result |
|------|-----------------------|
| `standalone_scripts/run_onnx_a1_a3_inference.py` | Correctly rejects lite graphs, creates the 16 required tensors, converts compact stored A1/A3 indices to int64, pads/truncates A3 to 16 hops, and consumes the ONNX model's probability output without applying a second softmax. Its default model and three-class map match the current package. |
| `migration_to_c++/migration_to_c/no_a2_72k_epoch50_onnx/brepmfr_a1_a3.onnx` | Runtime inspection confirms 16 inputs: five face inputs, six edge inputs, `attn_bias`, `spatial_pos`, `edge_path`, and two masks. Output is `probabilities[total_nodes,3]`. |
| `migration_to_c++/migration_to_c/no_a2_72k_epoch50_onnx/model_config.json` | Declares `inference_profile=no_a2`, 16-hop A3 paths, spatial cutoff 32, three classes, and the architecture used by the exported checkpoint. |
| `migration_to_c++/migration_to_c/no_a2_72k_epoch50_onnx/label_map.json` | Confirms `0=Stock`, `1=Thread`, `2=Text`; the mapping is unchanged from the lite three-class model. |
| `scripts/inference/json_to_brepmfr_pyg_optimized.py` | Must now be called with `--inference_profile no_a2`, not `lite`. It creates `spatial_pos` and `edge_path` and deliberately omits A2 histograms. |
| `data/collator.py` | Defines the training/PyTorch preprocessing reference. A checked 368-face real graph produced tensors exactly equal to the standalone runner for all 16 inputs. It additionally masks face pairs whose raw distance is at least `spatial_pos_max=32`; the standalone runner currently omits that mask, which can change results for disconnected or long-diameter graphs. |
| `scripts/threads/train_a1_a3_from_lite.ps1` | Fine-tuning used `max_nodes_for_a3=768`: graphs above 768 faces trained with A1 but without A3. |
| `migration_to_c++/migration_to_c/export_a1_a3_onnx.py` | Forces `max_nodes_for_a3=None` during export, so the ONNX graph always executes A3. This differs from training for graphs above 768 faces and can consume substantial `O(N² × 16)` memory. A deployment decision is required: constrain inference graph size, preserve the 768-face A3 policy in a compatible export/runner, or deliberately accept full-A3 inference. |
| `test_a1_a3_onnx_out/*` | Existing real-graph smoke evidence: one 368-face no_a2 graph passed, produced 15 Stock and 353 Text predictions, and matched all available labels. This proves execution, not broad PyTorch-vs-ONNX parity. |

Conclusion: the runner is correct for ordinary connected graphs within the A3 training
cap, but it is not yet a perfect production-parity implementation for all graph sizes.
The distance-32 attention mask should be mirrored, and the A3 behavior above 768 faces
should be made explicit before deployment.

### C++ integration diagnosis: `node_data` rank mismatch

Observed ONNX Runtime error:

```text
Invalid rank for input: node_data Got: 5 Expected: 4
```

The ONNX model deliberately uses flattened real nodes for its convolutional face
encoder. C++ supplied `node_data` as `[1,N,5,5,7]`; the exported contract is
`[N,5,5,7]`. No re-export is required. The contiguous data buffer is identical, so
the C++ `Ort::Value::CreateTensor<float>` shape must simply omit the leading batch
dimension. Only dense graph-structure tensors and masks carry an explicit batch
dimension:

```text
node_data         float32 [N,5,5,7]
face_area         float32 [N]
face_type         int64   [N]
face_loop         int64   [N]
in_degree         int64   [N]
edge_data         float32 [E,5,7]
edge_type         int64   [E]
edge_len          float32 [E]
edge_ang          float32 [E]
edge_conv         int64   [E]
edge_index        int64   [2,E]
attn_bias         float32 [1,N+1,N+1]
spatial_pos       int64   [1,N,N]
edge_path         int64   [1,N,N,16]
padding_mask      bool    [1,N]
edge_padding_mask bool    [1,E]
probabilities     float32 [N,3]
```

## 0. Latest update (2026-07-20) — delta raw-JSON sync and clean no_a2 scratch run

New state: `D:\thread_and_text\no_a2\pyg` has ~48k A1+A3 graphs, while `root_json` has ~70k JSONs. The additional ~22k JSONs have raw SolidWorks labels and no lite `.pt`, so the lite-upgrade path cannot process them. The 48k dataset must remain untouched; the expanded corpus is built under `D:\thread_and_text\no_a2_large`.

**Implemented complete delta workflow:**

| File | Role and interaction |
|------|----------------------|
| `scripts/threads/remap_missing_no_a2_json_labels.py` | Selects only JSON stems absent from `no_a2/pyg`, adds identity mappings for already-normalized targets, strictly audits unknown labels, then remaps `-10/-1/0→0`, `70→1`, `101→2`. Existing processed JSONs are untouched. |
| `scripts/threads/prepare_no_a2_scratch_delta.ps1` | Two-phase orchestrator. Reads `BaseNoA2Root=no_a2`, targets `OutputRoot=no_a2_large`, and never writes the base. Without `-ApplyLabelRemap`, performs a no-write strict audit. With the switch, hard-link-seeds the new tree by default, remaps root+ABC deltas, converts, verifies coverage, rebuilds splits, recomputes weights, validates tensors, and quarantines unusable graphs. |
| `scripts/inference/json_to_brepmfr_pyg_optimized.py` | Called without `--label_out_dir`, making `.pt` existence the sole skip condition. It scans all root JSONs cheaply, skips the ~48k files seeded into `no_a2_large`, and directly converts only missing stems using profile `no_a2`. |
| `scripts/threads/make_random_splits.py` | Rebuilds train/val/test over the complete corpus; STEP variants remain atomic and the optional ABC ≥80% train quota remains available. |
| `scripts/training/compute_class_weights.py` | Recomputes weights from the new train split because adding ~22k graphs changes class frequencies. |
| `scripts/threads/validate_a1_a3_finetune_data.py` | Used without a lite reference because new JSON-only samples have no lite counterpart; validates split coverage, no_a2 tensors, and that every embedded label is in `[0, 2]`. |
| `scripts/threads/train_no_a2_from_scratch.ps1` | Defaults to `no_a2_large` and its separately named class weights. Starts a unique 100-epoch run with no checkpoint flags, A1/A3 active at scale 1 from epoch 0, equal `0.002` LRs, warmup, length buckets, and A3 cap 768. |
| `scripts/threads/README_thread_text.md` | Section 7 contains the exact dry-run, apply, and from-scratch training commands. |

This differs from checkpoint recovery: splits and class weights are regenerated, no lite checkpoint is loaded, and no A1/A3 ramp is needed.

**Progress visibility fix:** all long-running PowerShell workflow calls now use `conda run --no-capture-output`, so tqdm/indexing/training output streams immediately instead of appearing frozen until the child process exits. The delta remapper also prints before indexing the ~70k JSON and ~48k `.pt` directory entries.

**Disk-full/remap-speed fix:** a copy-seeded `no_a2_large` began failing at `torch.save` with `PyTorchFileWriter ... cannot be opened` after many successful writes, consistent with the destination drive exhausting free space. The scratch preparer now defaults to hard-link seeding, supports `-ResetOutput` for deleting only the partial expanded tree, probes output writability, prints free space, and refuses conversion below `-MinFreeGB 20`. The JSON converter itself now aborts after three consecutive output-open failures and reports available space instead of printing thousands of identical failures. Delta label audit/remap is file-parallel (`-RemapWorkers`, default 8), uses `orjson` when available, performs atomic rewrites, and avoids a redundant pre-write scan after the orchestrator's strict audit. Root and ABC JSON directories are now both remapped, converted, and coverage-checked; ABC was previously used only during split generation.

**Late validation recovery:** the expanded 72,223-graph run completed conversion, split generation, ABC allocation, and class-weight calculation but exposed legacy/unlabeled graphs only during final validation. The validator previously stopped after 20 errors, which could force repeated full runs and made its validated-count message misleading. It now always scans the complete split, reports the true valid/invalid totals, and supports `--quarantine-invalid`: all unusable `.pt` files are moved out of `pyg`, their stems are atomically removed from train/val/test and the ABC manifest, and a JSON report records every reason. Full scratch preparation enables this automatically; an already completed run can execute the validator once with this flag and proceed without repeating prior stages.

**Cross-checkout compatibility fix:** training was launched from `Desktop\BrepMFR\brepmfr_pyg\BrepMFR` while development changes lived under `Desktop\BrepMFR_PyG\BrepMFR_PyG`. Its updated `segmentation.py` passed `max_nodes_for_a3` into a stale `CADSynth`, causing an immediate constructor error before any epoch. `scripts/threads/sync_a1_a3_training_code.ps1` now copies the complete coordinated set (entry point, dataset/collator, model/encoder/bias layer, and training wrapper), backs up target versions, compiles all synchronized Python files, and verifies required A1/A3 compatibility tokens.

**AMP dtype fix:** after sync, sanity-check crashed in `GraphAttnBias` with `Index put requires the source and destination dtypes match, got Float for the destination and Half for the source` when expanding A3 edge features under `--precision 16-mixed`. The bias layer now allocates edge buffers via `new_zeros`, casts the indexed write and A1/A2/A3 additions to the destination dtype, and casts the float32 `a1_a3_scale` buffer before multiplying half activations. Re-run the sync script before restarting training.

**Training throughput fix:** the first successful run exposed the real bottleneck: the legacy sampler assigned every graph above 300 faces to batch size 1, yielding 13,087 singleton large-graph batches and 16,995 total training batches per epoch. `LengthBucketBatchSampler` now supports a padded quadratic-cost budget (`batch_size × max_faces²`) and packs size-local graphs greedily while keeping the A3-enabled (`<= max_nodes_for_a3`) and A3-capped groups separate. Dense A1/A3 index tensors remain int32 through collation and host-to-device transfer, halving their traffic versus int64. The eight encoder layers now use PyTorch fused scaled-dot-product attention whenever weights are not requested; a numerical parity test against the legacy attention path showed maximum absolute error `1.79e-7`. The scratch wrapper defaults to a 4,000,000 node² budget, at most 64 graphs per batch, no redundant gradient accumulation, two DataLoader workers with prefetch two and pinned memory, TF32, no sanity-validation pass, a shorter 1,000-step warmup, and full validation every two epochs. Windows users can set workers to zero if commit memory or process spawning becomes unstable. This preserves full graphs and A1/A3 behavior while targeting step-count, transfer, and kernel-launch overhead—the dominant issues that Cython cannot improve in GPU attention training.

## 0a. Prior update (2026-07-20) — fast A1+A3 dataset build (lite upgrade)

Building `no_a2` (A1+A3) by re-running JSON→PyG was impractically slow (~15s/file → days for ~40–48k graphs). Root causes:

1. **Torch cell-write BFS**: `_shortest_paths_from_adj_serial` wrote each hop into a `torch.Tensor` element-by-element (~60–85s for N≈700–850). A NumPy rewrite of the same algorithm is bit-identical and ~50–80× faster (~0.7–1.1s).
2. **Per-graph ProcessPool**: `--shortest_path_workers 8` spawned/destroyed eight Windows processes for every file. That dominates wall time for small/medium graphs and fights the JSON→`.pt` serial loop.
3. **Unnecessary JSON work**: lite `.pt` files already store UV features, labels, and `edge_index`. Only `spatial_pos` + `edge_path` are missing for `no_a2`.

**New fast path (target: complete ~48k graphs in well under 2 hours):**

| File | Role |
|------|------|
| `scripts/threads/upgrade_lite_pt_to_no_a2.py` | Loads each lite `.pt`, runs NumPy A1/A3 BFS, writes `no_a2` graphs with `has_a1/has_a3`, copies split lists. Persistent **file-level** process pool (`--file-workers`, default ≤12). BLAS/OMP threads forced to 1 per worker to avoid OpenBLAS RAM blow-ups. Resume-safe (skips existing outputs). |
| `scripts/threads/prepare_a1_a3_finetune.ps1` | Defaults to the lite-upgrade path when `LiteRoot\pyg` exists (`-FileWorkers`). Pass `-FromJson` to force the old JSON converter. Then runs validation. |
| `scripts/inference/json_to_brepmfr_pyg_optimized.py` | Serial A1/A3 BFS now NumPy-backed (helps any JSON `no_a2`/`full` conversion too). Per-source ProcessPool threshold raised to N≥512; prefer file-level workers for corpora. |
| `scripts/threads/validate_a1_a3_finetune_data.py` | Unchanged: A1/A3 flags, shapes, label/topology parity vs lite. |

**Measured on this corpus (lite split lists, Z: network share):**

- Face counts: min=3, p50=86, mean≈181, p90≈422, max≈1794 (sample of 200).
- Large-graph BFS: N=788 torch≈77s vs NumPy≈0.93s (match=True).
- Smoke upgrade of 8 graphs + lite label/`edge_index` parity: OK.
- Estimated BFS-only with 12 file workers: ~0.14h; wall time will be higher due to network `torch.load`/`torch.save`, but 2 hours remains the design target. Use local SSD staging if Z: I/O saturates.

**How the pieces interact for A1+A3 fine-tune:**

```
lite/pyg/*.pt  --upgrade_lite_pt_to_no_a2.py-->  no_a2/pyg/*.pt
lite/{train,val,test}.txt  --copy-->  no_a2/
no_a2/  --validate_a1_a3_finetune_data.py-->  OK
no_a2/ + lite best.ckpt  --train_a1_a3_from_lite.ps1-->  fine-tuned A1+A3 model
```

Main system goal unchanged: recover structural A1/A3 attention bias on top of a trained lite Thread/Text checkpoint without rebuilding labels or splits.

## 0a. Prior update (2026-07-20) — salvage lite checkpoint with gradual A1+A3 fine-tuning

Thread+text README converted with `--inference_profile lite` (skips **A1+A2+A3**).
User intended A1/A3 proximities (and/or full A2):

| Profile | A1 `spatial_pos` | A2 `d2`/`angle` | A3 `edge_path` |
|---------|------------------|-----------------|----------------|
| `lite` (what was used) | no | no | no |
| `no_a2` (A1+A3) | yes | no | yes |
| `full` (true A2 too) | yes | yes | yes |

The 39-epoch lite checkpoint is reusable: profile selection changes input tensors, not model shape. Its `GraphAttnBias` A1/A3 parameters exist in the checkpoint but remained effectively at initialization because lite batches never called those branches.

**Implemented recovery workflow:**

| File | Role and interaction |
|------|----------------------|
| `scripts/threads/prepare_a1_a3_finetune.ps1` | Fast path upgrades lite `.pt` → `no_a2/pyg` (or JSON fallback with `-FromJson`), copies split lists, validates. Never overwrites `lite/`. |
| `scripts/threads/upgrade_lite_pt_to_no_a2.py` | File-parallel NumPy A1/A3 upgrade from lite graphs. |
| `scripts/threads/validate_a1_a3_finetune_data.py` | Confirms every split stem exists, flags are A1=true/A2=false/A3=true, pairwise tensor shapes are valid, and labels/topology match each lite reference graph. Reports graphs above the proposed A3 memory cap. |
| `scripts/threads/train_a1_a3_from_lite.ps1` | Starts a new `--pre_train` run from the lite checkpoint with backbone LR `1e-4`, graph-bias LR `1e-3`, 1000-step warmup, five-epoch bias ramp, and A3 cap 768. |
| `segmentation.py` | Exposes learning-rate, optimizer-warmup, A1/A3-ramp, and A3-cap CLI controls; rejects mixing `--pre_train` with exact `--resume_from_checkpoint`; passes the A3 cap to datasets and model. |
| `models/brepseg_model.py` | Selectively loads the lite checkpoint as before, controls the A1/A3 scale by epoch, uses separate AdamW parameter groups, applies warmup before the first update, and preserves backward checkpoint compatibility for the new scale buffer. |
| `models/modules/layers/brep_encoder_layer.py` | Stores the active A1/A3 scale in the checkpoint and multiplies only A1 shortest-path/graph-token and A3 multi-hop edge contributions. A2 behavior is unchanged. |
| `data/dataset.py` + `data/collator.py` | Skip A3 before allocating its dense padded tensor when a batch exceeds `max_nodes_for_a3`; A1 still reaches `GraphAttnBias`. This protects both host and GPU memory. |
| `scripts/threads/README_thread_text.md` | Documents preparation, fine-tuning, evaluation, and the distinction between `--pre_train` and `--resume_from_checkpoint`. |

**Data invariants:** reuse the original `train.txt`, `val.txt`, and `test.txt`. Class weights do not need recomputation because graph labels and split membership are unchanged. Use `no_a2` for A1+A3; use `full` only when real `face_pairs` A2 histograms are desired and available.

## 0b. Prior update (2026-07-20) — diagnosing weak Text class vs Thread/Stock

User report: training strong on stock+thread, weak on text. Primary validation path (not model architecture first):

1. `repair_json_face_labels.py --dry-run --fail-on-unknown` on `root_json` and `abc_jsons` with `remap_maps/thread_text_sw_to_brep.json` (`-10/-1/0→0`, `70→1`, `101→2`).
2. `count_thread_label_distribution.py` separately on root JSON, ABC JSON, and `lite/pyg` with `--group "0:stock,1:thread,2:text"`.
3. Confirm ABC is almost all stock (if ABC has many `101`/`2`, labeling is wrong).
4. Inspect class-weights JSON + train-split face counts for text rarity / weight collapse.
5. Confusion from `segmentation.py test` / TensorBoard per-class metrics: text→stock collapse vs never predicting text.

## 0b. Prior update (2026-07-20) — Pillow `_imaging` DLL failure in `brep_mfr_pyg`

**Symptom:** `python segmentation.py test ...` fails at `from pytorch_lightning import Trainer` → matplotlib → `from PIL import Image` → `ImportError: DLL load failed while importing _imaging`.

**Cause chain:**
1. Mixed/corrupt Pillow leftovers (pip 9.4 vs conda metadata drift) in `envs/brep_mfr_pyg`.
2. After clean conda-forge Pillow 9.2, `_imaging.pyd` still failed because it loads `tiff.dll`, which requires **`libdeflate.dll`**.
3. Current `libdeflate` 1.25 (conda-forge) only ships **`deflate.dll`** (rename), so Windows cannot resolve the old import name.

**Fix applied:**
- Removed mixed Pillow; installed `conda-forge::pillow=9.2.0`.
- Copied `Library\bin\deflate.dll` → `Library\bin\libdeflate.dll` so `tiff.dll` loads.
- Verified: `from PIL import Image` and `from pytorch_lightning import Trainer` succeed.

**Files involved in this failure path:** `segmentation.py` → Lightning → torchmetrics → matplotlib → Pillow (`PIL._imaging`).

## 0b. Prior update (2026-07-20) — test precision/recall + `standalone_scripts/`

**Test metrics** (`models/brepseg_model.py`, `models/transfer_model.py` `on_test_epoch_end`):
- For each of the `num_classes` training classes, logs **precision** and **recall**
- Named aliases when classes are 0/1/2: `test_Stock_*`, `test_Thread_*`, `test_Text_*`
- Also `test_class_{i}_precision` / `_recall` / `_acc`, plus `macro_precision` / `macro_recall`
- Printed as a table at end of `segmentation.py test`

**Folder rename:** former `migration_to_c/` is now **`standalone_scripts/`** (ONNX demos, Jenkins, allowlists, STEP/JSON housekeeping, VBA, etc.). Run as `python standalone_scripts/<script>.py`.

## 0b. Prior update (2026-07-19) — prune leftover pyg before splits

Conversion wrote ~48k graphs but `Z:\thread_and_text\lite\pyg` still had **136,753** `.pt` (old leftovers). Splits used all of them → class-weight load hit corrupt `PytorchStreamReader ... data/2`.

Fix: `scripts/threads/prune_pyg_to_json_stems.py` (keep only stems in `root_json` + `abc_json`), regenerate splits, then `compute_class_weights.py --skip-bad`.

## 0b. Prior update (2026-07-19) — training split leakage fix + abc_jsons + label -10

Training pipeline changes (`scripts/threads/` + converter):

| Change | File(s) | Behavior |
|--------|---------|----------|
| STEP-aware train/val/test | `make_random_splits.py` | Groups by `..._step_NNN`; all variants of one STEP stay in one split |
| ABC ≥80% in train | same + `--abc-json-dir` | ABC stems measured from that folder; leftover ABC → val/test only |
| Dual JSON → PyG | `json_to_brepmfr_pyg_optimized.py --abc_json_dir` | Converts `root_json` + `abc_jsons` into same `pyg`/`label`; writes `abc_stems.txt` |
| Label `-10` → stock | `remap_maps/thread_text_sw_to_brep.json` | `-10` maps to `0` (with `-1`/`0`) |

Docs: `scripts/threads/README_thread_text.md`, `post_thread_text_pyg_export.ps1`.

## 0b. Prior update (2026-07-19) — delete NON-allowlist JSONs (keep no-Thread/Text)

**Script:** `standalone_scripts/delete_jsons_on_allowlist.py` (logic inverted)

- **KEEP** JSONs whose STEP key is on the allowlist (~9464 — no confident Thread/Text)
- **DELETE** JSONs whose STEP key is NOT on the allowlist (~6785)

Previous run deleted the wrong set; restore `E:\jsons_from_all_machines` then:

```bash
python standalone_scripts/delete_jsons_on_allowlist.py --dry-run
python standalone_scripts/delete_jsons_on_allowlist.py
```

## 0b. Prior update (2026-07-19) — post-macro cleanup → infer → allowlist → match

After SolidWorks batch finishes, run in order:
1. `delete_duplicate_jsons.py` (clean `C:\jsons`)
2. `run_onnx_json_batch_inference.py` (CSVs + Stage-2 filter)
3. `export_step_allowlist_from_inference.py` (refresh `allowed_step_keys_p1/p2/p3.txt`)
4. `match_allowlist_vs_jsons_folder.py` (vs `E:\jsons_from_all_machines`)

## 0b. Prior update (2026-07-19) — copy STEPs not in allowlist

**Script:** `standalone_scripts/copy_steps_not_in_allowlist.py`  
Takes JSON keys in `E:\jsons_from_all_machines` that are **not** in the no-Thread/Text allowlist, finds STEPs in `\\GR-SW65551\abc_steps`, copies to `C:\abc_steps_not_in_allowlist`.

Live: 661 non-allowlist keys → **151** still in abc_steps (copied); 510 already gone.

## 0b. Prior update (2026-07-19) — match allowlist vs E:\jsons_from_all_machines

**Script:** `standalone_scripts/match_allowlist_vs_jsons_folder.py`  
Loads `allowed_step_keys_p1/p2/p3.txt` (2688 keys), matches `E:\jsons_from_all_machines` on `..._step_NNN` only.

Live result: **188 / 2688** allowlist keys present; **4331 / 4732** JSON files belong to those keys (many variants per key).

## 0b. Prior update (2026-07-19) — STEP open timeout via watchdog

VBA cannot cancel `LoadFile4` mid-call. `BatchJsonExport.vba` now starts `C:\jsons\Watchdog-StepOpen.ps1` before each open; if `in_progress.txt` remains after `OPEN_TIMEOUT_SEC` (default 60), watchdog appends the file to `skip_list.txt` and kills `SLDWORKS.exe`. Re-run the macro to continue.

## 0b. Prior update (2026-07-18) — Jenkins allowlist seed (no UNC)

Agents cannot read `\\LP76-RZA2-DSA\jsons\...`. Updated `Jenkinsfile.filter_abc_steps_no_thread_text` to **embed** `allowed_step_keys.txt` and `writeFile` it into each agent `WORKSPACE` before filtering.

Regenerate with: `python standalone_scripts/_gen_filter_jenkinsfile.py`

## 0b. Prior update (2026-07-18) — Jenkinsfile comment fix

`Jenkinsfile.filter_abc_steps_no_thread_text` header comments must use `//`, not `#` (Groovy/Pipeline). `#` caused `expecting '!', found ' '`.

## 0b. Prior update (2026-07-18) — filter abc_steps by no-Thread/Text allowlist

After Stage-2 flagged ~2677 JSONs with no confident Thread/Text:

| File | Role |
|------|------|
| `export_step_allowlist_from_inference.py` | Builds `C:\jsons\inference\allowed_step_keys.txt` (unique `..._step_NNN` keys) |
| `filter_abc_steps_by_allowlist.py` | Copy matching STEPs → `abc_steps_filtered` (local or remote) |
| `Filter-AbcStepsByAllowlist.ps1` | Same filter in PowerShell for agents |
| `Jenkinsfile.filter_abc_steps_no_thread_text` | Parallel filter on 10 old VMs |

**Jenkins:** do **not** embed filenames in the Groovy. Point `ALLOWLIST_UNC` at the shared `allowed_step_keys.txt`.

## 0b. Prior update (2026-07-18) — two-stage JSON ONNX batch inference

**Script:** `standalone_scripts/run_onnx_json_batch_inference.py`

| Stage | Action |
|-------|--------|
| 1 | Convert each `C:\jsons\*.json` → lite graph → `BrepMFR_lite_onnx_pyg_demo_v2/brepmfr_lite.onnx` → CSV in `C:\jsons\inference\` |
| 2 | Flag JSONs with no face where `prob_Thread` or `prob_Text` > 0.80 → `no_confident_thread_or_text.csv` / `.txt` |

Uses `scripts/inference/json_to_brepmfr_pyg.build_pyg_from_json_path(..., inference_profile="lite")` and the same 7-input lite ORT contract as the v2 demo.

## 0b. Prior update (2026-07-18) — delete leftover SLDPRT; STEP delete != 4k

Deleted leftover `C:\jsons\*.SLDPRT` temps. JSON count ≈ unique STEP keys (~4k).

Running `delete_step_files.py` (JSON=`C:\jsons` / `\\LP76-RZA2-DSA\jsons`, STEP=`abc_steps`) only deletes STEPs whose key **still exists** in `abc_steps`. Keys whose STEP was already removed earlier do not reduce the STEP count further — so “4k unique JSONs” ≠ “exactly 4k STEPs deleted” unless all 4k keys still have a STEP present.

## 0b. Prior update (2026-07-18) — why Explorer shows ~6k but scripts ~4k

`C:\jsons` contains **both** `.json` and leftover `.SLDPRT` temp parts from VBA `SaveAs`.

| What | Count (manual scan) |
|------|--------------------:|
| Total files Explorer sees | **6626** |
| `.json` only | **4015** |
| `.SLDPRT` leftover temps | **2611** |

Scripts only count `*.json`, so ~4k is correct. Naming pattern for all 4015 JSONs:
`{8digitId}_{32hexHash}_step_{NNN}_{bodyId}.json` (e.g. `..._step_000_101.json`).

**Multi-body JSON duplicates right now: 0.** Every STEP key has exactly one JSON (4015 unique keys / 4015 JSONs). Body ids still vary (`_101` mostly, some `_102`+ when that was the only body kept). The ~6k vs ~4k gap is almost entirely junk `.SLDPRT`, not duplicate JSONs.

## 0b. Prior update (2026-07-17) — one JSON per STEP (macro + cleanup)

**VBA:** `standalone_scripts/BatchJsonExport.vba`
- Still calls `BaselineOutputCmd 100040` (plugin emits all bodies).
- After each STEP: `KeepOnlyOneBodyJson` keeps lex-first `stem_*.json` (usually `*_101`) and deletes other bodies.
- Skip if any `stem_*.json` already exists.
- Output folder: `C:\jsons`.

**Python cleanup (existing folder):** `standalone_scripts/delete_duplicate_jsons.py` → `C:\jsons`, keep one per `..._step_NNN` key (~1446).

## 0b. Prior update (2026-07-17) — Training treats each body JSON as one sample

**Question:** For Stage-1 training, do we need all body JSONs (`*_101`, `*_102`, …) from a multi-body STEP, or is one body enough?

**Answer:** Training never merges bodies. Each JSON → one `.pt` → one `CADSynth` item. One body per STEP is enough to train; keep all bodies only if you want every solid as extra face-graph samples.

| Component | Path | Role |
|-----------|------|------|
| Ingest | `scripts/inference/json_to_brepmfr_pyg.py` | `glob("*.json")`; `convert_one_json` writes `{stem}.pt` (1:1) |
| Dataset | `data/dataset.py` (`CADSynth`) | Matches `train.txt` stems to `*[0-9].pt`; `__getitem__` = one graph |
| Splits | `scripts/threads/make_random_splits.py` | One stem per line (full name including `_101` / `_102`) |
| Multi-body tooling | `tools/bins/missing_files_checker_mfcadpp.py` | Counts IDs with >1 JSON as multi-body; does not merge |
| Housekeeping (drops bodies) | `standalone_scripts/delete_duplicate_jsons.py` | Keeps lex-first per STEP key (usually `*_101`) |

**Labels/features:** Nodes = faces of that one solid; `label_feature[i] = face[i].label`; UV grids on `node_data` / `edge_data`. No assembly-level graph.

**Naming note:** Trailing `_N` is also used as face-count hint in `data/length_bucket_batch_sampler.py` and as `data_id` in ingest; for MFCAD++ multi-body, `_101`/`_102` are body suffixes (different solids).

## 0b. Prior update (2026-07-17) — JSON `_101/_102` are bodies, not duplicates

Investigation of `C:\jsons` vs VBA `BaselineOutputCmd 100040, outFolder & "|5"`:

- One STEP often yields many JSONs: `..._step_000_101.json`, `..._step_000_102.json`, …
- These are **per-body / per-solid exports** from multi-body STEP parts, not accidental duplicates.
- Across all multi-key groups, variant files **differ in size and face count** (e.g. 14 faces vs 270 faces).
- Exact byte-identical copies within a key are rare/none for true dups; do **not** delete by unique STEP key if you need all solids.

Script: `standalone_scripts/count_unique_json_vs_steps.py` counts unique STEP keys; “unique keys << file count” is expected for multi-body parts.

## 0b. Prior update (2026-07-17) — unique JSON vs STEP count report

**Script:** `standalone_scripts/count_unique_json_vs_steps.py`

Compares unique `..._step_NNN` keys from `C:\jsons` against `\\GR-SW65551\abc_steps` and reports matches / missing on each side.

## 0b. Prior update (2026-07-17) — STEP cleanup from `\\GR-SW26859\abc` JSONs

**Script:** `standalone_scripts/delete_step_files_from_abc_jsons.py`

| Role | Path |
|------|------|
| JSON outputs (variants) | `\\GR-SW26859\abc` (~6987 `*.json`) |
| STEP inputs | `\\GR-SW65551\abc_steps` |

JSON names include many suffixes for the same STEP (`both_v8`, `engrave`, `thread_v5`, body ids). Matching key strips those and uses `..._step_NNN` only, so one STEP is deleted even if many JSON variants exist.

Related earlier scripts: `delete_step_files.py` (vs `\\LP76-RZA2-DSA\jsons`), `delete_duplicate_jsons.py`.

## 0b. Prior update (2026-07-17) — STEP cleanup + JSON dedupe

**Scripts:**
- `standalone_scripts/delete_step_files.py` — delete STEPs that already have JSON outputs
- `standalone_scripts/delete_duplicate_jsons.py` — keep 1 JSON per STEP key, delete variant suffixes

| Role | Path |
|------|------|
| JSON outputs | `\\LP76-RZA2-DSA\jsons` (~9065 → keep ~1089, delete ~7976 dups) |
| STEP inputs | `\\GR-SW65551\abc_steps` (~14705 `*.step` remaining) |

**Goals:** (1) remove processed STEPs; (2) dedupe JSONs by key `..._step_NNN`, keep lexicographically first (usually `*_101.json`).

**Caveat:** `_101`/`_102` may be different bodies from a multi-body STEP, not byte-identical copies.

**Interactions:** SolidWorks / batch export writes JSONs into the share; these scripts are housekeeping. Independent of the ONNX/PyG model path.

## 0b. Prior update (2026-07-16) — 100K model ONNX share package v2

A second developer-facing ONNX package was created from the new 100K / ~80-epoch
Lightning checkpoint, mirroring the previous `BrepMFR_lite_onnx_pyg_demo` layout.

| Item | Path / value |
|------|----------------|
| Source checkpoint | `standalone_scripts/100K_MODEL_80EPOCH/best-v2.ckpt` |
| Run name | `thread_text_new_100k_run` |
| Epoch / step | 79 / 1,188,560 |
| Classes | 3 (Stock / Thread / Text), lite profile |
| Export script | `standalone_scripts/model_conversion_onnx.py` |
| Intermediate export dir | `standalone_scripts/exported_v2/` |
| Share folder | `standalone_scripts/BrepMFR_lite_onnx_pyg_demo_v2/` |
| Share zip | `standalone_scripts/BrepMFR_lite_onnx_pyg_demo_v2.zip` (~18.4 MB) |

**Share package contents (flat zip, same contract as v1 demo):**

- `brepmfr_lite.onnx` — ONNX graph (7 lite inputs → `logits` `[N,3]`)
- `run_onnx_pyg_inference.py` — self-contained ORT runner for `.pt` graphs
- `label_map.json` — `{0:Stock, 1:Thread, 2:Text}`
- `model_config.json` — input dtypes/shapes + checkpoint metadata
- `README.md` / `requirements.txt` — install and usage notes

**Export validation:** PyTorch sanity checks passed; ORT vs PyTorch on 5 dummy
graph sizes — all labels matched, `max_abs_diff ≈ 8.94e-07`.

**How pieces interact for this handoff:**

```text
100K_MODEL_80EPOCH/best-v2.ckpt
        │  model_conversion_onnx.py
        v
exported_v2/{brepmfr_lite.onnx, model_config.json, label_map.json}
        │  copied into share folder
        v
BrepMFR_lite_onnx_pyg_demo_v2/  (+ runner, README, requirements)
        │  Compress-Archive
        v
BrepMFR_lite_onnx_pyg_demo_v2.zip  --> give to other developer
        │  they run run_onnx_pyg_inference.py --input <lite .pt>
        v
per-face CSV predictions
```

Previous (v1) package remains at `BrepMFR_lite_onnx_pyg_demo/` /
`BrepMFR_lite_onnx_pyg_demo.zip` (53k / epoch-64 weights). Do not overwrite it.

## 1. Main goals of the system

BrepMFR_PyG is a graph-neural-network pipeline for **B-rep CAD face segmentation**. The
goal relevant to this tracking file is deployment / inference of the **Stage-1 lite
model** that classifies every face of a CAD part into one of three manufacturing
meaning classes:

| Class id | Display name | Meaning |
|----------|--------------|---------|
| 0 | Stock | bulk / non-feature face |
| 1 | Thread | threaded face |
| 2 | **Text** | embossed / engraved text face (new in v2) |

The ONNX export (`brepmfr_lite.onnx`) accepts only the **lite** PyG layout —
`node_data`, `face_area`, `face_type`, `face_loop`, `in_degree`, `attn_bias`,
`padding_mask`. A1/A2/A3 and edge tensors must be absent (they were optimized out
at export time). The ONNX wrapper was exported for **one graph per call**, so each
part is inferred separately.

Two runner scripts exist for this ONNX model:

- `standalone_scripts/run_onnx_pyg_inference.py` — original (v1), 2-class era CLI,
  flat `--input <file|dir>` interface.
- `standalone_scripts/run_onnx_pyg_inference_v2.py` — **new**, 3-class
  (Stock/Thread/**Text**) with a dataset-root oriented CLI that matches the
  production PyG inference scripts and supports `test.txt` split lists plus
  optional ground-truth metrics.

## 2. Files in this workflow and how they interact

```text
                    PyG .pt graphs  +  label/<stem>.json sidecars  +  test.txt
                                          (dataset root)
                                                  |
                                                  v
              standalone_scripts/run_onnx_pyg_inference_v2.py
                              (loads)
                                                  |
          +-------------------------------------+-------------------------------------+
          |                                     |                                     |
   exported/brepmfr_lite.onnx        exported/label_map.json (optional)         ORT session
   (the ONNX model)                  (3-class map; v2 also has built-in)        (CPU/CUDA)
                                                  |
                                                  v
                          per-graph CSV  +  onnx_inference_summary.csv
                          (+ confusion_matrix.csv / per_class.csv / summary.md
                            when ground-truth labels are present)
                                  written to --output-dir
```

### 2.1 `standalone_scripts/run_onnx_pyg_inference_v2.py`  *(the file added in this change)*

Self-contained CLI runner (no repo bootstrap needed). Responsibilities:

- **Arg parsing** — `parse_args()`. Two input modes:
  - `--dataset-path <root>` (new): dataset root containing a `pyg/` (or `pug/`)
    sub-folder of `*.pt` graphs, an optional `label/` sidecar folder, and an
    optional `test.txt` split list.
  - `--input <file|dir>` (v1-compatible): a single `.pt` or a flat directory.
  - `--output-dir` defaults to `\\Gr-sw66464\d\brepmfr_sw_inference\csv_inference`
    so the production command is short.
- **Graph resolution** — `resolve_graphs()`. Handles split-list filtering,
  `pug`/`pyg` typo fallback, missing-stem warnings, and `--max-files` capping.
- **Label map** — `load_label_map()`. Uses the built-in 3-class
  `{0:Stock, 1:Thread, 2:Text}` unless `--label-map` points at a JSON file
  (e.g. `exported/label_map.json`).
- **ORT setup** — `select_providers()` (auto prefers CUDA), validates the ONNX
  input contract against `LITE_REQUIRED_INPUTS`.
- **Lite batch construction** — `ensure_lite_graph()` + `make_lite_batch()` +
  `batch_to_ort_feed()`. Reproduce the production collator for one unpadded
  lite graph; cast dtypes to the ONNX contract (float32 / int64 / bool).
- **Prediction writing** — `write_predictions()`. One row per face with
  `face_index`, `predicted_class_id`, `predicted_label`, `confidence`, one
  `prob_<class>` column per class, and (when GT exists) `ground_truth_*` +
  `correct_top1`.
- **Ground-truth resolution** — `resolve_gt()`. Prefers
  `graph.label_feature`; otherwise reads
  `<dataset>/label/<stem>.json` (key `labels`, length = num faces).
- **Metrics** — when any labelled faces are found, writes
  `confusion_matrix.csv`, `per_class.csv` (precision/recall/IoU), and a
  human-readable `summary.md` to `--metrics-dir` (defaults to `--output-dir`).
- **Summary CSV** — always writes `onnx_inference_summary.csv` with one row
  per graph (status PASS/SKIP/FAIL, face count, mean confidence, class counts,
  has_gt).

Exit code is `1` if any graph failed, otherwise `0`.

### 2.2 `standalone_scripts/run_onnx_pyg_inference.py`  *(v1, unchanged)*

The original runner. Same ONNX contract and same lite-batch construction as v2,
but with a flat `--input` interface, no split-list support, and a label map
loaded strictly from `--label-map` (default `exported/label_map.json`). Kept
for backward compatibility smoke tests.

### 2.3 `standalone_scripts/exported/brepmfr_lite.onnx`

The exported ONNX model. Single output named `logits` with semantics
`softmax_probabilities` (see `model_config.json`). Expects exactly the seven
lite inputs listed in `LITE_REQUIRED_INPUTS`.

### 2.4 `standalone_scripts/exported/label_map.json`

```json
{ "0": "Stock", "1": "Thread", "2": "Text" }
```

Already a 3-class map. v2 bakes the same map in as `DEFAULT_LABEL_MAP` so the
script works even without the JSON file, but `--label-map` can still point here
(or to a custom map).

### 2.5 `standalone_scripts/exported/model_config.json`

Documents the export: `num_classes=3`, `inference_profile=lite`, ONNX input
shapes/dtypes, the optimized-out tensors, and parity-validation results
(`label_match 1080/1080`, `max_abs_diff 2.38e-06`). v2 cross-checks the output
class count against `len(label_map)` at runtime.

### 2.6 `standalone_scripts/model_conversion_onnx.py`

The export script that produced `brepmfr_lite.onnx` from a Lightning
checkpoint. Not invoked by the runner; listed here because it defines the
ONNX input contract that v2 enforces in `main()`.

### 2.7 `standalone_scripts/validate_onnx_real_graphs.py` and `validate_lite_dataset.py`

Pre-deployment validation scripts that compare ONNX outputs to the PyTorch
model on real lite graphs. They are the source of the parity numbers in
`model_config.json`. Not used at inference time.

### 2.8 `scripts/threads/run_thread_pyg_inference.py`

Production PyTorch-side batch inference for the **2-class thread** model
(stock=0, thread=1). v2's dataset-root CLI (`--dataset-path`, split list,
sidecar GT, metrics bundle) is modelled on this script so operators can move
between the PyTorch and ONNX runners without learning a new interface.

### 2.9 `scripts/inference/run_pyg_inference.py`

The generic PyG inference module that `run_thread_pyg_inference.py` imports
(`load_brepseg_for_inference`, `predict_probs_per_node`, `_resolve_gt`,
`_load_sidecar_labels`, `_num_faces`). v2 re-implements the small helper
subset it needs (`resolve_gt`, the sidecar format) inline so it stays
self-contained and does not require the repo bootstrap.

### 2.10 `scripts/threads/README_thread_text.md`

Documents the 3-class Thread+Text training pipeline (label remap, JSON→PyG
lite conversion, splits, Stage-1 training, subgraph training). The class
mapping `Stock=0 / Thread=70→1 / Text(emboss)=101→2` is the canonical source
for the 3-class label map that v2 uses.

### 2.11 `test_unseen_data.py`

End-to-end evaluation script that goes from raw JSON → PyG lite → PyTorch
inference, with optional `--test_txt` and `--train_txt` for unseen-part
enforcement. Uses the same 3-class custom name map `{0:Stock, 1:Thread,
2:Text}` that v2 bakes in. Useful for sanity-checking v2's ONNX outputs
against the PyTorch model on the same parts.

## 3. Production deployment layout (this engagement)

```text
\\Gr-sw66464\d\brepmfr_sw_inference\
├── pyg_lite\                     <-- dataset root passed via --dataset-path
│   ├── pyg\   (or pug\)          <-- *.pt graphs (lite profile)
│   ├── label\                    <-- <stem>.json sidecars with {"labels": [...]} (optional)
│   └── test.txt                  <-- split list, one stem per line (optional)
└── csv_inference\                <-- --output-dir (created if missing)
    ├── <stem>_predictions.csv    <-- one per graph
    ├── onnx_inference_summary.csv
    └── (confusion_matrix.csv, per_class.csv, summary.md  when GT exists)
```

## 4. Open items / notes

- v2's `--output-dir` default is hard-coded to the UNC path above for operator
  convenience; pass `--output-dir` explicitly to redirect.
- If the graph sub-folder is literally named `pug` (a known typo on the
  production share), either pass `--pyg-subdir pug` or rely on the automatic
  `pug`↔`pyg` fallback in `resolve_graphs()`.
- The ONNX model is one-graph-per-call, so very large directories are slower
  than the batched PyTorch runner; `--max-files` is provided for smoke tests.
- Ground-truth metrics are only emitted when sidecars (or `label_feature`)
  are present. Without GT the runner still produces per-graph and summary
  CSVs.

## 5. GrabCAD JSON → PyG → GraphML → inference workflow (2026-07-14)

### Goal and completed result

This workflow converts 28 SolidWorks B-rep JSON files into the lite PyG format,
exports topology and scalar CAD attributes for graph visualization, and performs
three-class face inference with `best-v8.ckpt`.

```text
Z:\Demo\grab_cad_brepmfr_testing\jsons\<stem>.json
    │  scripts/inference/json_to_brepmfr_pyg_optimized.py
    v
Z:\Demo\grab_cad_brepmfr_testing\pyg\<stem>.pt
    ├── scripts/visualization/pyg_to_graphml.py
    │       └── graphml_dir\<stem>.graphml
    └── scripts/threads/run_thread_pyg_inference.py + best-v8.ckpt
            └── inference_csvs\<stem>.csv
```

Verified output:

- 28 input JSON files.
- 28 lite PyG files; conversion failures: 0.
- 28 valid GraphML XML files; export failures: 0.
- 28 inference CSV files containing 8,690 face rows in total.
- Every JSON/PyG/GraphML/CSV stem matches, and each CSV row count equals its
  corresponding graph's face count.
- Predicted faces: 2,336 Stock, 5,643 Thread, and 711 Text.
- Input graphs contain `label_feature=-1` for all 8,690 faces, so these are
  prediction-only CSVs with no ground-truth accuracy fields.

### Files and interactions

#### `scripts/inference/json_to_brepmfr_pyg_optimized.py`

Reads each source JSON's `faces` and `edges`, sorts faces by CAD face ID, creates
directed adjacency arcs, and packs face UV grids, face scalars, edge UV samples,
edge scalars, and topology into a `torch_geometric.data.Data` object. This run
used `--inference_profile lite`, so A1 shortest-path, A2 pairwise histogram, and
A3 edge-path tensors are omitted. The resulting `.pt` files retain the tensors
required by the trained lite model and provide `edge_index` for GraphML export.

#### `scripts/visualization/pyg_to_graphml.py`

New batch exporter added for this workflow. It loads each `.pt` on CPU and
constructs a directed `networkx.MultiDiGraph`, preserving every directed PyG
edge, including parallel arcs. Graph-level metadata includes source filename,
face count, edge count, profile, and data ID. Node attributes include face type,
area, loop count, adjacency value, source label, and degree. Edge attributes
include type, length, angle, and convexity. High-dimensional `node_data` and
`edge_data` UV tensors are deliberately omitted because GraphML attributes must
be scalar. NetworkX serializes one `<stem>.graphml` file for Gephi, Cytoscape,
and other GraphML-compatible visualization tools.

#### `scripts/threads/run_thread_pyg_inference.py`

Loads the checkpoint through the generic inference module, batches each lite
graph through the repository collator, runs the BrepSeg encoder and classifier,
and writes one face-level CSV per graph. Its display-name map now includes class
2 as `text`, matching this checkpoint's Stock/Thread/Text training metadata.
CSV rows contain face index, predicted class ID, display name, and top-class
probability. Ground-truth columns are omitted because all source labels are -1.

#### `scripts/inference/run_pyg_inference.py`

Imported dynamically by the thread inference runner. It reconstructs the
`BrepSeg` architecture from Lightning checkpoint hyperparameters, loads the
`brep_encoder`, `attention`, and `classifier` weights, moves the model to CUDA,
and provides the per-node probability forward path. The checkpoint reports
`num_classes=3`, `d_model=512`, `dim_node=256`, 32 attention heads, and 8 encoder
layers.

#### `data/collator.py`

Called by the inference runner to convert one or more PyG objects into the
batched dictionary expected by BrepSeg. For lite graphs it supplies the padded
attention mask while leaving A1/A2/A3 inputs disabled, matching checkpoint
training.

#### `models/brepseg_model.py` and encoder modules

`BrepSeg` is instantiated from checkpoint hyperparameters. Its graph encoder
builds face embeddings, its attention module combines node and graph context,
and its classifier returns per-face probabilities for Stock, Thread, and Text.

#### `C:\Users\RZA2\Downloads\best-v8.ckpt`

Lightning model checkpoint used for inference. Metadata confirms a three-class
Thread+Text lite model trained on `Z:\thread_and_text\lite`. It is consumed
read-only and is not copied into output directories.

#### Generated dataset files

- `jsons\<stem>.json`: raw B-rep source; provides faces, adjacency edges, and UV
  geometry.
- `pyg\<stem>.pt`: compact lite PyG graph; shared input to visualization export
  and model inference.
- `graphml_dir\<stem>.graphml`: visualization topology and scalar attributes,
  one file per PyG graph.
- `inference_csvs\<stem>.csv`: one prediction row per CAD face, one file per PyG
  graph.

### Execution environment

All conversion, export, validation, and inference commands ran through the
`brep_mfr_pyg` Conda environment. CUDA was available and used for checkpoint
inference; graph conversion, GraphML serialization, and final validation ran on
CPU.
