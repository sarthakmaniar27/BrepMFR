# Internship Notes — Rajvi Zala

> **Project:** BrepMFR_PyG — Machining Feature Recognition on B-rep CAD models (Graph Transformer / PyTorch Geometric)
> **Organization:** Dassault Systèmes (SolidWorks group) · **Repo:** `BrepMFR_PyG` (fork of the public BrepMFR work by zhang_linux)
> **Period covered:** June 28 – August 24, 2026 · **Workstations:** dev box `LP76-RZA2-DSA`, training box `GR-SW66464` (Windows, RTX GPU, conda env `brep_mfr_pyg`)
> **Handoff date:** End of internship, August 2026

---

## 1. Project Context — What the System Does

BrepMFR segments every **face of a B-rep solid** into one of N machining feature classes (stock, slots, pockets, holes, chamfer, fillet…). It is a Graph Transformer (Graphormer-style) over the face-adjacency graph:

- **Node features** = per-face UV-grid point clouds `(F, 10×10×7)` sampled from the surface parametrization
- **Edge features** = per-edge UV-grids `(E, 10×7)`
- **Structural attention biases:** A1 = shortest-path spatial position matrix (`spatial_pos`, max hop 32), A2 = pairwise path/angle distance tensors (`d2_distance`, `angle_distance`, dense), A3 = edge-path attention bias (`edges_path`, max len 16)
- Architecture used throughout my experiments: `dim_node=256, d_model=512, n_heads=32, n_layers=8`, dropout 0.2–0.3, trained with PyTorch Lightning, mixed precision (`16-mixed`)
- **Stage 1** = supervised training on synthetic CADSynth data; **Stage 2** = unsupervised domain adaptation to real parts (DANN / IWDAN with GRL)
- **Inference profiles:** `lite` (no A1/A2/A3 — fastest, used for bulk ONNX filtering), `no_a2` (A1+A3), `full` (all three)

### State of the repo when I started (inherited from Sarthak Maniar & Satish Kanjarkar)
| Author | When | What they left |
|---|---|---|
| zhang_linux | Apr 2024 | Original DGL-based BrepMFR code |
| Sarthak Maniar | Apr 13–17, 2026 | Forked repo; UV-Net-style `.bin` tooling (JSON→bin converter, chunking, label remaps for MFCAD++/CADSynth/ToolboxParts); dataset audit scans (~100k-file per-file summaries); `domain_adapt.py` + Stage-2 GRL debugging docs (`markdowns/stage1_stage2_debugging_session.md` et al.) |
| Satish Kanjarkar | May 9 – Jun 24, 2026 | **DGL→PyG migration** (JSON→`.pt` train path), thread-only pipeline, thread+text pipeline scaffolding, HPO tooling |

**Locked reference baseline (May 12, 2026, pre-internship, full-A2 pipeline, reproduces the paper's Table 3):**

| Eval | Face Acc | Mean Per-Class Acc | mIoU |
|---|---|---|---|
| Stage 1 — CADSynth test (274,279 faces) | **98.94 %** | 99.70 % | 96.76 % |
| Stage 1 — MFCAD++ test, source-only (267,964 faces) | 82.28 % | 73.19 % | 66.48 % |
| Stage 2 — MFCAD++ after domain adaptation | **94.51 %** | 90.95 % | 86.74 % |

---

## 2. My Work at a Glance (Timeline)

| Phase | Dates | Theme |
|---|---|---|
| 0. Onboarding | Late June | Data-generation deep dives, SwOrchestrator C# app, STEP sharding |
| 1. Imbalance attack | Jun 29 – Jul 1 | Class imbalance analysis, subgraph sampling, smart batching, leakage tests |
| 2. Customer demo prep | Jul 14 | GrabCAD 28-part inference + GraphML visualization bundle |
| 3. Data ops & infra | Jul 16–19 | ABC ingestion at scale, dedup, allowlists, Jenkins distribution, SolidWorks watchdog |
| 4. Throughput engineering | Jul 19–21 | Compiled BFS kernel, sampler overhaul, fused attention, memory-safe training |
| 5. ABC integration & model lineage | Jul 20–27 | 48k→72k corpus, fine-tunes vs scratch, stock-only holdout |
| 6. Klavuz regression hunt | Jul 25–27 | 4-model comparison, geometry/scale/class-prior diagnostics |
| 7. Final 5-class model & C++/ONNX handoff | Aug 5–23 | Chamfer+Fillet extension, 39k→60k→84k scratch runs, export & validation |

---

## 3. Phase 0 — Onboarding & Data Understanding

**Deliverables:** two long-form internal docs committed in `data_understanding_files/`:
- `PART1_data_generation_deep_dive.md` — why synthetic data is needed; STEP/B-rep fundamentals; the 25 machining feature classes; the full generation pipeline (hash-sharding across VMs → SolidWorks augmentation → JSON export → graph compilation); tensor-by-tensor explanation of the `.pt` format
- `PART2_model_training_inference_deep_dive.md` — GNN → transformer primer; how A1/A2/A3 biases enter attention; layer-by-layer BrepMFR architecture walkthrough; Stage-1 loss mechanics; Stage-2 DANN/IWDAN theory

**Tooling built:**
- **SwOrchestrator** (C#/.NET WPF app + CLI): orchestrates SolidWorks on remote VMs — job queue, live log viewer, crash recovery, event stream (~1,500 lines committed Jun 28)
- **`ThreadCreationScript8.bas`** — SolidWorks VBA macro (491 lines) that programmatically inserts thread/text features into ABC STEP bodies to synthesize labeled training parts
- **`stage_shard.py`** + `Jenkinsfile` — deterministic hash-sharding of STEP files across multiple SolidWorks VM workers
- Root-level transfer utilities: `transfer_sldprts.py`, `transfer_uv_jsons.py`, `missing_sldprts_report.txt` (257 KB reconciliation of failed transfers), Groovy jobs `Distribute-Priority-STEP-Files.groovy` / `Transfer-Files-To-SW36912.groovy`

---

## 4. Phase 1 — The Class Imbalance Problem (core research thread)

### The numbers that drove everything
From the thread+text corpus class-weight audit (`proposed_solutions_for_data_imbalance.md`, committed):

| Class | Faces | Share | Weight (α=0.5) |
|---|---|---|---|
| Stock | 1,015,768 | 14.2 % | 0.534 |
| **Thread** | **57,350** | **0.8 %** | 2.248 |
| Text | 6,095,914 | 85.0 % | 0.218 |

Key finding: text outnumbers thread **106×**. Sqrt-inverse weights only correct ~10×, leaving a 10× residual gap; full inverse-frequency (α=1) destabilizes gradients. Additional diagnosis: text embosses fragment into *many faces each* (geometric root cause), so face-count imbalance understates part-level imbalance.

Documented four systemic issues in `class_imbalance_analysis.md`: focal-loss alpha inversion in the existing setup, no sampling strategy, augmentation disabled, no stratified splitting; then ranked mitigations (decoupled training + class-balanced loss, LDAM+DRW, Dice-CE hybrid, balanced sampling, generate more thread data).

### Experiment — Subgraph (k-hop neighborhood) training ✅ implemented
- Built `data/subgraph_sampler.py` (~400 lines): seed faces biased toward rare classes, extract k-hop induced subgraph, slice all node/edge/A-tensors consistently, pure node-level classification so subgraphs remain valid examples
- Configurable per-class seed budgets, e.g. `--subgraph_seeds_per_class "2,3,3,2,2"` (stock, thread, text, chamfer, fillet)
- Smoke-tested via `scripts/smoke/smoke_subgraph_sampler.py`; wired into `segmentation.py` as opt-in flags (`--subgraph_training --subgraph_k_hop 2`)
- Runs launched: `results/stage1/subgraph_k2_seeds233__2026-06-29` (+ `_v2`, `thread_text_subgraph_k2_`) — exploratory; kept opt-in rather than default after smoke evaluation

### Experiment — Length-bucketed smart batching ✅ shipped
- `data/length_bucket_batch_sampler.py`: buckets graphs by padded face count so batches are homogeneous → replaces batch_size 1 + accumulate 32 with batch_size 8 + accumulate 2
- Committed Jun 30; later rewritten around a **padded-quadratic cost budget** (see §7)

### Experiment — Leakage & generalization sanity tests
- `test_leakage.py` (Jul 1): verified STEP-key-aware splitting keeps all `..._step_NNN` variants of one body in a single split
- `test_unseen_data.py` (Jul 1): held-out-part evaluation harness (JSON → PyG → PyTorch with unseen-part enforcement)

---

## 5. Phase 2 — Customer Demo: GrabCAD Parts (Jul 14)

End-to-end demonstration on 28 externally sourced SolidWorks parts (prediction-only, no GT):

```
Z:\Demo\grab_cad_brepmfr_testing\jsons\*.json
  → json_to_brepmfr_pyg_optimized.py (--inference_profile lite) → *.pt
  → pyg_to_graphml.py (NEW batch exporter)                      → *.graphml (Gephi/Cytoscape)
  → run_thread_pyg_inference.py + best-v8.ckpt                  → inference_csvs\*.csv
```

- 28/28 conversions OK, 28/28 GraphML XML valid, 8,690 face rows written; stem↔row-count parity verified
- Predictions: **2,336 Stock / 5,643 Thread / 711 Text**, CUDA inference
- Built `scripts/visualization/pyg_to_graphml.py`: directed `MultiDiGraph` preserving parallel arcs; scalar node/edge attrs (face type, area, loop, adjacency, degree; edge type, length, angle, convexity); UV tensors deliberately omitted (GraphML must be scalar)

---

## 6. Phase 3 — Data Ops & Pipeline Infrastructure (Jul 16–19)

Operational log maintained in the training box's `project_progress_tracking.md` (43 KB living doc). Highlights with hard numbers:

**ABC corpus hygiene**
- Discovered `Z:\thread_and_text\lite\pyg` held **136,753 stale `.pt`** vs ~48 k live stems → wrote `prune_pyg_to_json_stems.py`; corrupt-read class-weight crashes eliminated; weights recomputed with `--skip-bad`
- Established that `*_101/_102` suffixes are **distinct solid bodies** of multi-body STEPs (differ in face count, e.g. 14 vs 270 faces) — not duplicates; one body per STEP suffices for training (each JSON = one `.pt` = one dataset item)
- Dedup accounting: Explorer showed 6,626 files in `C:\jsons` → actually **4,015 JSONs + 2,611 leftover `.SLDPRT` temps**; 0 true duplicates after `delete_duplicate_jsons.py` (~1,446 variants removed in an earlier pass); 14,705 STEPs remaining in `\\GR-SW65551\abc_steps`

**Allowlist-driven filtering loop** (keep parts whose faces never exceed Thread/Text prob 0.80)
- Stage-2 flagged ~2,677 clean JSONs → allowlist shards `allowed_step_keys_p1/p2/p3.txt` (**2,688 keys**); keep-list run: **9,464 keep vs ~6,785 delete** (logic inversion bug caught and fixed after one wrong-set deletion — restore path documented)
- Coverage audits: 188/2,688 allowlist keys present in the all-machines JSON share; 4,331/4,732 JSONs belong to those keys; non-allowlist backfill: 661 keys → 151 STEPs copied, 510 already gone
- Jenkins agents can't read UNC shares → `_gen_filter_jenkinsfile.py` **embeds** the allowlist into `Jenkinsfile.filter_abc_steps_no_thread_text` (parallel across 10 VMs); Groovy header-comment syntax fix (`//` not `#`)

**SolidWorks reliability**
- VBA cannot cancel a hung `LoadFile4` → wrote `Watchdog-StepOpen.ps1`: macro arms it before each open; if `in_progress.txt` persists past `OPEN_TIMEOUT_SEC` (60 s), append to `skip_list.txt` and kill `SLDWORKS.exe`; re-run resumes
- Post-batch order codified: dedup → batch ONNX inference → refresh allowlist → match vs share

---

## 7. Phase 4 — Training Throughput Engineering (Jul 19–21)

Commit series `optimzaed cpython bfs` → `Improve BrepMFR PyG functionality and structure` → `Fix fused QKV attention autograd and optimizer clipping`. Measured outcomes:

1. **NumPy BFS rewrite (A1/A3 construction)** — legacy BFS wrote hops element-by-element into a torch tensor: **77 s for N=788**; NumPy port is bit-identical (`match=True`) at **0.93 s ≈ 50–80× faster**. Killed per-graph ProcessPool spawning (8 Windows processes *per file*) in favor of persistent **file-level** pools (`--file-workers 12`, BLAS/OMP threads pinned to 1 per worker, resume-safe). Design target: rebuild the ~48 k-graph A1+A3 tree in <2 h (BFS-only estimate 0.14 h; network `torch.load/save` dominates).
2. **Sampler overhaul** — legacy sampler emitted **13,087 singleton batches** for large graphs (>300 faces) out of **16,995 total batches/epoch**. New `LengthBucketBatchSampler` packs greedily under a **padded-quadratic cost budget** (`batch_size × max_faces²`, default 4 M node², ≤64 graphs/batch), keeps A3-enabled (≤ `max_nodes_for_a3` 768) and A3-capped groups separate. Dense A1/A3 index tensors moved to **int32** through collation + H2D transfer (halved traffic).
3. **Fused scaled-dot-product attention** in all 8 encoder layers when weights aren't requested — numerical parity vs legacy path: **max abs error 1.79e-7**. Separate commit fixed a fused-QKV autograd bug + optimizer clipping interaction.
4. **AMP dtype fix** — `GraphAttnBias` crashed under `16-mixed` ("Index put requires Float… got Half"); edge buffers now allocated via `new_zeros` with explicit destination-dtype casts, float32 `a1_a3_scale` buffer cast before multiplying half activations.
5. **Cross-checkout sync tooling** — training launched from `Desktop\BrepMFR\...\BrepMFR` while dev lived in `Desktop\BrepMFR_PyG\...\BrepMFR_PyG`; a stale `CADSynth` crashed every launch. Wrote `sync_a1_a3_training_code.ps1`: copies the coordinated set (entry point, dataset/collator, encoder/bias layers, training wrapper), backs up targets, byte-compiles, and verifies required compatibility tokens.
6. **Trainer defaults** — TF32, no sanity val pass, 1,000-step warmup, full validation every 2 epochs, 2 dataloader workers × prefetch 2 × pinned memory (Windows fallback workers=0); all long-running PowerShell wrappers switched to `conda run --no-capture-output` so tqdm streams instead of appearing frozen.
7. **Speed probe run**: `results/logs/stage1/no_a2_speed_test_20260720_232610`.
8. **Manager reporting** — `analyze_tensorboard_run.py` emits self-contained HTML reports prioritizing macro class accuracy/mIoU over raw face accuracy.

Canonical hyperparameters across my runs: CE loss with α=0.5 sqrt-inverse class weights (train-split only), AdamW + warmup, `16-mixed`, 100 epochs, equal LRs 0.002 for scratch runs.

---

## 8. Phase 5 — ABC Integration, Corpus Growth & Model Lineage (Jul 20–27)

### Corpus evolution (the dataset behind every run below)

| Stage | Graphs | Notes |
|---|---|---|
| thread+text root_json | ~39 k | initial 3-class corpus (Stock/Thread/Text) |
| `no_a2` (A1+A3 attach) | ~48 k | lite→no_a2 upgrade-in-place |
| `root_json` grown | ~70 k | +22 k raw-label JSONs arrived |
| `no_a2_large` | **72,223** | delta-built expanded tree (hard-link seeded), quarantined invalids |
| 5-class (chamfer+fillet) | 39,450 → 60 k → **84 k** | August extension |

Delta-build engineering (all mine, documented in the tracking doc): select only stems missing from `no_a2/pyg`, identity-safe remap with strict unknown-label audits, `orjson` atomic rewrites, **hard-link seeding** of the base tree (a copy-seeded tree died mid-run on disk exhaustion → added free-space probes, `-MinFreeGB 20` refusal, abort-after-3-consecutive-write-failures, `-ResetOutput`), coverage verification, split rebuild (STEP-atomic, ABC ≥80 % train quota, seed 42), weight recomputation, and a validator that scans the *entire* split and supports `--quarantine-invalid` (moves bad graphs out, atomically fixes splits + ABC manifest, writes a reasons report) instead of dying at 20 errors.

Class-weight file lineage mirrors this: `source_train_alpha05.json` → `no_a2_large_70k_train_alpha05.json` → `cadsynth_5class_a1_a3_train_alpha05.json` → `_60k_` → `_84k_` (+ `new_abc_finetune_alpha05.json`, `abc_for_modelA_train_counts.json`).

### Checkpoint zoo (recipes; ~62 MB each, no_a2 profile unless noted)

| Checkpoint / run | Date | Recipe |
|---|---|---|
| `53k_thread_text` | Jul 20 | 3-class on ~39–53 k JSONs (lite era; v1 ONNX pkg = epoch-64 weights) |
| `abc_included_48k` | Jul 20 | same corpus + ABC mixed |
| `abc_with_no_a2` / `…_no_finetuning` | Jul 21 | A1+A3 upgrade pair: fresh fine-tune vs direct load control |
| `abc_finetune_43k` | Jul 21 | 43 k-graph ABC fine-tune (training box) |
| `30k_abc_finetuning` | Jul 25 | 30 k ABC-enriched fine-tune (best-v9 selected) |
| `abc_unique_prev_lite_and_noa2_finetuning` | Jul 25 | ABC parts unique vs prior corpora (best-v7) |
| `abc_finetune_forzen_bc` | Jul 26 | same, **BN stats frozen** (best-v3) |
| `thread_text_no_a2_70k_scratch_20260720_212117` | Jul 20 | first 70 k scratch attempt (aborted) |
| `thread_text_no_a2_70k_optimized_20260720_235522` | Jul 20 | same night, post-throughput-fix rerun ✅ |
| `thread_text_new_abc_finetune_v1` → `100K_MODEL_80EPOCH/best-v2.ckpt` | Jul | "new" ABC fine-tune line; run `thread_text_new_100k_run` reached **epoch 79 / step 1,188,560**; exported as ONNX demo v2 |
| `thread_text_full_a1_a3_scratch_abc70k_v1` | Jul 27 | full from-scratch A1+A3, ~70 k graphs incl. ABC |

Supporting QA scripts: `prepare_approved_abc_stock_jsons.py`, `prepare_model_a_unique_abc_dataset.py`, `quarantine_invalid_training_jsons.py`, `quarantine_known_empty_model_a_graphs.py`, `make_stock_only_eval_split.py`, `summarize_stock_only_inference.py`.

### Lite-checkpoint salvage workflow (recovered a "wrong-profile" trained model)
A 39-epoch checkpoint had been trained with `--inference_profile lite` (no A1/A2/A3 ever exercised). Since profile choice changes inputs, not shapes, the A1/A3 bias parameters existed but sat at init. Recovery: upgrade the same graphs to `no_a2` (keeping identical splits — no weight recomputation needed), then a **new** `--pre_train` fine-tune (never exact-resume) with separate AdamW param groups — backbone LR 1e-4, graph-bias LR 1e-3 — 1,000-step warmup and a five-epoch A1/A3 contribution ramp stored as a checkpoint-compatible scale buffer.

### Continuous ingestion pipeline
Documented in `END_TO_END_PIPELINE.md`, `WORKFLOW_GUIDE.md`, `CONTINUOUS_PIPELINE.md`: ABC STEPs → VBA JSON export (watchdog-guarded) → dedup → lite ONNX inference → Thread/Text-confidence filter → allowlist → Jenkins multi-agent copy to `C:\abc_steps_filtered` → requeue/dedup ledgers (`pipeline_dedup/`). Production deployment root: `\\Gr-sw66464\d\brepmfr_sw_inference` (`pyg_lite\` in, `csv_inference\` out).

---

## 9. Phase 6 — Klavuz Regression Hunt (Jul 25–27)

**Setup:** single real production part ("Klavuz_101", 502 faces: 243 stock / 161 thread / 98 text GT) as a regression fixture. Four ONNX models compared (`compare_klavuz_onnx_models.py`):

| Model | Recipe | Accuracy | Errors | mIoU | Failure mode |
|---|---|---|---|---|---|
| **A** | frozen-BN fine-tune (`best-v3`) | **97.41 %** | 13 | 94.70 % | 5 Stock→Thread, 8 Stock→Text; zero thread/text misses |
| B | unfrozen fine-tune | 76.49 % | 118 | 68.28 % | catastrophic Stock→Thread (111 faces) |
| C | other ABC variant | 81.08 % | 95 | 74.39 % | same Stock→Thread drift (94) |
| D | scratch-lineage checkpoint | 67.53 % | 163 | 52.58 % | opposite failure: all 161 threads → Stock |

**Root-cause diagnostics written Jul 27** (`standalone_scripts/diag_*.py`): geometry normalization & scale-shift (`diag_scale_shift`, `diag_klavuz_geometry`), angle-wrap artifacts (`diag_edge_angle_wrap`), helical thread signature/extent (`diag_thread_signature`, `diag_thread_extent`), stock label poisoning hypothesis (`diag_stock_label_poison`), angular-feature plots (`diag_klavuz_angular`, `diag_klavuz_plot`), prior/rescoring (`diag_class_prior`, `diag_rescore_klavuz`).

**Fix validation:** the retrained successor (`new_best-v5`) corrected **all 13 known failure faces** to Stock at 0.87–0.9999 confidence (`artifacts/regression/Klavuz_101/known_failure_comparison.csv`), while frozen-BN Model A re-validated standalone (PASS, mean conf 0.963) and Model D was additionally screened against a **411-graph stock-only holdout** (`artifacts/model_d_stock_holdout/`) as a false-positive guard. Conclusions recorded in `model_d_comparison_report.md`: BN-stat drift dominates (freeze it), and skipping ABC collapses thread recall.

---

## 10. Phase 7 — Final Five-Class Model & C++/ONNX Handoff (Aug 5–23)

Taxonomy extended {Stock, Thread, Text} → **{Stock, Thread, Text, Chamfer, Fillet}** (raw SW ids 15/24/70/101 → 0–4, identity-safe map for mixed folders). Corpus: `Z:\thread_and_text\cadsynth_with_fillets_and_champer\root_json` → parallel remap → `lite` → `no_a2` A1+A3 attach.

**Scratch-run ladder (all random init, A1+A3 at scale 1.0 from epoch 0, A3 cap 768, one-batch smoke before each launch, exact-resume support):**

| Run | Window | Outcome |
|---|---|---|
| `five_class_a1_a3_scratch_20260806_214444` | Aug 6–8 (≈29 h) | completed, 100/100 epochs |
| `five_class_a1_a3_scratch_20260806_233844` | Aug 6 | aborted/restart variant (no logs) |
| `five_class_a1_a3_60k_scratch_20260820` | Aug 20–21 (≈31 h) | completed, 100/100 → `5_class_60k_model` champion |
| `five_class_a1_a3_84k_scratch_20260823` | Aug 23–24 (≈21 h) | **internship ended mid-run at ≈ep 53/100** — metrics still climbing |

### Exact training results (extracted from TensorBoard event files on `\\Gr-sw66464`, decoder-parsed 2026-08-24)

Validation cadence: global metrics every ~half-epoch (tagged by optimizer *step*), per-class every epoch. All values below are **best-over-run unless marked final**.

#### 3-class runs (Stock / Thread / Text)

| Run | Wall time | Face Acc (best) | mIoU (best) | Eval loss (min) | Per-class IoU S / T / T | Recall S / T / T | Notes |
|---|---|---|---|---|---|---|---|
| `…no_a2_70k_scratch_20260720_212117` | 1.8 h | — | — | — | — | — | **aborted pre-warmup**: collapsed to all-Thread (T recall 1.00 but IoU 0.206; Stock recall 0.4 %) — killed at ep 0 |
| `no_a2_speed_test_20260720_232610` | 24 min | — | — | train 0.668 | — | — | 1-epoch throughput probe |
| `…no_a2_70k_optimized_20260720_235522` | ≈34.6 h | **99.56 %** @step 74,151 | **99.16 %** | 0.0144 | 98.85 / 99.64 / 99.09 | 99.75 / 99.97 / 99.65 | post-throughput-fix rerun; log ends ≈ep 54/100 |
| `thread_text_new_abc_finetune_v1` | ≈5.8 h | 99.51 % @step 2,124 (=first val) | 99.13 % | 0.0143 | 98.84 / 99.74 / 98.80 | 99.77 / 99.91 / 99.42 | **peaked at first validation then drifted down** to 98.33 % / 97.08 % by ep 14 — textbook synthetic→real fine-tune drift; motivated best-ckpt selection + frozen BN |
| `thread_text_full_a1_a3_scratch_abc70k_v1` | **63 h** | 99.42 % @step 85,100 | 98.90 % | 0.0226 | 98.60 / 99.59 / 98.59 | 99.75 / 99.80 / 99.55 | full 100 epochs; source of the Klavuz spot-check (conf 0.988); final 98.87 % / 97.92 % |

#### 5-class runs (Stock / Thread / Text / Chamfer / Fillet)

| Run | Face Acc (best) | Feat-Acc (best) | mIoU (best) | Eval loss (min) | Per-class IoU St / Th / Tx / Ch / Fi | Recall St / Th / Tx / Ch / Fi |
|---|---|---|---|---|---|---|---|
| `…scratch_20260806_214444` (39 k corpus) | 99.72 % @step 80,167 | 99.87 % | 98.64 % | **0.0040** | 99.19 / 99.82 / 99.34 / **95.25** / 99.58 | 99.60 / 99.99 / 99.76 / 98.56 / 99.89 |
| `…60k_scratch_20260820` ⭐ | **99.72 %** @step 94,011 | 99.86 % | **98.93 %** | 0.0063 | 99.19 / 99.81 / 99.42 / **96.76** / 99.64 | 99.50 / 99.99 / 99.76 / 99.21 / 99.89 |
| `…84k_scratch_20260823` *(incomplete, ≈ep 53/100)* | 99.64 % @step 91,103 | 99.83 % | 98.91 % | 0.0091 | 99.01 / 99.66 / 99.38 / **97.05** / 99.49 | 99.30 / 99.98 / 99.80 / **99.42** / 99.90 |

**Reading of the curves (worth saying out loud in a review):**
- **Chamfer is the hard class.** First-validation IoU was 67.2 % in the Aug-6 run and 83.8 % in the 84 k run — vs ≥96 % for everything else — converging slowly toward ~97 %. It also regressed late in completed runs (Aug-6 run: best 95.25 % @ep 87 → 91.70 % @ep 99), so best-checkpoint selection matters more than final weights for chamfer-heavy work.
- **The 84 k run was the strongest yet at cutoff:** at roughly half training it already beat the 60 k champion on chamfer IoU (97.05 vs 96.76) and chamfer recall (99.42 vs 99.21) with comparable globals — clear evidence to resume it rather than restart.
- **Fine-tunes drift; scratch holds.** `new_abc_finetune_v1` degraded monotonically after its first validation, while same-data scratch runs improved for 100 epochs — the quantitative backing for the frozen-BN / scratch strategy in §9.
- Typical full-run cost on GR-SW66464: **~30 h per 100-epoch five-class run** (70 k-graph three-class took 63 h pre-optimization-era settings).

Data-repair tooling shipped with it: `remap_missing_no_a2_json_labels.py`, `repair_unloadable_lite_pt.py`, `relocate_no_a2_tree.py`, `validate_a1_a3_finetune_data.py` (full-split scan, 5-class label bounds, A3-cap overflow report).

**Verification & deployment:**
- Held-out A1+A3 graph smoke (`test_a1_a3_onnx_out/summary.md`): 368/368 faces correct, mIoU 100 %
- Final scratch spot check on Klavuz_101 (last.ckpt, abc-70k line): Stock=147 / Thread=258 / Text=97 @ mean conf 0.988 — thread recall restored vs fine-tuned variants
- ONNX exports (`export_a1_a3_onnx.py`, `model_conversion_onnx.py`, ~810 lines each) → bundles with `label_map.json` + `model_config.json` (float32/int64 contract, `no_a2` profile). Parity discipline: v2 lite export validated `label_match 1080/1080, max_abs_diff 2.38e-6` on real graphs; A1+A3 exports zipped as `no_a2_72k_epoch50_onnx.zip`, `exported_a1_a3.zip`, `5_class_60k_model_onnx.zip`
- Real-graph validators before shipping: `validate_onnx_real_graphs.py`, `validate_a1_a3_onnx_real_graph.py`, `run_onnx_a1_a3_inference.py`

---

## 11. Where Everything Lives (handoff map)

- **Dev checkout:** `C:\Users\RZA2\Desktop\BrepMFR_PyG\BrepMFR_PyG` — code, scripts, docs, eval artifacts
- **Training box:** `\\Gr-sw66464\rza2\Desktop\BrepMFR\brepmfr_pyg\BrepMFR` — the checkout where all GPU runs executed; contains the 43 KB living ops log (`project_progress_tracking.md`) and **TensorBoard event files under `results/logs/stage1/<run_name>/tensorboard/version_0/`** for every run above (open with `tensorboard --logdir results/logs`)
- **Datasets:** `Z:\thread_and_text\{lite,no_a2,no_a2_large,five_class_a1_a3}` + `\\Gr-sw66464\D\thread_and_text\no_a2\pyg`
- **Production inference root:** `\\Gr-sw66464\d\brepmfr_sw_inference`

## 12. Repository Impact

- **9 commits** (Jun 28 – Jul 25), ~5,700 tracked insertions + binary assets; dozens of uncommitted experiment runs (46 GB of checkpoints/logs locally + the training box's `results/`)
- **Created:** `data/subgraph_sampler.py`, `data/length_bucket_batch_sampler.py`, Cython BFS kernel + build, 2 smoke suites, 10+ `diag_*` tools, ~20 dataset-lifecycle scripts, SwOrchestrator C# solution, 6 Jenkinsfiles/Groovy jobs, GraphML visualizer, ONNX export/validation suite, 3 operator manuals
- **Docs:** 2 deep dives (734 + 1,076 lines), 2 imbalance analyses (~600 lines), README_thread_text 5-class/recovery/subgraph sections, and the 666-line living ops log on the training box

## 13. Key Takeaways / What I'd Tell the Next Person

1. **Freeze BatchNorm when fine-tuning synthetic→real** — biggest single lever (97.4 % vs 76 % on the real part).
2. **Aggregate face accuracy lies on imbalanced corpora** — report macro per-class recall/mIoU; thread is 0.8 % of faces but is the product requirement.
3. **STEP-key-aware splits are non-negotiable** — near-duplicate body variants otherwise leak between train/test.
4. **Profile mismatches are recoverable, not fatal** — a lite-trained checkpoint regained A1/A3 via upgrade-in-place + grouped-LR fine-tune (backbone 1e-4, bias 1e-3, ramped 5 epochs).
5. **Bound A3 early** (`max_nodes_for_a3`) and pack batches by quadratic padding cost — went from 13,087 singleton batches to dense packed ones; int32 indices halve transfer time.
6. **Smoke-run everything** — one batch, one epoch, then scale; plus `conda run --no-capture-output` so you can see progress.
7. **Seed dataset trees with hard links, probe free space, and quarantine invalid graphs atomically** — three separate multi-hour failures were caused by disk exhaustion and silent corrupt `.pt`s.
8. **The 106× text/thread imbalance is structural** — balanced sampling + more thread synthesis beats bigger α.
9. **Keep a checkpoint lineage table from day one** (recipe + date per run name) and keep the ops log living — this document was reconstructed from both.

## 14. Open Items for Whoever Continues

- **Resume `five_class_a1_a3_84k_scratch_20260823`** (`results/stage1/…/last.ckpt`, exact-resume) — at ep ≈53/100 it was already beating the 60 k champion on chamfer; leaving it half-trained wastes the clearest gain on the table
- Subgraph sampling needs a full 100-epoch bake-off vs the length-bucket baseline
- Logit-adjustment / LDAM+DRW paths scaffolded (`logit_adjust_eval.py`, `post_hoc_logit_adjustment.md`) but unexplored on 5-class
- Scratch models still fail closed-world thread recall on some real parts — helix-parameter coverage gaps in synthesis suspected (`diag_thread_signature.py`)
- Chamfer needs targeted attention: slowest-converging class (67 % first-val IoU), late-run regression in completed runs — consider per-class checkpoint selection or a chamfer-heavy augmentation pass
- Continuous retraining loop (`CONTINUOUS_PIPELINE.md`) designed but optional/unstarted; C++ runtime consumes the ONNX bundles
- TensorBoard event files for every run live on `\\Gr-sw66464\...\results\logs\stage1\<run>\tensorboard\version_0\`; all headline numbers were decoded from them into §10 of this document on 2026-08-24
