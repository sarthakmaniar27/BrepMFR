# CAD Data Generation and Model Training Handoff

This document explains the main pipeline used to create the CAD training data,
where that data lives, and how it reaches the GPU machine for model training.
It intentionally focuses on the operational handoff rather than the internal
implementation of every Jenkins job or SolidWorks macro.

The project has three main stages:

1. Generate thread and text data from CADSynth STEP files.
2. Add fillets and chamfers to the generated thread/text SolidWorks parts.
3. Collect the JSON data, prepare PyTorch Geometric datasets, and train the ML
   model on the GPU machine.

## 1. Pipeline at a glance

```text
CADSynth STEP files
\\DZ4-SMR52-DSA\cadsynth_data\orginal_authors\step
        |
        | Jenkins distributes the STEP files across approximately 40 VMs
        v
Each VM: C:\Threads\cadsynth\cad_steps_filtered
        |
        | SwOrchestrator + SolidWorks + threadplustextgen12.swp
        v
Thread/text SLDPRTs and JSONs on each VM
  C:\Threads\sldprts
  C:\Threads\jsons
        |
        | SLDPRTs are consolidated for the second generation stage
        v
Thread/text SLDPRT source
\\GR-SW66711\cadsynth\sldprts_with_no_fillets_and_chamfer
        |
        | SolidWorks fillet/chamfer macro also reads the original UV JSONs
        | from \\DZ4-SMR52-DSA\cadsynth_data\sw_cadsynth\uv_json
        v
Each VM: fillet/chamfer SLDPRTs and five-class JSONs
  C:\Threads\fillet_and_chamfer_v7\output_sldprts
  C:\Threads\fillet_and_chamfer_v7\output_jsons
        |
        | JSON outputs are collected centrally
        v
Main shared data root
\\GR-SW67118\thread_and_text
        |
        | Copy/sync the required data to the GPU machine
        v
GPU machine GR-SW66464
  D:\thread_and_text\...
  JSON -> lite PyG -> no_a2 PyG dataset -> model training
        |
        v
Checkpoints and TensorBoard/CSV results
\\GR-SW66464\rza2\Desktop\BrepMFR\brepmfr_pyg\BrepMFR\results
```

The spelling in the real paths is important. Do not correct
`orginal_authors` or `champer` in scripts.

## 2. Important machines and shared locations

| Machine/location | Purpose |
| --- | --- |
| `\\DZ4-SMR52-DSA\cadsynth_data` | Main CADSynth data server: original STEP files, original UV JSONs, and supporting CADSynth data |
| `GR-SW34959` | Representative SolidWorks worker and shared Jenkins/data-generation file hub |
| `\\GR-SW34959\Threads` | Remote view of `C:\Threads` on `GR-SW34959`; shows the folder layout used on the generation VMs |
| Approximately 40 Jenkins VMs | Run SolidWorks generation jobs in parallel. Each normally uses the same local `C:\Threads` structure |
| `\\GR-SW66711\cadsynth` | Consolidated thread/text SLDPRT source used by the fillet/chamfer stage |
| `\\GR-SW67118\thread_and_text` | Main shared handoff/data root containing collected JSONs and prepared dataset copies |
| `GR-SW66464` | GPU training machine |
| `\\GR-SW66464\d\thread_and_text` | Data working area on the GPU machine; locally this is `D:\thread_and_text` |
| `\\GR-SW66464\rza2\Desktop\BrepMFR\brepmfr_pyg\BrepMFR` | Repository checkout used for the GPU training runs |
| `\\GR-SW65551\abc_steps` | Shared ABC STEP-file pool used by the additional ABC data pipeline |

On any worker that exposes a `Threads` share:

```text
\\<MACHINE>\Threads\...  =  C:\Threads\... on that machine
```

For example, `\\GR-SW34959\Threads\jsons` is the remote view of
`C:\Threads\jsons` on `GR-SW34959`.

## 3. Stage 1: generate threads and text

### Purpose

Stage 1 takes the original CADSynth `.step`/`.stp` files and uses SolidWorks to
create parts containing threads, text, or both. Each successful output consists
of a SolidWorks part and a labelled Brep JSON file.

### Main paths

| Data/item | Location |
| --- | --- |
| Central CADSynth STEP input | `\\DZ4-SMR52-DSA\cadsynth_data\orginal_authors\step` |
| STEP input assigned to one VM | `C:\Threads\cadsynth\cad_steps_filtered` |
| Stage 1 macro | `C:\Threads\macro\threadplustextgen12.swp` |
| SolidWorks executable | `C:\images\image_08_03\WinRel64\sldworks.exe` |
| Stage 1 SLDPRT output on one VM | `C:\Threads\sldprts` |
| Stage 1 JSON output on one VM | `C:\Threads\jsons` |
| Stage 1 status/resume data | `C:\Threads\status` |
| Stage 1 orchestrator logs | `C:\Threads\SwOrchestrator10_CLI\LogFiles` |
| Collected thread/text JSONs | `\\GR-SW67118\thread_and_text\root_json` |
| Consolidated thread/text SLDPRTs used by Stage 2 | `\\GR-SW66711\cadsynth\sldprts_with_no_fillets_and_chamfer` |

Older Stage 1 SLDPRT stores referenced by the Jenkins scripts are:

```text
\\Gr-sw26877\d\brepmfr_sldprts\cadsynth
\\Gr-sw34959\d\brepmfr_sldprts\cadsynth
```

The consolidated `GR-SW66711` folder should be preferred for the current
fillet/chamfer workflow.

### How the VM fleet is used

Jenkins divides the central STEP collection into non-overlapping groups and
copies one group to each VM. The VMs process their groups independently, which
allows roughly 40 copies of SolidWorks to generate data in parallel.

Jenkins is used to distribute files and monitor the machines. mRemoteNG/RDP is
used to access a machine interactively, start or restart the generation
command, and clear any SolidWorks dialogs. SolidWorks must run in an interactive
Windows login session.

### Command run on each VM

Open a command prompt in `C:\Threads\SwOrchestrator10_CLI` and run:

```bat
SwOrchestrator.Cli.exe --steps C:\Threads\cadsynth\cad_steps_filtered --macro C:\Threads\macro\threadplustextgen12.swp --status C:\Threads\status --sw C:\images\image_08_03\WinRel64\sldworks.exe --output C:\Threads\sldprts --failure-threshold 1 --stall-timeout 200 --startup-grace 200 --poll-interval 5
```

The job can be restarted with the same command. Existing outputs and the files
under `C:\Threads\status` allow it to continue instead of starting the entire
VM shard again.

### Stage 1 outputs

A STEP can create several variants, for example:

```text
00014132_thread_v8.SLDPRT
00014132_engrave.SLDPRT
00014132_both_v7.SLDPRT
```

The corresponding JSON names can contain an additional numeric body suffix:

```text
00014132_thread_v8_105.json
00014132_engrave_102.json
```

The JSON files are the main input to the ML dataset preparation. The SLDPRTs
are retained because Stage 2 opens them again to add fillets and chamfers.

## 4. Stage 2: add fillets and chamfers

### Purpose

Stage 2 opens the existing thread/text SLDPRTs, adds fillet and chamfer
features, saves a new SLDPRT, and exports a new JSON. These JSONs contain the
five classes used by the final model: Stock, Thread, Text, Chamfer, and Fillet.

### Main paths

| Data/item | Location |
| --- | --- |
| Main Stage 2 input SLDPRTs | `\\GR-SW66711\cadsynth\sldprts_with_no_fillets_and_chamfer` |
| Original UV JSONs | `\\dz4-smr52-dsa\cadsynth_data\sw_cadsynth\uv_json` |
| Central Stage 2 manifest | `\\GR-SW34959\Threads\fillet_and_chamfer_creation\input_manifest.tsv` |
| Manifest on one VM | `C:\Threads\fillet_and_chamfer_v7\input_manifest.tsv` |
| VM assignment/configuration | `C:\Threads\fillet_and_chamfer_v7\machine_config.txt` |
| Fillet/chamfer macro | `C:\Threads\macro\fillet_chamfer_uv_json_merged.swp` |
| Stage 2 start command | `C:\Threads\SwOrchestrator10_CLI\START_FILLET_CHAMFER_V7.cmd` |
| Stage 2 SLDPRT output on one VM | `C:\Threads\fillet_and_chamfer_v7\output_sldprts` |
| Stage 2 JSON output on one VM | `C:\Threads\fillet_and_chamfer_v7\output_jsons` |
| Stage 2 status/resume data | `C:\Threads\fillet_and_chamfer_v7\status` |
| Collected five-class JSON corpus | `\\GR-SW67118\thread_and_text\cadsynth_with_fillets_and_champer\root_json` |

### How it is run

Jenkins distributes the same manifest plus a different machine assignment to
each participating VM. The manifest points to the source SLDPRT for every job,
and the machine configuration tells that VM which rows it owns.

Through mRemoteNG/RDP, double-click:

```text
C:\Threads\SwOrchestrator10_CLI\START_FILLET_CHAMFER_V7.cmd
```

This opens the assigned SLDPRTs and writes the two local output folders.
Example output pair:

```text
C:\Threads\fillet_and_chamfer_v7\output_sldprts\00000000_both_v8_fillet_chamfer.SLDPRT
C:\Threads\fillet_and_chamfer_v7\output_jsons\00000000_both_v8_fillet_chamfer_105.json
```

The JSONs are collected into the central five-class `root_json` folder shown
above. The supplied Jenkins collector historically targets:

```text
\\GR-SW66464\d\thread_and_text\cadsynth_with_fillets_and_champer\root_json
```

The main handoff copy is now under `\\GR-SW67118\thread_and_text`. Before a new
collection run, confirm whether the destination should be `GR-SW67118` or the
GPU working copy on `GR-SW66464`; otherwise new output can be split between the
two servers.

The supplied automation collects Stage 2 JSONs, but it does not contain a
central collection job for the Stage 2 SLDPRTs. Unless they are copied by a
separate/manual process, they remain in each VM's `output_sldprts` folder.

## 5. Central data used for model training

The main shared handoff root is:

```text
\\GR-SW67118\thread_and_text
```

Important contents are:

| Folder | Contents/use |
| --- | --- |
| `root_json` | Collected thread/text JSONs from Stage 1 |
| `cadsynth_with_fillets_and_champer\root_json` | Combined/final five-class CADSynth JSON corpus |
| `cadsynth_with_fillets_and_champer\five_class_a1_a3\lite` | Lite PyG graph dataset and `train.txt`, `val.txt`, `test.txt` |
| `abc_json` | Collected/generated ABC JSON data |
| `new_abc_json_25k` | Additional approximately 25K ABC-derived JSON corpus |
| `no_a2_large` | Prepared three-class A1/A3 dataset with `pyg` and split files |
| `no_a2_72k_plus_new_abc_30k` | Combined prepared dataset used for ABC-enriched training/fine-tuning |
| `demo_model_lite` | Smaller prepared dataset used for the lite/demo model |
| `pipeline_scripts` | Scripts used by the continuous ABC data pipeline |
| `pipeline_state` | State/ledger files for the ABC ingestion pipeline |
| `reports` | Dataset transfer, validation, and reconciliation reports |

Additional ABC paths:

| Data | Location |
| --- | --- |
| Shared ABC STEP pool | `\\GR-SW65551\abc_steps` |
| Filtered ABC STEPs on a worker | `C:\abc_steps_filtered` |
| ABC JSON corpus | `\\GR-SW67118\thread_and_text\abc_json` |
| New ABC JSON corpus | `\\GR-SW67118\thread_and_text\new_abc_json_25k` |

ABC data was used for the earlier three-class thread/text models and
fine-tuning experiments. The final five-class preparation script is based on
the CADSynth fillet/chamfer `root_json` corpus.

## 6. Transfer to the GPU machine

The GPU machine is `GR-SW66464`. Its main data working area is:

```text
D:\thread_and_text
```

Remote view:

```text
\\GR-SW66464\d\thread_and_text
```

The training scripts also use paths beginning with `Z:\thread_and_text`. `Z:`
is a mapped-drive form used in the scripts. Before running anything, verify
that `Z:` points to the expected data location. Using direct
`D:\thread_and_text` paths on `GR-SW66464` avoids depending on the drive map.

The five-class GPU data locations are:

```text
D:\thread_and_text\cadsynth_with_fillets_and_champer\root_json
D:\thread_and_text\cadsynth_with_fillets_and_champer\five_class_a1_a3\lite
D:\thread_and_text\cadsynth_with_fillets_and_champer\five_class_a1_a3\no_a2
```

The `no_a2` folder is the final training-ready dataset:

```text
no_a2\
|-- pyg\          # PyTorch Geometric .pt graph files
|-- train.txt
|-- val.txt
|-- test.txt
`-- quarantine_invalid_graphs\
```

## 7. Model dataset preparation and training

### Repository and environment

GPU checkout:

```text
C:\Users\RZA2\Desktop\BrepMFR\brepmfr_pyg\BrepMFR
```

Remote view:

```text
\\GR-SW66464\rza2\Desktop\BrepMFR\brepmfr_pyg\BrepMFR
```

Conda environment:

```text
brep_mfr_pyg
```

### Prepare the five-class dataset

The preparation script converts the collected JSONs into lite `.pt` graphs,
creates STEP-aware train/validation/test splits, adds the A1/A3 graph features,
validates the dataset, and calculates class weights:

```powershell
cd C:\Users\RZA2\Desktop\BrepMFR\brepmfr_pyg\BrepMFR

powershell -ExecutionPolicy Bypass -File scripts\threads\prepare_5class_a1_a3_scratch.ps1 `
  -JsonDir "D:\thread_and_text\cadsynth_with_fillets_and_champer\root_json" `
  -WorkRoot "D:\thread_and_text\cadsynth_with_fillets_and_champer\five_class_a1_a3" `
  -ApplyLabelRemap
```

Important: `-ApplyLabelRemap` changes labels in the JSON files. Run the first
audit without that switch when working with a newly copied corpus, and only
apply the remap to the intended working copy.

The prepared class-weight files live in the repository under:

```text
artifacts\class_weights\thread_text
```

### Start five-class training

```powershell
powershell -ExecutionPolicy Bypass -File scripts\threads\train_5class_a1_a3_from_scratch.ps1 `
  -DatasetRoot "D:\thread_and_text\cadsynth_with_fillets_and_champer\five_class_a1_a3\no_a2" `
  -ClassWeights "artifacts\class_weights\thread_text\cadsynth_5class_a1_a3_84k_train_alpha05.json" `
  -RunName "<new_run_name>"
```

For an exact resume, use the same dataset, class weights, run name, and model
configuration, and pass:

```powershell
-ResumeFromCheckpoint "results\stage1\<run_name>\last.ckpt"
```

### Training outputs

| Output | Location in the GPU repository |
| --- | --- |
| Checkpoints | `results\stage1\<run_name>\best.ckpt`, `best-v*.ckpt`, and `last.ckpt` |
| TensorBoard logs | `results\logs\stage1\<run_name>\tensorboard` |
| CSV metrics | `results\logs\stage1\<run_name>\csv_metrics` |
| Older selected model copies | `model_checkpoints` |
| Class weights | `artifacts\class_weights\thread_text` |

Remote checkpoint example:

```text
\\GR-SW66464\rza2\Desktop\BrepMFR\brepmfr_pyg\BrepMFR\results\stage1\five_class_a1_a3_84k_scratch_20260823
```

## 8. What to check before continuing the pipeline

1. Confirm the current Jenkins worker list and which VMs are online.
2. Confirm each worker has an interactive Windows session before starting
   SolidWorks.
3. For Stage 1, verify the input and output folders on a representative VM.
4. For Stage 2, verify `machine_config.txt`, `output_sldprts`, and
   `output_jsons`.
5. Do not treat a completion marker as proof that every input produced both
   files; compare unique basenames and require both files where appropriate.
6. Before collecting, confirm the intended central destination: shared
   handoff storage on `GR-SW67118` or the GPU working copy on `GR-SW66464`.
7. Before training, verify the selected `root_json`, `no_a2` dataset, split
   files, class-weight file, and checkpoint all belong to the same corpus/run.

## 9. Remaining handoff items to confirm

The supplied scripts do not completely establish:

1. The exact Jenkins/manual command used to consolidate Stage 1 worker
   SLDPRTs into the central SLDPRT stores.
2. Whether new Stage 2 JSON collections should write directly to
   `GR-SW67118` or first to `GR-SW66464` and then be copied.
3. Whether Stage 2 output SLDPRTs should remain on the VMs or be archived
   centrally. The supplied collector handles JSONs only.
4. The authoritative current list of the approximately 40 generation VMs.
5. The mRemoteNG connection-file and credential handoff. Credentials should
   be transferred securely and should not be stored in this Markdown file.

## 10. Supporting code locations

| Item | Location |
| --- | --- |
| Main development checkout | `C:\Users\RZA2\Desktop\BrepMFR_PyG\BrepMFR_PyG` |
| Jenkins scripts reviewed for this handoff | `C:\Users\RZA2\Desktop\jenkins_scripts` |
| Jenkins scripts on representative worker share | `\\GR-SW34959\Threads\jenkins` |
| Stage 1 macro on workers | `C:\Threads\macro\threadplustextgen12.swp` |
| Stage 2 macro on workers | `C:\Threads\macro\fillet_chamfer_uv_json_merged.swp` |
| Stage 1 CLI | `C:\Threads\SwOrchestrator10_CLI\SwOrchestrator.Cli.exe` |
| Stage 2 launcher | `C:\Threads\SwOrchestrator10_CLI\START_FILLET_CHAMFER_V7.cmd` |
| Five-class preparation script | `scripts\threads\prepare_5class_a1_a3_scratch.ps1` |
| Five-class training script | `scripts\threads\train_5class_a1_a3_from_scratch.ps1` |

Detailed model architecture, experiment history, and older recipes remain
under `docs` and `docs\notes`. They are supporting references; the main
operational pipeline and data locations are in this document.
