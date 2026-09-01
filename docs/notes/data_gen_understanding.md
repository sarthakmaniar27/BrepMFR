# B-Rep Data Generation & Pipeline Orchestration Guide

This document provides a detailed breakdown of the **data generation process** for the BrepMFR project. It details how raw 3D CAD models (STEP files) are converted into Graph representations (JSON and PyTorch Geometric `.pt` files), the roles of the automation tools (Jenkins, CLI Orchestrator, and SolidWorks macros), and the architectural limitations of this setup.

---

## 1. End-to-End Data Generation Workflow

The process of translating a 3D physical design into a machine-learning-ready graph involves three main stages: **Sharding & Staging**, **Synthetic Geometry Augmentation (SolidWorks)**, and **Graph Compilation (Python/PyG)**. Instead of passively reading existing files, the system actively generates new, complex geometry (threads and engravings) on top of basic shapes to create a rich dataset for the ML model.

```
+-------------------------------------------------------------------------------------------------+
|                                     1. SHARDING & STAGING                                       |
|                                                                                                 |
|   [Master STEP Share] --- (Hash Sharding) ---> [VM 1 Local STEPS] ... [VM N Local STEPS]       |
+-------------------------------------------------------------------------------------------------+
                                                     |
                                                     v
+-------------------------------------------------------------------------------------------------+
|                         2. SYNTHETIC GEOMETRY AUGMENTATION (SolidWorks)                         |
|                                                                                                 |
|   [Local STEPS] ---> [SwOrchestrator.Cli] ---> [SolidWorks + Macro] ---> [BrepJson File]        |
|                            ^                           | (Adds Threads & Text)                  |
|                            |-------(Heartbeats)--------|                                        |
+-------------------------------------------------------------------------------------------------+
                                                     |
                                                     v
+-------------------------------------------------------------------------------------------------+
|                                 3. GRAPH COMPILATION (Python/PyG)                               |
|                                                                                                 |
|   [BrepJson File] ----------------------> [json_to_brepmfr_pyg.py] --------------------> [.pt] |
|                                        (Shortest paths, UV grids, angles)                       |
+-------------------------------------------------------------------------------------------------+
```

---

## 2. Distributed Parallel Processing: The Role of Jenkins

SolidWorks is a massive, proprietary Windows application. Because it requires significant CPU and memory, and can crash on corrupt geometry, we cannot run it on a single machine for large datasets. 

We use **Jenkins** to distribute the workload across a cluster of Windows Virtual Machines (VMs) in parallel.

### Sharding & Staging (`stage_shard.py`)
To prevent multiple VMs from processing the same files, we split the master dataset.
1. The Jenkins job parameters define `SOURCE_STEPS` (a central file server share containing all STEP files) and `TOTAL_SHARDS` (the number of active VMs).
2. Before SolidWorks runs, Jenkins invokes a script called `stage_shard.py`.
3. This script performs **deterministic hash sharding**: it computes a hash of each file name and assigns it to a VM index (`0` to `TOTAL_SHARDS - 1`). 
4. The VM copies only its assigned files from the file server to its local disk (`C:\ThreadRecognition\STEPS`). Hardlinks are used if the source and target are on the same volume to save disk space and I/O.
5. This process is **idempotent**: files that are already present locally are skipped, making pipeline restarts fast and cheap.

### The "Session 0" Windows Service Catch
A critical systems administration detail is how the Jenkins agents are configured on the Windows VMs:
* **The Problem:** Standard Jenkins agents are often installed as Windows Services. Windows Services run in **Session 0**, which is a non-interactive session that lacks access to the graphical desktop interface. Because SolidWorks has a graphical UI, launching it in Session 0 causes it to freeze or crash instantly.
* **The Solution:** The Jenkins agent on each VM must **not** run as a service. Instead:
  1. The VM is configured to auto-login a dedicated build user on boot.
  2. Windows Task Scheduler runs the Jenkins agent (`java -jar agent.jar`) interactively **"At log on"**.
  3. This ensures the Jenkins agent runs in a regular user session with full GUI capabilities, allowing SolidWorks to open and render normally.

---

## 3. SolidWorks CLI Orchestration (`SwOrchestrator.Cli.exe`)

SolidWorks was not designed to run unattended overnight on thousands of files. When processing massive CAD datasets, SolidWorks will inevitably:
* Leak memory and slowly consume all system RAM.
* Freeze on complex, self-intersecting, or corrupt geometry.
* Display modal pop-up dialogue boxes (e.g., *"This file has template errors. Do you want to continue?"*) that block execution.

To solve this, we wrap SolidWorks in a custom C# application called **`SwOrchestrator.Cli.exe`**.

### Monitoring, Heartbeats, and Crash Recovery
1. The orchestrator launches SolidWorks as a subprocess:
   ```cmd
   "SLDWORKS.exe" /m "ThreadCreationScript8.swp"
   ```
   The `/m` flag tells SolidWorks to execute the specified VBA macro immediately upon startup.
2. While SolidWorks is running, the macro updates a communication file (`current_chunk.txt` or `heartbeat.txt`) with the ID of the file currently being processed.
3. The orchestrator monitors the time since the last heartbeat:
   * **Stall Timeout (`STALL_TIMEOUT`):** If the heartbeat file is not updated for a set period (e.g., 900 seconds), the orchestrator assumes SolidWorks has frozen or is stuck on a modal dialog.
   * **Startup Grace (`STARTUP_GRACE`):** If SolidWorks fails to start and register its first heartbeat within a set timeframe, it is assumed to have crashed.
4. **Forced Termination & Skip Lists:** If a timeout occurs, the orchestrator kills the SolidWorks process tree:
   ```cmd
   taskkill /F /IM sldworks.exe /T
   ```
   It records the file that caused the stall in `skip_files.txt` so it will be bypassed in future runs, waits 10 seconds for the system RAM to clear, and then launches SolidWorks again to process the next file.

---

## 4. Synthetic Augmentation & Graph Compilation

Once SolidWorks opens a part, the VBA macro (`threadplustextgen8.swp`) performs **synthetic data augmentation** before extracting the geometry. It uses AI-assisted heuristics to procedurally generate complex features on otherwise simple CAD bodies.

### The VBA Macro (`threadplustextgen8.swp`)
Inside SolidWorks, the macro acts as a sophisticated data generator:
1. **Plan Generation:** It generates a persistent deterministic plan (`.plan` file) for each STEP file. It assigns 4 variations to each file with fixed roles: `THREAD_ONLY`, `ENGRAVE_ONLY`, `BOTH`, and another `THREAD_ONLY`.
2. **Auto-Scaling:** Small parts are scaled up to a minimum dimension of 50mm to ensure threads and text render correctly.
3. **Synthetic Threads:** 
   - It calls an internal utility (`swAppInternal.AITrainUtils(AI_CMD_THREAD_RIM_EDGES)`) to find circular "rim" edges suitable for threading.
   - It randomly selects a thread profile (e.g., metric tap, inch die, sp4xx bottle) and procedurally generates a **Sweep Thread Feature** on the edge.
4. **Synthetic Engraving:** 
   - It calls another utility (`AITrainUtils(AI_CMD_ENGRAVE_RECTS)`) to find flat or cylindrical bounding regions.
   - It fetches random text (Chinese characters, symbols, English) from `SketchText.txt` and uses the SolidWorks **Wrap Feature** to engrave or deboss the text onto the face.
5. **JSON Export:** Finally, it saves the augmented part as a new `.SLDPRT` file and issues a command (`swAppInternal.BaselineOutputCmd 100040`) to export the complex B-Rep geometry and topological structures of the *newly augmented model* into a **`BrepJson`** file.

### Graph Compilation (`json_to_brepmfr_pyg.py`)
A raw CAD JSON file is still not a graph. The python script [json_to_brepmfr_pyg.py](file:///c:/Users/RZA2\Desktop\thread_project\BrepMFR\scripts\inference\json_to_brepmfr_pyg.py) processes the JSON and builds the final tensors:
1. **Nodes & Edges:** Maps faces to graph nodes, and touching boundaries to graph edges.
2. **Shortest Paths (A1 & A3):** Computes the shortest topological path distance between all pairs of faces. It records the sequence of edge indices along these paths.
3. **Histograms (A2):** Computes pairwise surface distance (D2) and dihedral angle (A3) histograms between all face pairs.
4. **Angle Normalization:** Normalizes directed edge coordinate angles to the $[-\pi, \pi)$ range to ensure numerical stability for the neural network.
5. **Export:** Saves the compiled features as a single PyTorch Geometric `Data` object in a `.pt` file.

---

## 5. Spotted Limitations in this Pipeline

While this data generation pipeline works, it contains several critical bottlenecks and design fragilities:

### 1. Windows & Desktop GUI Dependency (Infrastructure Lock-in)
* **The Limit:** SolidWorks only runs on Windows and requires an interactive desktop session (Session 1+).
* **The Consequence:** The data generation pipeline cannot be easily containerized (e.g., using Docker on Linux) or run on headless Linux servers. Maintaining active, auto-login Windows VMs with interactive Jenkins agents is fragile, insecure, and hard to scale in the cloud.

### 2. Single-Threaded Processing
* **The Limit:** SolidWorks macros and API calls run on a single thread. 
* **The Consequence:** Each VM can only process one CAD file at a time. The overall throughput of the pipeline is low, forcing the team to run multiple separate VMs to process larger datasets.

### 3. Fragility of VBA Macros (`.swp` files)
* **The Limit:** The core extraction macro is written in VBA and compiled into a binary `.swp` file.
* **The Consequence:** Binary macro files cannot be easily version-controlled using Git (diffs cannot be viewed). VBA is a legacy, fragile language lacking modern debugging, linting, and automated unit testing frameworks.

### 4. Sharding Mid-Batch Re-run Fragility
* **The Limit:** Sharding is deterministic based on hash and `TOTAL_SHARDS`.
* **The Consequence:** If a VM crashes, or if you decide to change the number of VMs mid-batch, the files are completely remapped. A file previously processed on Shard 1 might be reassigned to Shard 2. This breaks the local folder cache check and forces the system to re-download and re-extract files.

### 5. Silent Dialog Freezes
* **The Limit:** SolidWorks frequently pops up dialogue boxes when reading STEP files (e.g., template warnings, circular dependencies, file recovery prompts).
* **The Consequence:** Unless these dialogs are explicitly suppressed (which requires delicate registry keys or Win32 API hacks), the macro blocks, forcing the orchestrator to wait for the full `STALL_TIMEOUT` (e.g. 15 minutes) before killing the process. This adds substantial idle time to the pipeline.

### 6. Aggregation Bottleneck
* **The Limit:** Shards process data locally on each VM.
* **The Consequence:** To combine the dataset for training, output files must be copied back over the network to a central storage share. Network drops during copy commands can lead to partial or corrupted dataset assemblies, requiring manual verification passes.
