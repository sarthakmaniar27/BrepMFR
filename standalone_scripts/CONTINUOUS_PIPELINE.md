# Continuous Stage-1 + Stage-2 pipeline (no duplicates)

Producer/consumer design so filtering (Stage 1) and synthetic thread/text
generation (Stage 2) run **in parallel**, without redoing work.

```text
                    ┌─────────────────────────────────────┐
                    │  Central ledger (GR-SW66464)         │
                    │  D:\thread_and_text\pipeline_state\  │
                    │    pending_keys.txt                  │
                    │    stage1_seen_keys.txt              │
                    │    stage2_distributed_keys.txt       │
                    │    stage2_done_keys.txt  ◄── seed    │
                    │         from abc_json (~3k keys)     │
                    └───────────────┬─────────────────────┘
                                    │
         ┌──────────────────────────┼──────────────────────────┐
         ▼                          ▼                          ▼
  Stage-1 job (hourly)     Distribute job (~15m)        Cleanup job (hourly)
  infer + enqueue          append STEPs to              harvest C:\Threads\jsons
  NEW clean keys           C:\abc_steps_filtered        prune finished STEPs
                           on 10 agents                 merge → stage2_done
                           (NO CLI start)
                                    │
                                    ▼
                           Local CLI / SolidWorks on each
                           agent watches filtered folder
                           → writes C:\Threads\jsons
```

## Why not the old redistribute Jenkinsfile?

`Jenkinsfile.filter_jenkins_job` **clears** `C:\abc_steps_filtered`, rebalances
everything, then starts the CLI. That fights a continuous flow:

- Wiping the folder interrupts in-progress Stage-2 work
- Re-shipping already-done keys creates duplicates
- Coupling distribute + CLI prevents Stage-1 from feeding work while Stage-2 runs

New jobs **append only**, track keys in ledgers, and leave the CLI outside Jenkins.

---

## Folder map

| Path | Machine | Role |
|------|---------|------|
| `D:\thread_and_text\abc_json` | GR-SW66464 | Already-finished Stage-2 JSONs (~10k / ~3k STEP keys) — **seed source** |
| `D:\thread_and_text\pipeline_state\` | GR-SW66464 | Dedup ledgers (text files) |
| `D:\thread_and_text\pipeline_scripts\` | GR-SW66464 | Copy of `standalone_scripts/pipeline_dedup/` |
| `C:\jsons` (+ `inference\`) | Inference node | Stage-1 JSON + ONNX outputs |
| `C:\abc_steps` | Each of 10 agents | Full STEP pool |
| `C:\abc_steps_filtered` | Each of 10 agents | Stage-2 work queue (append-only) |
| `C:\Threads\jsons` | Each of 10 agents | Stage-2 synthetic JSON outputs |

Shared STEP key (same as the rest of the repo):

```text
JSON:  00000001_..._step_000_101.json
STEP:  00000001_..._step_000.step
Key:   00000001_..._step_000
```

---

## Dedup rules (never redo)

| Stage | Skip if key is in… |
|-------|--------------------|
| Stage-1 enqueue | `pending` ∪ `stage2_distributed` ∪ `stage2_done` |
| Distribute | `stage2_done` ∪ `stage2_distributed` (and only take from `pending`) |
| Agent copy | File already present in `C:\abc_steps_filtered` |
| Cleanup harvest | Keys already in `stage2_done` (merge is idempotent) |

**Seed once** so the ~3k already-processed STEPs never re-enter the queue:

```powershell
# On GR-SW66464, after copying pipeline_dedup scripts:
python D:\thread_and_text\pipeline_scripts\seed_stage2_done_keys.py
python D:\thread_and_text\pipeline_scripts\seed_stage2_done_keys.py --dry-run
```

---

## One-time setup

### Option A — via Jenkins (recommended)

Create a Pipeline job from `Jenkinsfile.pipeline_bootstrap_state` (Pipeline script
from SCM pointing at this repo). Run it once on `GR-SW66464` with defaults:

- `CHECKOUT_SCM=true` → checks out the repo on that agent
- copies `standalone_scripts/pipeline_dedup/*.py` → `D:\thread_and_text\pipeline_scripts`
- creates `D:\thread_and_text\pipeline_state`
- seeds `stage2_done_keys.txt` from `D:\thread_and_text\abc_json`

Safe to re-run (seed **merges**; empty sibling ledgers are created if missing).
Use `DRY_RUN=true` first if you want a preview.

If the job cannot use SCM checkout, set `CHECKOUT_SCM=false` and point
`SCRIPT_SOURCE` at a folder on that machine (or a share) that already contains
the `pipeline_dedup` `.py` files.

### Option B — manual (same steps)

1. On **GR-SW66464**, create folders:

```powershell
New-Item -ItemType Directory -Force D:\thread_and_text\pipeline_state
New-Item -ItemType Directory -Force D:\thread_and_text\pipeline_scripts
```

2. Copy `standalone_scripts/pipeline_dedup/*` → `D:\thread_and_text\pipeline_scripts\`

3. Seed done keys from existing outputs:

```powershell
python D:\thread_and_text\pipeline_scripts\seed_stage2_done_keys.py
```

4. Confirm Jenkins node labels match the Jenkinsfiles:
   - State / seed machine: `GR-SW66464`
   - 10 agents: `WALSWKQA19383` … `WALSWKQA19441` (edit lists if needed)
   - Inference node label in `Jenkinsfile.pipeline_stage1_enqueue`

5. Create three Jenkins Pipeline jobs (Pipeline script from SCM or pasted):

| Job name (suggested) | Jenkinsfile | Cron |
|----------------------|-------------|------|
| `Pipeline-Stage1-Infer-And-Enqueue` | `Jenkinsfile.pipeline_stage1_enqueue` | `H * * * *` |
| `Pipeline-Distribute-Filtered-Steps` | `Jenkinsfile.pipeline_distribute_only` | `H/15 * * * *` |
| `Pipeline-Cleanup-Stage2-Filtered` | `Jenkinsfile.pipeline_cleanup_stage2` | `H * * * *` |

6. On each agent, keep your existing Stage-2 CLI / SolidWorks process running
   against `C:\abc_steps_filtered` → `C:\Threads\jsons` **outside** Jenkins
   (or start it manually once). Jenkins will only feed and clean the folders.

---

## Parallel operation (what you asked for)

Example hour:

1. **:00** Stage-1 finishes ~500 new clean keys → appends to `pending_keys.txt`
2. **:05** Distribute job wakes → ships those 500 round-robin onto the 10 VMs
   (append into each `C:\abc_steps_filtered`) → marks them `distributed`
3. Agents’ local CLI immediately sees new STEPs and starts generating JSONs
4. **Meanwhile** Stage-1 can already be inferring the next batch
5. **:00 next hour** Cleanup harvests keys from each `C:\Threads\jsons`, deletes
   finished STEPs from filtered folders, merges into `stage2_done`

No job waits for the other to finish the whole corpus.

---

## Manual / dry-run commands

```powershell
# Preview enqueue
python standalone_scripts/pipeline_dedup/enqueue_filtered_keys.py `
  --allowlist C:\jsons\inference\allowed_step_keys.txt `
  --state-dir D:\thread_and_text\pipeline_state --dry-run

# Preview distribute plan
python standalone_scripts/pipeline_dedup/plan_distribute_chunks.py `
  --state-dir D:\thread_and_text\pipeline_state `
  --out-dir $env:TEMP\chunks --dry-run

# Local append-copy on one agent
python standalone_scripts/pipeline_dedup/append_steps_from_allowlist.py `
  --allowlist chunk_WALSWKQA19383.txt --dry-run
```

---

## Scripts in `pipeline_dedup/`

| Script | Role |
|--------|------|
| `key_utils.py` | Shared STEP-key parse + ledger IO |
| `seed_stage2_done_keys.py` | Seed done ledger from `abc_json` |
| `enqueue_filtered_keys.py` | Stage-1 → pending (dedup) |
| `plan_distribute_chunks.py` | Pending → per-node chunks + commit |
| `append_steps_from_allowlist.py` | Local append-copy helper |
| `cleanup_agent_filtered.py` | Agent harvest + prune helper |
| `merge_harvested_done_keys.py` | Merge agent harvests → done ledger |

---

## Machines with empty `C:\abc_steps`

**Do not** create an empty local `C:\abc_steps` and expect distribute to work.
There is still nothing to copy → every key is `MISSING in source`.

**Do this instead:**

1. Point `SOURCE_DIR` / Jenkins param `SHARED_SOURCE` at a **shared** STEP pool
   every agent can read, e.g. `\\GR-SW65551\abc_steps` (adjust to your real share).
2. Each agent copies from that share → local `C:\abc_steps_filtered` only.
3. Only list Jenkins nodes that are **online** and have the correct label
   (comment out labels like `GR-SW43701` until the agent exists).
4. Commit is **success-only**: missing/timeout keys stay in `pending_keys.txt`.

Optional later: sync a local `C:\abc_steps` cache onto each machine for speed;
until then, shared UNC is enough.

If a previous distribute run committed keys that never copied, put those keys
back into `pending_keys.txt` (or clear bogus entries from
`stage2_distributed_keys.txt`) before re-running.
