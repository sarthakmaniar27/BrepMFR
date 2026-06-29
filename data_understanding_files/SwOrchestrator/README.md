# SOLIDWORKS Thread Orchestrator

A Windows tool for running a SOLIDWORKS VBA thread-creation macro across
thousands of STEP files unattended. It launches SOLIDWORKS via
`SLDWORKS.exe /m <macro>.swp`, detects crashes and hangs, restarts as
needed, and auto-blacklists files that crash SOLIDWORKS repeatedly. The
macro itself is resumable so progress survives crashes.

Ships in three flavours sharing one core engine:

| | Use it for | Output |
| --- | --- | --- |
| **`SwOrchestrator.Gui.exe`** | Interactive monitoring; a person watching one VM | WPF window with progress bar + colour-coded log |
| **`SwOrchestrator.Cli.exe`** | Jenkins / scripts / unattended automation | Console: streams lines to stdout, sets exit code |
| **`SwOrchestrator.Core`** | Class library; both .exes link against it | n/a (referenced as a project) |

## Repository layout

```
SwOrchestrator/
├── SwOrchestrator.sln              -- Open this in Visual Studio 2022
├── README.md                       -- This file
├── Jenkinsfile                     -- Sample declarative pipeline (with hash sharding)
├── stage_shard.py                  -- Distributes STEP files across VMs by deterministic hash sharding
├── ThreadCreationScript8.bas       -- The VBA macro (import into VBA editor, save as .swp)
├── SwOrchestrator.Core/            -- Shared engine (AppSettings, Orchestrator, LogEntry, events)
│   ├── AppSettings.cs              -- Settings model + JSON persistence
│   ├── Orchestrator.cs             -- Launch / watch / restart loop
│   ├── OrchestratorEvent.cs        -- Event types pushed to consumers via IProgress<T>
│   ├── LogEntry.cs                 -- Pure-data log row (no WPF dependency)
│   └── SwOrchestrator.Core.csproj
├── SwOrchestrator.Gui/             -- WPF window
│   ├── App.xaml / .cs
│   ├── MainWindow.xaml / .cs
│   ├── MainViewModel.cs
│   ├── LogLevelToBrushConverter.cs -- XAML converter that maps LogLevel -> Brush
│   └── SwOrchestrator.Gui.csproj
└── SwOrchestrator.Cli/             -- Console app for Jenkins
    ├── Program.cs                  -- Arg parsing, event-to-stdout writer, exit codes
    └── SwOrchestrator.Cli.csproj
```

## Building

Open `SwOrchestrator.sln` in Visual Studio 2022 (".NET desktop development"
workload), then build the solution. Both `.exe`s are produced.

From the command line:

```
dotnet build SwOrchestrator.sln -c Release
```

That builds all three projects. The executables land in:

```
SwOrchestrator.Gui\bin\Release\net8.0-windows\SwOrchestrator.Gui.exe
SwOrchestrator.Cli\bin\Release\net8.0\SwOrchestrator.Cli.exe
```

For a deployable single-file `.exe` of either project:

```
dotnet publish SwOrchestrator.Cli\SwOrchestrator.Cli.csproj -c Release -r win-x64
dotnet publish SwOrchestrator.Gui\SwOrchestrator.Gui.csproj -c Release -r win-x64
```

Target machines need the **.NET 8 Desktop Runtime** installed. To eliminate
that requirement, set `<SelfContained>true</SelfContained>` in the
project's Release `PropertyGroup` and re-publish; the output gets larger
but is fully portable.

## Installing the macro on each VM

The orchestrator launches SOLIDWORKS with a `.swp` macro file. To make one
from the bundled `ThreadCreationScript8.bas`:

1. Open SOLIDWORKS, then `Tools -> Macro -> Edit...` and pick your existing
   v7 `.swp` (or any `.swp`).
2. In the VBA editor, open the existing module, `Ctrl+A` to select all,
   paste the contents of `ThreadCreationScript8.bas` over the top, or
   `File -> Import File...` and pick the `.bas` directly.
3. Under `Tools -> References...`, make sure **SldWorks Type Library** and
   **SldWorks Constant Type Library** are checked (same as before).
4. `File -> Save` to write `ThreadCreationScript8.swp`.
5. Verify the path constants at the top of the macro (`STEP_FOLDER`,
   `STATUS_FOLDER`, `BREP_JSON_OUT`) match the per-VM layout.

This step is the same on every VM.

## Using the GUI

Run `SwOrchestrator.Gui.exe`. The window has four panels:

- **Configuration**: paths to `SLDWORKS.exe`, the `.swp` macro, the STEPS
  folder, and the status folder. Plus the numeric tunables. Saved to
  `%APPDATA%\SwOrchestrator\settings.json` automatically on close / Start /
  the "Save Settings" button.
- **Progress strip**: progress bar, current count, run-state dot, current
  macro status, current STEP file, last heartbeat time.
- **Activity log + Skip list**: colour-coded events; auto-scrolls. Files
  the orchestrator has auto-blacklisted after repeated crashes appear on
  the right.
- **Buttons**: Start, Stop (confirms), Reset state (clears heartbeat and
  done marker but not outputs / skip list), Open status folder.

Closing the window while a run is in progress asks for confirmation and
stops cleanly.

## Using the CLI

```
SwOrchestrator.Cli.exe --steps  C:\ThreadRecognition\STEPS ^
                       --macro  C:\ThreadRecognition\ThreadCreationScript8.swp ^
                       --status C:\ThreadRecognition
```

`--solidworks` defaults to the SOLIDWORKS Corp install path; pass it
explicitly if your install is elsewhere. All tunables (`--stall-timeout`,
`--startup-grace`, `--poll-interval`, `--cooldown`, `--crash-threshold`,
`--variations`, `--max-restarts`) are optional. Run `--help` for the full
list.

Each event prints one line, e.g.:

```
[2026-05-19 17:30:01] [INFO ] Pre-launch progress: 0 / 2847
[2026-05-19 17:30:01] [STATE] Running
[2026-05-19 17:30:01] [INFO ] Launched SOLIDWORKS pid=4532
[2026-05-19 17:33:14] [HB   ] status=var_start_3 file=00000055.stp
[2026-05-19 17:33:18] [PROG ] 1 / 2847 (0.0%)
...
[2026-05-19 20:14:55] [INFO ] Batch complete. Final: 2847 / 2847.
[2026-05-19 20:14:55] [STATE] Complete
```

Warnings and errors go to stderr; everything else to stdout. The exit code
is:

| Code | Meaning |
| --- | --- |
| `0` | Batch completed - every STEP file has all variations on disk |
| `1` | Failed (bad config, max restarts exceeded, exception) |
| `2` | Cancelled by Ctrl+C (or `kill`) |
| `3` | Bad command-line arguments |

Press Ctrl+C once for a graceful stop (kills SOLIDWORKS, exits with 2),
twice to force-quit.

You can also load a settings JSON file with `--config <path>` (use the
file from `%APPDATA%\SwOrchestrator\settings.json` after configuring once
in the GUI). CLI flags still override anything loaded from the file.

## Driving it from Jenkins

See `Jenkinsfile` in this folder for a working declarative pipeline that
runs the CLI in parallel across multiple VMs.

### One-time Jenkins setup per VM

The critical gotcha for SOLIDWORKS on Jenkins: **the Jenkins agent must
not run as a Windows service.** Services run in Session 0 with no
interactive desktop, and SOLIDWORKS will not start there.

The standard workaround on each VM:

1. Create or pick a dedicated build user account with permission to run
   SOLIDWORKS.
2. Enable **auto-login** for that user (Group Policy or `netplwiz`).
3. In Task Scheduler, create a task with trigger **"At log on of <user>"**
   that runs:
   ```
   java -jar agent.jar -url http://<jenkins>/ -secret <SECRET> -name <node-name> -workDir C:\jenkins
   ```
   The exact command (with the secret) is shown on each node's status page
   in Jenkins under "Run from agent command line".
4. Verify the agent comes online (green dot) in Manage Jenkins -> Nodes
   after the next login.
5. Label each VM uniquely (e.g. `solidworks-vm1`, `solidworks-vm2`) so the
   `agent { label '...' }` blocks in the Jenkinsfile route work correctly.
6. Install **.NET 8 Desktop Runtime** on each VM.
7. Install **Python 3** on each VM and make sure `python` is on PATH.
   `winget install Python.Python.3.12` is the fastest path. Python is
   only used by `stage_shard.py` to distribute STEP files; the
   orchestrator itself is .NET and doesn't need it at runtime.
8. Copy `SwOrchestrator.Cli.exe` (and the macro) to a known location on
   each VM. `stage_shard.py` is checked out into `%WORKSPACE%` by
   Jenkins if you use "Pipeline script from SCM"; otherwise copy it to
   a known path too and update `STAGE_SHARD_PY` in the Jenkinsfile.

### Splitting STEP files across VMs (hash sharding)

The bundled Jenkinsfile uses **deterministic hash sharding** so each VM
processes a stable, disjoint slice of one master STEP folder. You don't
have to pre-split files manually: point all VMs at the same source via
the `SOURCE_STEPS` parameter and `stage_shard.py` handles distribution.

How it works:

- `stage_shard.py` enumerates `SOURCE_STEPS`, hashes each filename with
  MD5, mods that hash by `TOTAL_SHARDS`, and stages files where
  `hash mod TOTAL_SHARDS == shard_index` into the local `STEPS_FOLDER`.
- The hash is stable across runs and across VMs - the same file always
  goes to the same shard. Adding files to the master folder picks them up
  on the next run, on whichever VM their hash routes them to.
- Files are placed via Windows hardlinks (instant, zero disk cost). If
  `SOURCE_STEPS` is on a different volume than `STEPS_FOLDER`,
  `stage_shard.py` falls back to copy.
- Files already in the target folder are skipped, so re-staging on every
  build is cheap.
- The Jenkinsfile has one stage per VM, each calling `stageShard(N)`
  with a literal shard index (0, 1, 2, ...) before invoking
  `runOrchestrator()`. To scale: add a new stage block with the next
  index, increment `TOTAL_SHARDS` to match.

**Important: don't change `TOTAL_SHARDS` mid-batch.** The hash is modular,
so changing N reassigns every file to a different shard. Pick a value when
you set up the fleet and stick with it. If you genuinely need to resize
the fleet, finish the current batch first.

If you'd rather pre-split manually (more robust against accidentally
double-claiming files - the staging script trusts itself but a stray
filesystem operation could in theory put the same file in two STEPS
folders), point each VM's `STEPS_FOLDER` at its own pre-populated folder
and remove the `stageShard()` calls from each VM stage. The orchestrator
doesn't care how files arrived in `STEPS_FOLDER`.

### Sharing one network folder directly (NOT recommended)

For a single shared network folder accessed by all VMs simultaneously,
you'd need a claim mechanism (e.g. each VM atomically renames a file
before processing). That's not built in - keep it simple with hash
sharding or disjoint local folders unless / until you outgrow that.

### What Jenkins gives you, vs what the orchestrator gives you

| Layer | Responsibility |
| --- | --- |
| Jenkins | Fleet view, parallel scheduling, log archive, notifications, retry of whole jobs, manual triggering, parameterised runs |
| `SwOrchestrator.Cli` | Single-VM SOLIDWORKS babysitting: launch, watch, restart on crash, blacklist poison files, resume on next launch |

These compose - they don't overlap. The GUI is still useful for hands-on
debugging when you RDP into a VM to investigate something Jenkins flagged
red.

## Recommended: quiet Windows Error Reporting

When SOLIDWORKS crashes Windows can pop a "SOLIDWORKS has stopped working"
dialog and briefly block teardown. The orchestrator kills `WerFault.exe`
as part of cleanup, so this isn't fatal, but suppressing the dialog
system-wide is cleaner. In `regedit` under
`HKLM\SOFTWARE\Microsoft\Windows\Windows Error Reporting`:

- `DontShowUI` = `1` (DWORD) - recommended
- `Disabled` = `1` (DWORD) - optional, fully disables WER

Admin rights required. Skip this if IT policy says no.

## Tuning

Defaults err on the conservative side:

| Setting | Default | Meaning |
| --- | --- | --- |
| `--stall-timeout` | 900 s | After the first file, hang detected if no progress this long |
| `--startup-grace` | 300 s | Longer window for the very first file (SOLIDWORKS launch) |
| `--poll-interval` | 5 s | How often the watcher wakes up |
| `--cooldown` | 8 s | Wait between killing leftovers and the next launch |
| `--crash-threshold` | 3 | Auto-blacklist a file after N crashes blamed on it |
| `--max-restarts` | 10000 | Hard ceiling on the restart loop |

"Progress" counts either as a new SLDPRT file on disk or a heartbeat
update from the macro, so a single slow file doesn't trigger a false
stall.

## Troubleshooting

**SOLIDWORKS opens but the macro doesn't run.** The macro path must point
to a `.swp` file, not `.bas`. The `/m` switch only runs compiled `.swp`s.
Re-save in the VBA editor.

**Macro status stuck on "(no heartbeat yet)".** The macro's
`HEARTBEAT_FILE` constant doesn't match the orchestrator's status folder.
Edit the .bas, fix the path, re-save the .swp, and update the
orchestrator's `--status` / Status folder accordingly.

**SOLIDWORKS shows a license / login / "what's new" dialog at startup.**
The orchestrator can't click it. Open SOLIDWORKS once manually to clear
the dialog, close it, then re-run.

**Same file keeps getting blacklisted but isn't actually broken.**
Something else is making SOLIDWORKS crash and that file just happens to
be at the head of the queue. Edit `skip_files.txt` in the status folder
to remove blacklist entries; the orchestrator will re-attempt them.

**"SOLIDWORKS exe not found" at start.** The default path uses the
unversioned `SOLIDWORKS` folder. Some installers put it under
`SOLIDWORKS 2024\` - use Browse (GUI) or `--solidworks` (CLI).

**Jenkins agent appears offline after VM reboot.** Auto-login or the
Task Scheduler trigger isn't working. Log in manually once; if the agent
comes online then, the auto-login setup needs fixing (often a missed
"Users must enter a username and password" checkbox in `netplwiz`).

**`bat` step in Jenkinsfile fails with "command not found".** Wrap the
exe path in double quotes inside the triple-quoted Groovy string (already
done in the sample). The `^` characters at line ends are cmd.exe's line
continuations, not Groovy's - leave them.
