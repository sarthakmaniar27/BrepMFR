# JSON_Generator_2027 + SwJsonExport (sibling babysitter)

Do **not** use `SwOrchestrator.Cli.exe` for this job. Keep your thread command
unchanged. Use `SwJsonExport.ps1` + an updated `.swp` compiled from
`JSON_Generator_2027.bas`.

## Paths (from the macro)

| Role | Path |
|------|------|
| Input SLDPRTs | `C:\Threads\conversion\sldprts` |
| Output JSONs | `C:\Threads\conversion\jsons` |
| UV JSONs | `C:\Threads\conversion\uv_jsons` |
| Status (heartbeat / skip / done) | `C:\Threads\conversion\status` |

Status is **not** `C:\Threads\status` (that belongs to the thread pipeline).

## One-time: rebuild the `.swp`

1. Open SOLIDWORKS → Tools → Macro → Edit…
2. Open your existing `JSON_Generator_2027.swp` (or any `.swp`).
3. Replace the module contents with `JSON_Generator_2027.bas` from this folder
   (or File → Import File… and import the `.bas`).
4. Tools → References: SldWorks Type Library + SldWorks Constant Type Library.
5. File → Save As → `C:\Threads\macro\JSON_Generator_2027.swp`
   (`/m` only runs `.swp`, not `.bas`).

## Run command

```bat
powershell -ExecutionPolicy Bypass -File "C:\path\to\SwJsonExport.ps1" ^
  -SwExe   "C:\images\image_08_03\WinRel64\sldworks.exe" ^
  -Macro   "C:\Threads\macro\JSON_Generator_2027.swp" ^
  -Parts   "C:\Threads\conversion\sldprts" ^
  -JsonOut "C:\Threads\conversion\jsons" ^
  -Status  "C:\Threads\conversion\status" ^
  -FailureThreshold 3 ^
  -StallTimeoutSec 900 ^
  -StartupGraceSec 600
```

Copy `SwJsonExport.ps1` next to the macro on the machine (or keep the full path).

## What changed vs your original VBA

- Removed `MsgBox` (blocks unattended runs)
- Heartbeat / skip list / `batch_done.marker` / `ExitApp` (babysitter contract)
- Leftover `in_progress.txt` → auto skip on next launch (hang recovery)
- Optional `MAX_SLDPRT_BYTES` (still `0` = no filter; set if you want to skip huge parts)
- Paths unchanged from your script

## Unchanged thread command (do not mix)

```bat
SwOrchestrator.Cli.exe --steps C:\Threads\cadsynth\cad_steps_filtered --macro C:\Threads\macro\threadplustextgen12.swp --status C:\Threads\status --sw C:\images\image_08_03\WinRel64\sldworks.exe --output C:\Threads\sldprts --failure-threshold 3
```
