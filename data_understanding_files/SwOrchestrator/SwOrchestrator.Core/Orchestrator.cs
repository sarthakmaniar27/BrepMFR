using System.Diagnostics;
using System.IO;

namespace SwOrchestrator;

/// <summary>
/// Drives the SOLIDWORKS macro through a folder of STEP files, restarting on
/// crash or hang. UI gets updates via an IProgress&lt;OrchestratorEvent&gt;.
/// </summary>
public sealed class Orchestrator
{
    private readonly AppSettings _cfg;
    private readonly IProgress<OrchestratorEvent> _reporter;

    /// <summary>Per-file crash counts for the auto-blacklist.</summary>
    private readonly Dictionary<string, int> _crashesPerFile =
        new(StringComparer.OrdinalIgnoreCase);

    private int _progressAtAttemptStart;

    public Orchestrator(AppSettings cfg, IProgress<OrchestratorEvent> reporter)
    {
        _cfg = cfg;
        _reporter = reporter;
    }

    // -- Main loop --------------------------------------------------------

    public async Task RunAsync(CancellationToken ct)
    {
        try
        {
            _reporter.Report(new RunStateEvent(RunState.Running));

            if (!ValidatePreFlight()) { _reporter.Report(new RunStateEvent(RunState.Failed)); return; }

            // Clear any stale done marker
            try
            {
                if (File.Exists(_cfg.DoneMarker))
                {
                    File.Delete(_cfg.DoneMarker);
                    Log(LogLevel.Info, "Cleared stale batch_done.marker from previous run.");
                }
            }
            catch (Exception ex) { Log(LogLevel.Warn, $"Couldn't delete done marker: {ex.Message}"); }

            // Make sure status folder exists
            Directory.CreateDirectory(_cfg.StatusFolder);

            // Initial progress snapshot
            var (done, total) = CountProcessed();
            _reporter.Report(new ProgressEvent(done, total));
            Log(LogLevel.Info, $"Starting. {done} / {total} STEP files already complete.");

            if (total == 0)
            {
                Log(LogLevel.Warn, $"No STEP files found in {_cfg.StepsFolder}.");
                _reporter.Report(new RunStateEvent(RunState.Failed));
                return;
            }
            if (done >= total)
            {
                Log(LogLevel.Info, "Already complete - nothing to do.");
                _reporter.Report(new RunStateEvent(RunState.Complete));
                return;
            }

            for (int attempt = 1; attempt <= _cfg.MaxTotalRestarts; attempt++)
            {
                ct.ThrowIfCancellationRequested();
                Log(LogLevel.Info, $"=== Attempt {attempt} ===");

                (done, total) = CountProcessed();
                _progressAtAttemptStart = done;
                _reporter.Report(new ProgressEvent(done, total));
                Log(LogLevel.Info, $"Pre-launch progress: {done} / {total}");

                KillProcesses();
                await Task.Delay(_cfg.CooldownSeconds * 1000, ct);

                Process? proc = LaunchSolidWorks();
                if (proc == null)
                {
                    _reporter.Report(new RunStateEvent(RunState.Failed));
                    return;
                }
                Log(LogLevel.Info, $"Launched SOLIDWORKS pid={proc.Id}");

                var outcome = await WatchAsync(proc, ct);

                // Always tear down whatever's still alive
                try { proc.Refresh(); } catch { }
                KillProcesses();
                await Task.Delay(_cfg.CooldownSeconds * 1000, ct);

                if (outcome == WatchOutcome.BatchDone)
                {
                    (done, total) = CountProcessed();
                    _reporter.Report(new ProgressEvent(done, total));
                    Log(LogLevel.Info, $"Batch complete. Final: {done} / {total}.");
                    _reporter.Report(new RunStateEvent(RunState.Complete));
                    return;
                }

                MaybeBlacklistAfterCrash();

                (done, total) = CountProcessed();
                _reporter.Report(new ProgressEvent(done, total));
                Log(LogLevel.Info, $"Post-attempt progress: {done} / {total}.");

                if (done >= total)
                {
                    Log(LogLevel.Info, "All outputs present; treating as complete.");
                    _reporter.Report(new RunStateEvent(RunState.Complete));
                    return;
                }
            }

            Log(LogLevel.Error, $"Max restarts ({_cfg.MaxTotalRestarts}) reached. Giving up.");
            _reporter.Report(new RunStateEvent(RunState.Failed));
        }
        catch (OperationCanceledException)
        {
            Log(LogLevel.Warn, "Stop requested. Killing SOLIDWORKS.");
            KillProcesses();
            _reporter.Report(new RunStateEvent(RunState.Idle));
        }
        catch (Exception ex)
        {
            Log(LogLevel.Error, $"Unhandled exception: {ex.Message}");
            try { KillProcesses(); } catch { }
            _reporter.Report(new RunStateEvent(RunState.Failed));
        }
    }

    // -- Pre-flight -------------------------------------------------------

    private bool ValidatePreFlight()
    {
        var ok = true;
        if (!File.Exists(_cfg.SolidWorksExe))
        {
            Log(LogLevel.Error, $"SOLIDWORKS exe not found: {_cfg.SolidWorksExe}");
            ok = false;
        }
        if (!File.Exists(_cfg.MacroPath))
        {
            Log(LogLevel.Error, $"Macro file not found: {_cfg.MacroPath}");
            ok = false;
        }
        if (!Directory.Exists(_cfg.StepsFolder))
        {
            Log(LogLevel.Error, $"STEPS folder not found: {_cfg.StepsFolder}");
            ok = false;
        }
        return ok;
    }

    // -- One attempt's watch loop ----------------------------------------

    private enum WatchOutcome { BatchDone, SolidWorksExited, Stalled }

    private async Task<WatchOutcome> WatchAsync(Process proc, CancellationToken ct)
    {
        var attemptStart = DateTime.UtcNow;
        var (lastDone, _) = CountProcessed();
        var lastProgressTime = attemptStart;
        var lastHbMtime = SafeHeartbeatMtime();

        while (true)
        {
            // Throws OperationCanceledException (caught by outer RunAsync catch)
            await Task.Delay(_cfg.PollIntervalSeconds * 1000, ct);

            // Batch done marker beats everything else
            if (File.Exists(_cfg.DoneMarker))
            {
                Log(LogLevel.Info, "batch_done.marker detected.");
                return WatchOutcome.BatchDone;
            }

            // Did SOLIDWORKS die on us?
            try { proc.Refresh(); } catch { }
            if (proc.HasExited)
            {
                int code;
                try { code = proc.ExitCode; } catch { code = -1; }
                Log(LogLevel.Warn, $"SOLIDWORKS process exited (code={code}).");
                return WatchOutcome.SolidWorksExited;
            }

            // Filesystem & heartbeat signals
            var now = DateTime.UtcNow;
            var (done, total) = CountProcessed();
            var hbMtime = SafeHeartbeatMtime();

            bool fsMoved = done > lastDone;
            bool hbMoved = hbMtime.HasValue && lastHbMtime.HasValue && hbMtime > lastHbMtime;

            if (fsMoved)
            {
                _reporter.Report(new ProgressEvent(done, total));
                Log(LogLevel.Info, $"Progress: {done} / {total} (+{done - lastDone})");
            }

            // Push any heartbeat update to the UI (even without filesystem progress)
            if (hbMtime != lastHbMtime)
            {
                var hb = ReadHeartbeat();
                if (hb != null)
                {
                    _reporter.Report(new HeartbeatEvent(hb.Value.status, hb.Value.stepFile, hb.Value.timestamp));
                }
            }

            if (fsMoved || hbMoved)
            {
                lastDone = done;
                lastProgressTime = now;
                lastHbMtime = hbMtime;
            }

            // Stall check. Use the longer grace period until the first file
            // finishes after launch.
            var sinceProgress = (now - lastProgressTime).TotalSeconds;
            int graceSec = (lastDone == _progressAtAttemptStart)
                ? _cfg.StartupGraceSeconds
                : _cfg.StallTimeoutSeconds;

            if (sinceProgress > graceSec)
            {
                Log(LogLevel.Warn,
                    $"No progress for {sinceProgress:F0}s (grace={graceSec}s). Declaring stall.");
                return WatchOutcome.Stalled;
            }
        }
    }

    // -- Filesystem helpers ----------------------------------------------

    /// <summary>Returns (done, total) counting STEP files whose 6 SLDPRTs all exist.</summary>
    private (int done, int total) CountProcessed()
    {
        if (!Directory.Exists(_cfg.StepsFolder)) return (0, 0);

        var files = ListStepFiles();
        int done = 0;
        foreach (var f in files)
            if (IsStepFullyProcessed(f)) done++;

        return (done, files.Count);
    }

    private IReadOnlyList<string> ListStepFiles()
    {
        // Case-insensitive enumeration for both .step and .stp
        var all = Directory.EnumerateFiles(_cfg.StepsFolder, "*.*", SearchOption.TopDirectoryOnly)
            .Where(p =>
            {
                var ext = Path.GetExtension(p);
                return ext.Equals(".step", StringComparison.OrdinalIgnoreCase)
                    || ext.Equals(".stp",  StringComparison.OrdinalIgnoreCase);
            });
        return all.OrderBy(p => p, StringComparer.OrdinalIgnoreCase).ToList();
    }

    private string VariationPath(string stepPath, int v)
    {
        var dir  = Path.GetDirectoryName(stepPath) ?? string.Empty;
        var stem = Path.GetFileNameWithoutExtension(stepPath);
        return Path.Combine(dir, $"{stem}_{v}.SLDPRT");
    }

    private bool IsStepFullyProcessed(string stepPath)
    {
        for (int v = 1; v <= _cfg.NumVariations; v++)
            if (!File.Exists(VariationPath(stepPath, v))) return false;
        return true;
    }

    // -- Heartbeat parsing -----------------------------------------------

    private DateTime? SafeHeartbeatMtime()
    {
        try
        {
            if (!File.Exists(_cfg.HeartbeatFile)) return null;
            return File.GetLastWriteTimeUtc(_cfg.HeartbeatFile);
        }
        catch { return null; }
    }

    private (string status, string stepFile, DateTime timestamp)? ReadHeartbeat()
    {
        try
        {
            if (!File.Exists(_cfg.HeartbeatFile)) return null;
            // The macro overwrites this file; read with shared read since the macro
            // may be re-opening it concurrently.
            string text;
            using (var fs = new FileStream(_cfg.HeartbeatFile, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
            using (var sr = new StreamReader(fs))
            {
                text = sr.ReadToEnd().Trim();
            }
            if (string.IsNullOrEmpty(text)) return null;

            var parts = text.Split('|', 3);
            if (parts.Length < 3) return null;

            DateTime ts;
            if (!DateTime.TryParse(parts[0].Trim(), out ts))
                ts = DateTime.Now;

            return (parts[1].Trim(), parts[2].Trim(), ts);
        }
        catch { return null; }
    }

    // -- Crash attribution & blacklist -----------------------------------

    /// <summary>
    /// If the macro was actively working on a file when SOLIDWORKS died, count
    /// the crash against that file. After N consecutive crashes blamed on it,
    /// append it to skip_files.txt so the macro skips it on the next launch.
    /// </summary>
    private void MaybeBlacklistAfterCrash()
    {
        var culprit = FileBeingProcessedAtCrash();
        if (string.IsNullOrEmpty(culprit)) return;

        var key = culprit!.ToLowerInvariant();
        _crashesPerFile.TryGetValue(key, out var n);
        n++;
        _crashesPerFile[key] = n;

        Log(LogLevel.Warn, $"Crash attributed to '{culprit}' (#{n} crash on this file).");

        if (n >= _cfg.SameFileCrashThreshold)
        {
            Log(LogLevel.Error,
                $"Adding '{culprit}' to skip list after {n} crashes.");
            AppendToSkipList(culprit, $"auto-blacklisted after {n} consecutive crashes");
            _reporter.Report(new SkipListEvent(ReadSkipList()));
        }
    }

    /// <summary>The STEP file name the macro most recently said it was actively on.</summary>
    private string? FileBeingProcessedAtCrash()
    {
        var hb = ReadHeartbeat();
        if (hb == null) return null;
        var status = hb.Value.status;

        // Only "active" statuses indicate the macro was mid-work on a file
        if (!status.StartsWith("file_start", StringComparison.Ordinal)
            && !status.StartsWith("var_start_", StringComparison.Ordinal)
            && !status.StartsWith("var_done_",  StringComparison.Ordinal))
        {
            return null;
        }

        var step = hb.Value.stepFile;
        return string.IsNullOrEmpty(step) ? null : Path.GetFileName(step);
    }

    private void AppendToSkipList(string basename, string reason)
    {
        Directory.CreateDirectory(_cfg.StatusFolder);
        var existing = ReadSkipList();
        if (existing.Any(s => s.Equals(basename, StringComparison.OrdinalIgnoreCase))) return;

        using var sw = new StreamWriter(_cfg.SkipListFile, append: true);
        sw.WriteLine($"# {DateTime.Now:yyyy-MM-ddTHH:mm:ss}  {reason}");
        sw.WriteLine(basename);
    }

    public IReadOnlyList<string> ReadSkipList()
    {
        if (!File.Exists(_cfg.SkipListFile)) return Array.Empty<string>();
        try
        {
            return File.ReadAllLines(_cfg.SkipListFile)
                .Select(line => line.Trim())
                .Where(line => line.Length > 0 && !line.StartsWith("#"))
                .ToList();
        }
        catch { return Array.Empty<string>(); }
    }

    // -- Process management ----------------------------------------------

    private Process? LaunchSolidWorks()
    {
        try
        {
            var psi = new ProcessStartInfo
            {
                FileName = _cfg.SolidWorksExe,
                Arguments = $"/m \"{_cfg.MacroPath}\"",
                UseShellExecute = false,
                CreateNoWindow = false,
            };
            return Process.Start(psi);
        }
        catch (Exception ex)
        {
            Log(LogLevel.Error, $"Failed to launch SOLIDWORKS: {ex.Message}");
            return null;
        }
    }

    private void KillProcesses()
    {
        var subs = _cfg.ProcessKillSubstrings;
        if (subs.Count == 0) return;

        Process[] all;
        try { all = Process.GetProcesses(); }
        catch { return; }

        var killed = new List<string>();
        foreach (var p in all)
        {
            string name;
            try { name = p.ProcessName; }
            catch { continue; }

            bool match = false;
            foreach (var s in subs)
            {
                if (name.IndexOf(s, StringComparison.OrdinalIgnoreCase) >= 0)
                {
                    match = true;
                    break;
                }
            }
            if (!match) continue;

            try
            {
                p.Kill(entireProcessTree: true);
                killed.Add(name);
            }
            catch { /* already gone, access denied, etc. */ }
            finally
            {
                try { p.Dispose(); } catch { }
            }
        }
        if (killed.Count > 0)
        {
            var unique = killed.Distinct(StringComparer.OrdinalIgnoreCase);
            Log(LogLevel.Info, $"Killed leftover process(es): {string.Join(", ", unique)}");
        }
    }

    // -- Misc ------------------------------------------------------------

    private void Log(LogLevel level, string message)
        => _reporter.Report(new LogEvent(level, message));
}
