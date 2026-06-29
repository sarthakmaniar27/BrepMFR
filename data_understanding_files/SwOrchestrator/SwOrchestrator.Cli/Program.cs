using System.IO;
using System.Text.Json;
using SwOrchestrator;

internal static class Program
{
    // Exit codes used by the CLI. Jenkins (and shells) use these to decide
    // pass / fail / retry behaviour.
    private const int ExitSuccess      = 0;
    private const int ExitFailed       = 1;
    private const int ExitCancelled    = 2;
    private const int ExitBadArguments = 3;

    // Last reported run state - drives the final exit code.
    private static RunState s_finalState = RunState.Idle;
    private static CancellationTokenSource? s_cts;
    private static int s_ctrlCCount;

    static async Task<int> Main(string[] args)
    {
        try
        {
            if (args.Length == 0 || HasHelpFlag(args))
            {
                PrintUsage();
                return args.Length == 0 ? ExitBadArguments : ExitSuccess;
            }

            AppSettings settings;
            try { settings = ParseArgs(args); }
            catch (ArgumentException ex)
            {
                Console.Error.WriteLine($"Argument error: {ex.Message}");
                Console.Error.WriteLine();
                PrintUsage();
                return ExitBadArguments;
            }

            // Echo effective settings so the build log self-documents what was run.
            PrintHeader(settings);

            // Hook Ctrl-C so Jenkins (or a human) can stop the run gracefully.
            s_cts = new CancellationTokenSource();
            Console.CancelKeyPress += OnCancelKeyPress;

            // Synchronous progress reporter - events print in order, no thread-pool reshuffling.
            var reporter = new SyncProgress<OrchestratorEvent>(HandleEvent);
            var orch = new Orchestrator(settings, reporter);

            await orch.RunAsync(s_cts.Token);

            // Map the final run state to an exit code.
            return s_finalState switch
            {
                RunState.Complete => ExitSuccess,
                RunState.Idle     => s_cts.IsCancellationRequested ? ExitCancelled : ExitFailed,
                RunState.Failed   => ExitFailed,
                _                 => ExitFailed,
            };
        }
        catch (OperationCanceledException)
        {
            // Should be caught inside the orchestrator, but just in case.
            Console.Error.WriteLine("Cancelled.");
            return ExitCancelled;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Fatal: {ex.GetType().Name}: {ex.Message}");
            Console.Error.WriteLine(ex.StackTrace);
            return ExitFailed;
        }
    }

    // -- Arg parsing -----------------------------------------------------

    private static AppSettings ParseArgs(string[] args)
    {
        // Defaults come from AppSettings, optionally overridden by --config FILE,
        // and finally by individual CLI flags.
        var settings = new AppSettings();

        // First pass: --config gets applied before other flags so subsequent
        // flags can still override it.
        for (int i = 0; i < args.Length - 1; i++)
        {
            if (args[i] is "--config" or "-c")
            {
                var path = args[i + 1];
                if (!File.Exists(path))
                    throw new ArgumentException($"Config file not found: {path}");
                try
                {
                    var json = File.ReadAllText(path);
                    var loaded = JsonSerializer.Deserialize<AppSettings>(json);
                    if (loaded != null) settings = loaded;
                }
                catch (Exception ex)
                {
                    throw new ArgumentException($"Could not parse config file: {ex.Message}");
                }
                break;
            }
        }

        for (int i = 0; i < args.Length; i++)
        {
            string a = args[i];
            switch (a)
            {
                case "--solidworks": case "--sw":
                    settings.SolidWorksExe = RequireValue(args, ref i, a); break;
                case "--macro": case "-m":
                    settings.MacroPath = RequireValue(args, ref i, a); break;
                case "--steps": case "-s":
                    settings.StepsFolder = RequireValue(args, ref i, a); break;
                case "--status":
                    settings.StatusFolder = RequireValue(args, ref i, a); break;
                case "--variations": case "-n":
                    settings.NumVariations = ParseInt(RequireValue(args, ref i, a), a); break;
                case "--stall-timeout":
                    settings.StallTimeoutSeconds = ParseInt(RequireValue(args, ref i, a), a); break;
                case "--startup-grace":
                    settings.StartupGraceSeconds = ParseInt(RequireValue(args, ref i, a), a); break;
                case "--poll-interval":
                    settings.PollIntervalSeconds = ParseInt(RequireValue(args, ref i, a), a); break;
                case "--cooldown":
                    settings.CooldownSeconds = ParseInt(RequireValue(args, ref i, a), a); break;
                case "--crash-threshold":
                    settings.SameFileCrashThreshold = ParseInt(RequireValue(args, ref i, a), a); break;
                case "--max-restarts":
                    settings.MaxTotalRestarts = ParseInt(RequireValue(args, ref i, a), a); break;
                case "--config": case "-c":
                    // Already handled in first pass, just skip the value.
                    i++; break;
                case "--help": case "-h": case "/?":
                    // Handled by caller; ignore here.
                    break;
                default:
                    if (a.StartsWith("-"))
                        throw new ArgumentException($"Unknown option: {a}");
                    throw new ArgumentException($"Unexpected positional argument: {a}");
            }
        }

        return settings;
    }

    private static string RequireValue(string[] args, ref int i, string optName)
    {
        if (i + 1 >= args.Length)
            throw new ArgumentException($"Option '{optName}' requires a value.");
        return args[++i];
    }

    private static int ParseInt(string value, string optName)
    {
        if (!int.TryParse(value, out var n) || n < 0)
            throw new ArgumentException($"Option '{optName}' expects a non-negative integer, got '{value}'.");
        return n;
    }

    private static bool HasHelpFlag(string[] args)
        => args.Any(a => a is "--help" or "-h" or "/?");

    // -- Output ----------------------------------------------------------

    private static void PrintHeader(AppSettings s)
    {
        Console.WriteLine("=== SOLIDWORKS Thread Orchestrator (CLI) ===");
        Console.WriteLine($"  SolidWorks exe : {s.SolidWorksExe}");
        Console.WriteLine($"  Macro          : {s.MacroPath}");
        Console.WriteLine($"  STEPS folder   : {s.StepsFolder}");
        Console.WriteLine($"  Status folder  : {s.StatusFolder}");
        Console.WriteLine($"  Variations/file: {s.NumVariations}");
        Console.WriteLine($"  Stall timeout  : {s.StallTimeoutSeconds}s   Startup grace: {s.StartupGraceSeconds}s   Poll: {s.PollIntervalSeconds}s");
        Console.WriteLine($"  Cooldown       : {s.CooldownSeconds}s   Crash threshold: {s.SameFileCrashThreshold}   Max restarts: {s.MaxTotalRestarts}");
        Console.WriteLine("============================================");
    }

    private static void HandleEvent(OrchestratorEvent ev)
    {
        var ts = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");
        switch (ev)
        {
            case LogEvent log:
                var levelTag = log.Level switch
                {
                    LogLevel.Debug => "DEBUG",
                    LogLevel.Info  => "INFO ",
                    LogLevel.Warn  => "WARN ",
                    LogLevel.Error => "ERROR",
                    _              => "INFO ",
                };
                var sink = log.Level >= LogLevel.Warn ? Console.Error : Console.Out;
                sink.WriteLine($"[{ts}] [{levelTag}] {log.Message}");
                break;

            case ProgressEvent pe:
                var pct = pe.Total == 0 ? 0 : 100.0 * pe.Done / pe.Total;
                Console.WriteLine($"[{ts}] [PROG ] {pe.Done} / {pe.Total} ({pct:F1}%)");
                break;

            case HeartbeatEvent hb:
                var fname = string.IsNullOrEmpty(hb.StepFile)
                    ? "(none)"
                    : Path.GetFileName(hb.StepFile);
                Console.WriteLine($"[{ts}] [HB   ] status={hb.Status} file={fname}");
                break;

            case RunStateEvent rs:
                s_finalState = rs.State;
                Console.WriteLine($"[{ts}] [STATE] {rs.State}");
                break;

            case SkipListEvent sl:
                Console.WriteLine($"[{ts}] [SKIP ] Skip list now has {sl.CurrentSkipList.Count} entr{(sl.CurrentSkipList.Count == 1 ? "y" : "ies")}");
                break;
        }

        // Flush so Jenkins / piped consumers see lines immediately.
        Console.Out.Flush();
        Console.Error.Flush();
    }

    private static void OnCancelKeyPress(object? sender, ConsoleCancelEventArgs e)
    {
        s_ctrlCCount++;
        if (s_ctrlCCount >= 2)
        {
            // Let the second Ctrl-C through - kills hard.
            Console.Error.WriteLine();
            Console.Error.WriteLine("[CTRL-C] Force-exiting.");
            return;
        }
        Console.Error.WriteLine();
        Console.Error.WriteLine("[CTRL-C] Stopping. Press Ctrl-C again to force quit.");
        e.Cancel = true;          // don't terminate immediately
        try { s_cts?.Cancel(); }
        catch { /* already disposed */ }
    }

    private static void PrintUsage()
    {
        Console.WriteLine("SwOrchestrator.Cli - runs the SOLIDWORKS thread macro across a folder of STEP files,");
        Console.WriteLine("                     restarting SOLIDWORKS automatically on crashes or hangs.");
        Console.WriteLine();
        Console.WriteLine("Usage:");
        Console.WriteLine("  SwOrchestrator.Cli.exe --steps <dir> --macro <swp> --status <dir>");
        Console.WriteLine("                         [--solidworks <exe>] [--config <json>] [tunables...]");
        Console.WriteLine();
        Console.WriteLine("Required paths:");
        Console.WriteLine("  --steps,      -s  <dir>   Folder full of .step / .stp files");
        Console.WriteLine("  --macro,      -m  <swp>   Compiled SOLIDWORKS macro (.swp)");
        Console.WriteLine("  --status          <dir>   Folder for heartbeat / done marker / skip list");
        Console.WriteLine("                            (must match STATUS_FOLDER inside the macro source)");
        Console.WriteLine();
        Console.WriteLine("Optional:");
        Console.WriteLine("  --solidworks, --sw <exe>  Path to SLDWORKS.exe");
        Console.WriteLine("                            (default: C:\\Program Files\\SOLIDWORKS Corp\\SOLIDWORKS\\SLDWORKS.exe)");
        Console.WriteLine("  --config,     -c  <json>  Load defaults from a settings JSON file (CLI flags still override)");
        Console.WriteLine("  --variations, -n  <int>   Variations per file (default 6)");
        Console.WriteLine("  --stall-timeout   <sec>   Hang threshold after first file finishes (default 900)");
        Console.WriteLine("  --startup-grace   <sec>   Hang threshold for the very first file (default 300)");
        Console.WriteLine("  --poll-interval   <sec>   How often to check for progress (default 5)");
        Console.WriteLine("  --cooldown        <sec>   Wait between kill and re-launch (default 8)");
        Console.WriteLine("  --crash-threshold <int>   Blacklist a file after N crashes blamed on it (default 3)");
        Console.WriteLine("  --max-restarts    <int>   Hard ceiling on the restart loop (default 10000)");
        Console.WriteLine("  --help, -h, /?            Show this message");
        Console.WriteLine();
        Console.WriteLine("Exit codes:");
        Console.WriteLine($"  {ExitSuccess}  Batch completed (all files have all variation outputs)");
        Console.WriteLine($"  {ExitFailed}  Failure (validation error, max restarts exceeded, unhandled exception)");
        Console.WriteLine($"  {ExitCancelled}  Cancelled by Ctrl-C / signal");
        Console.WriteLine($"  {ExitBadArguments}  Bad command-line arguments");
    }
}

/// <summary>
/// A synchronous <see cref="IProgress{T}"/> implementation. The default
/// <see cref="Progress{T}"/> queues callbacks on the captured
/// SynchronizationContext (or thread pool when none exists), which can
/// reorder console output. For a CLI we want each Report to write to stdout
/// in-order on the caller's thread.
/// </summary>
internal sealed class SyncProgress<T> : IProgress<T>
{
    private readonly Action<T> _handler;
    public SyncProgress(Action<T> handler) => _handler = handler;
    public void Report(T value) => _handler(value);
}
