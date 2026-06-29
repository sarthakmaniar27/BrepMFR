using System.Collections.ObjectModel;
using System.ComponentModel;
using System.IO;
using System.Runtime.CompilerServices;
using System.Windows.Media;

namespace SwOrchestrator;

public sealed class MainViewModel : INotifyPropertyChanged
{
    public AppSettings Settings { get; }

    public ObservableCollection<LogEntry> LogEntries { get; } = new();
    public ObservableCollection<string>   SkipListItems { get; } = new();

    public MainViewModel()
    {
        Settings = AppSettings.LoadOrDefault();
        UpdateInitialProgress();
        RefreshSkipList();
    }

    // -- Run lifecycle ----------------------------------------------------

    private CancellationTokenSource? _cts;
    private Task? _runTask;

    public bool IsRunning => _runTask is { IsCompleted: false };

    private RunState _runState = RunState.Idle;
    public RunState RunState
    {
        get => _runState;
        private set
        {
            if (_runState == value) return;
            _runState = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(RunStateText));
            OnPropertyChanged(nameof(RunStateBrush));
            OnPropertyChanged(nameof(IsStartEnabled));
            OnPropertyChanged(nameof(IsStopEnabled));
            OnPropertyChanged(nameof(IsConfigEditable));
        }
    }

    public string RunStateText => _runState switch
    {
        RunState.Idle     => "Idle",
        RunState.Running  => "Running",
        RunState.Stopping => "Stopping…",
        RunState.Complete => "Complete",
        RunState.Failed   => "Failed",
        _ => _runState.ToString(),
    };

    public Brush RunStateBrush => _runState switch
    {
        RunState.Running  => Brushes.SeaGreen,
        RunState.Stopping => Brushes.DarkOrange,
        RunState.Complete => Brushes.RoyalBlue,
        RunState.Failed   => Brushes.Crimson,
        _ => Brushes.Gray,
    };

    public bool IsStartEnabled    => _runState is RunState.Idle or RunState.Complete or RunState.Failed;
    public bool IsStopEnabled     => _runState is RunState.Running;
    public bool IsConfigEditable  => _runState is RunState.Idle or RunState.Complete or RunState.Failed;

    public void Start()
    {
        if (IsRunning) return;

        try { Settings.Save(); } catch { /* best effort */ }

        _cts = new CancellationTokenSource();
        var progress = new Progress<OrchestratorEvent>(HandleEvent);
        var orch = new Orchestrator(Settings, progress);
        var ct = _cts.Token;

        // The Progress<T> was created on the UI thread, so callbacks marshal
        // back here automatically. The orchestrator itself runs on a worker.
        _runTask = Task.Run(() => orch.RunAsync(ct), ct);
    }

    public void Stop()
    {
        if (!IsRunning) return;
        RunState = RunState.Stopping;
        try { _cts?.Cancel(); } catch { }
    }

    public void ResetState()
    {
        // Clear orchestrator status files. Does NOT delete SLDPRT outputs or skip list.
        try
        {
            if (File.Exists(Settings.HeartbeatFile)) File.Delete(Settings.HeartbeatFile);
            if (File.Exists(Settings.DoneMarker))   File.Delete(Settings.DoneMarker);
            LogEntries.Add(new LogEntry(LogLevel.Info, "Cleared heartbeat and done marker."));
        }
        catch (Exception ex)
        {
            LogEntries.Add(new LogEntry(LogLevel.Warn, $"Reset failed: {ex.Message}"));
        }
        UpdateInitialProgress();
        RefreshSkipList();
    }

    // -- Event handler ---------------------------------------------------

    private const int MaxLogEntries = 2000;

    private void HandleEvent(OrchestratorEvent ev)
    {
        switch (ev)
        {
            case LogEvent log:
                LogEntries.Add(new LogEntry(log.Level, log.Message));
                while (LogEntries.Count > MaxLogEntries) LogEntries.RemoveAt(0);
                break;

            case ProgressEvent pe:
                FilesDone = pe.Done;
                FilesTotal = pe.Total;
                break;

            case HeartbeatEvent hb:
                LastHeartbeatStatus = hb.Status;
                LastHeartbeatStepFile = string.IsNullOrEmpty(hb.StepFile)
                    ? "(none)"
                    : Path.GetFileName(hb.StepFile);
                LastHeartbeatTime = hb.Timestamp;
                break;

            case RunStateEvent rs:
                RunState = rs.State;
                if (rs.State is RunState.Complete or RunState.Failed or RunState.Idle)
                    UpdateInitialProgress();
                break;

            case SkipListEvent sl:
                SkipListItems.Clear();
                foreach (var item in sl.CurrentSkipList) SkipListItems.Add(item);
                break;
        }
    }

    // -- Progress bound properties ---------------------------------------

    private int _filesDone;
    public int FilesDone
    {
        get => _filesDone;
        set { if (_filesDone == value) return; _filesDone = value; OnPropertyChanged(); OnPropertyChanged(nameof(ProgressPercent)); OnPropertyChanged(nameof(ProgressText)); }
    }

    private int _filesTotal;
    public int FilesTotal
    {
        get => _filesTotal;
        set { if (_filesTotal == value) return; _filesTotal = value; OnPropertyChanged(); OnPropertyChanged(nameof(ProgressPercent)); OnPropertyChanged(nameof(ProgressText)); }
    }

    public double ProgressPercent => _filesTotal == 0 ? 0 : 100.0 * _filesDone / _filesTotal;

    public string ProgressText =>
        _filesTotal == 0
            ? "No files found"
            : $"{_filesDone} / {_filesTotal}  ({ProgressPercent:F1}%)";

    // -- Heartbeat bound properties --------------------------------------

    private string _lastHeartbeatStatus = "(no heartbeat yet)";
    public string LastHeartbeatStatus
    {
        get => _lastHeartbeatStatus;
        set { if (_lastHeartbeatStatus == value) return; _lastHeartbeatStatus = value; OnPropertyChanged(); }
    }

    private string _lastHeartbeatStepFile = "—";
    public string LastHeartbeatStepFile
    {
        get => _lastHeartbeatStepFile;
        set { if (_lastHeartbeatStepFile == value) return; _lastHeartbeatStepFile = value; OnPropertyChanged(); }
    }

    private DateTime? _lastHeartbeatTime;
    public DateTime? LastHeartbeatTime
    {
        get => _lastHeartbeatTime;
        set { _lastHeartbeatTime = value; OnPropertyChanged(); OnPropertyChanged(nameof(LastHeartbeatTimeText)); }
    }

    public string LastHeartbeatTimeText =>
        _lastHeartbeatTime.HasValue
            ? _lastHeartbeatTime.Value.ToString("HH:mm:ss")
            : "—";

    // -- Helpers ---------------------------------------------------------

    /// <summary>
    /// Re-count from disk and push to UI. Used at startup and after Reset to
    /// give an accurate baseline without needing to start a run.
    /// </summary>
    public void UpdateInitialProgress()
    {
        try
        {
            if (!Directory.Exists(Settings.StepsFolder))
            {
                FilesDone = 0; FilesTotal = 0;
                return;
            }
            var files = Directory.EnumerateFiles(Settings.StepsFolder, "*.*", SearchOption.TopDirectoryOnly)
                .Where(p =>
                {
                    var ext = Path.GetExtension(p);
                    return ext.Equals(".step", StringComparison.OrdinalIgnoreCase)
                        || ext.Equals(".stp",  StringComparison.OrdinalIgnoreCase);
                })
                .ToList();
            int done = 0;
            foreach (var f in files)
            {
                bool all = true;
                for (int v = 1; v <= Settings.NumVariations; v++)
                {
                    var stem = Path.GetFileNameWithoutExtension(f);
                    var dir = Path.GetDirectoryName(f) ?? string.Empty;
                    var vp = Path.Combine(dir, $"{stem}_{v}.SLDPRT");
                    if (!File.Exists(vp)) { all = false; break; }
                }
                if (all) done++;
            }
            FilesDone = done;
            FilesTotal = files.Count;
        }
        catch
        {
            // Ignore - probably bad path
            FilesDone = 0;
            FilesTotal = 0;
        }
    }

    public void RefreshSkipList()
    {
        SkipListItems.Clear();
        try
        {
            if (!File.Exists(Settings.SkipListFile)) return;
            foreach (var line in File.ReadAllLines(Settings.SkipListFile))
            {
                var s = line.Trim();
                if (s.Length > 0 && !s.StartsWith("#"))
                    SkipListItems.Add(s);
            }
        }
        catch { /* silent */ }
    }

    // -- INPC ------------------------------------------------------------

    public event PropertyChangedEventHandler? PropertyChanged;
    private void OnPropertyChanged([CallerMemberName] string? name = null)
        => PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
}
