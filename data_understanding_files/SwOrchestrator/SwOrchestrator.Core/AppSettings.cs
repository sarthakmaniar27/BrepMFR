using System.ComponentModel;
using System.IO;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace SwOrchestrator;

/// <summary>
/// User-editable configuration. Persisted as JSON in
/// %APPDATA%\SwOrchestrator\settings.json. The class implements INPC so the
/// UI updates immediately when settings change from Browse dialogs.
/// </summary>
public sealed class AppSettings : INotifyPropertyChanged
{
    // -- Path settings ----------------------------------------------------

    private string _solidWorksExe = @"C:\Program Files\SOLIDWORKS Corp\SOLIDWORKS\SLDWORKS.exe";
    public string SolidWorksExe
    {
        get => _solidWorksExe;
        set => SetField(ref _solidWorksExe, value);
    }

    private string _macroPath = @"C:\ThreadRecognition\ThreadCreationScript8.swp";
    public string MacroPath
    {
        get => _macroPath;
        set => SetField(ref _macroPath, value);
    }

    private string _stepsFolder = @"C:\ThreadRecognition\STEPS";
    public string StepsFolder
    {
        get => _stepsFolder;
        set => SetField(ref _stepsFolder, value);
    }

    private string _statusFolder = @"C:\ThreadRecognition";
    public string StatusFolder
    {
        get => _statusFolder;
        set => SetField(ref _statusFolder, value);
    }

    // -- Macro behavior ---------------------------------------------------

    private int _numVariations = 6;
    public int NumVariations
    {
        get => _numVariations;
        set => SetField(ref _numVariations, value);
    }

    // -- Crash/hang detection ---------------------------------------------

    private int _stallTimeoutSeconds = 900;
    public int StallTimeoutSeconds
    {
        get => _stallTimeoutSeconds;
        set => SetField(ref _stallTimeoutSeconds, value);
    }

    private int _startupGraceSeconds = 300;
    public int StartupGraceSeconds
    {
        get => _startupGraceSeconds;
        set => SetField(ref _startupGraceSeconds, value);
    }

    private int _pollIntervalSeconds = 5;
    public int PollIntervalSeconds
    {
        get => _pollIntervalSeconds;
        set => SetField(ref _pollIntervalSeconds, value);
    }

    private int _cooldownSeconds = 8;
    public int CooldownSeconds
    {
        get => _cooldownSeconds;
        set => SetField(ref _cooldownSeconds, value);
    }

    private int _sameFileCrashThreshold = 3;
    public int SameFileCrashThreshold
    {
        get => _sameFileCrashThreshold;
        set => SetField(ref _sameFileCrashThreshold, value);
    }

    private int _maxTotalRestarts = 10_000;
    public int MaxTotalRestarts
    {
        get => _maxTotalRestarts;
        set => SetField(ref _maxTotalRestarts, value);
    }

    // -- Process kill list ------------------------------------------------

    /// <summary>
    /// Case-insensitive substring matches against running process names.
    /// </summary>
    public List<string> ProcessKillSubstrings { get; set; } = new()
    {
        "sldworks",
        "werfault",
        "swspmanager",
        "swshellfileeventserver",
    };

    // -- Derived paths (not serialized) ----------------------------------

    [JsonIgnore]
    public string HeartbeatFile => Path.Combine(StatusFolder, "heartbeat.txt");

    [JsonIgnore]
    public string DoneMarker => Path.Combine(StatusFolder, "batch_done.marker");

    [JsonIgnore]
    public string SkipListFile => Path.Combine(StatusFolder, "skip_files.txt");

    // -- Persistence ------------------------------------------------------

    /// <summary>
    /// Location of the settings JSON file on disk.
    /// </summary>
    public static string SettingsPath
    {
        get
        {
            var appData = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            return Path.Combine(appData, "SwOrchestrator", "settings.json");
        }
    }

    public static AppSettings LoadOrDefault()
    {
        try
        {
            if (File.Exists(SettingsPath))
            {
                var json = File.ReadAllText(SettingsPath);
                var loaded = JsonSerializer.Deserialize<AppSettings>(json, JsonOpts);
                if (loaded != null) return loaded;
            }
        }
        catch
        {
            // Bad / missing / corrupt - fall through to defaults
        }
        return new AppSettings();
    }

    public void Save()
    {
        var dir = Path.GetDirectoryName(SettingsPath)!;
        Directory.CreateDirectory(dir);
        var json = JsonSerializer.Serialize(this, JsonOpts);
        File.WriteAllText(SettingsPath, json);
    }

    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        WriteIndented = true,
    };

    // -- INPC plumbing ----------------------------------------------------

    public event PropertyChangedEventHandler? PropertyChanged;

    private void SetField<T>(ref T field, T value, [CallerMemberName] string? prop = null)
    {
        if (EqualityComparer<T>.Default.Equals(field, value)) return;
        field = value;
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(prop));
    }
}
