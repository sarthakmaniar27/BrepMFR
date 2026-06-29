namespace SwOrchestrator;

public enum LogLevel
{
    Debug,
    Info,
    Warn,
    Error,
}

/// <summary>
/// One row of the activity log. Pure data — no WPF references, so this can
/// be shared between the GUI and the CLI projects. The GUI maps Level to a
/// brush via a XAML IValueConverter (see LogLevelToBrushConverter); the CLI
/// just prints the level as text.
/// </summary>
public sealed class LogEntry
{
    public DateTime Timestamp { get; }
    public LogLevel Level { get; }
    public string Message { get; }

    public LogEntry(LogLevel level, string message)
    {
        Timestamp = DateTime.Now;
        Level = level;
        Message = message;
    }

    public string TimestampText => Timestamp.ToString("HH:mm:ss");

    public string LevelText => Level switch
    {
        LogLevel.Debug => "DEBUG",
        LogLevel.Info  => "INFO",
        LogLevel.Warn  => "WARN",
        LogLevel.Error => "ERROR",
        _ => Level.ToString().ToUpper(),
    };
}
