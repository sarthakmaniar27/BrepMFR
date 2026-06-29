namespace SwOrchestrator;

/// <summary>
/// Base for events the Orchestrator pushes to the UI via IProgress.
/// </summary>
public abstract record OrchestratorEvent;

/// <summary>A line for the activity log.</summary>
public sealed record LogEvent(LogLevel Level, string Message) : OrchestratorEvent;

/// <summary>Files-processed counter update.</summary>
public sealed record ProgressEvent(int Done, int Total) : OrchestratorEvent;

/// <summary>The macro just reported it's working on a particular file/variation.</summary>
public sealed record HeartbeatEvent(string Status, string StepFile, DateTime Timestamp) : OrchestratorEvent;

/// <summary>Overall orchestrator state transition.</summary>
public sealed record RunStateEvent(RunState State) : OrchestratorEvent;

/// <summary>A file was added to the skip list.</summary>
public sealed record SkipListEvent(IReadOnlyList<string> CurrentSkipList) : OrchestratorEvent;

public enum RunState
{
    Idle,
    Running,
    Stopping,
    Complete,
    Failed,
}
