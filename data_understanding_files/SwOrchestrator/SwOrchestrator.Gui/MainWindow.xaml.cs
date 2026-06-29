using System.Collections.Specialized;
using System.Diagnostics;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using Microsoft.Win32;

namespace SwOrchestrator;

public partial class MainWindow : Window
{
    private readonly MainViewModel _vm;

    public MainWindow()
    {
        InitializeComponent();
        _vm = new MainViewModel();
        DataContext = _vm;

        // Auto-scroll the log to the newest entry.
        _vm.LogEntries.CollectionChanged += (_, e) =>
        {
            if (e.Action == NotifyCollectionChangedAction.Add && LogList.Items.Count > 0)
            {
                LogList.ScrollIntoView(LogList.Items[LogList.Items.Count - 1]);
            }
        };

        Closing += MainWindow_Closing;
    }

    // -- Close handling --------------------------------------------------

    private void MainWindow_Closing(object? sender, System.ComponentModel.CancelEventArgs e)
    {
        if (_vm.IsRunning)
        {
            var ans = MessageBox.Show(
                "Orchestrator is still running. Stop it and exit?",
                "Confirm exit", MessageBoxButton.YesNo, MessageBoxImage.Question);
            if (ans != MessageBoxResult.Yes) { e.Cancel = true; return; }
            _vm.Stop();
        }

        try { _vm.Settings.Save(); } catch { /* best effort */ }
    }

    // -- Browse buttons --------------------------------------------------

    private void BrowseSolidWorksExe_Click(object sender, RoutedEventArgs e)
    {
        var dlg = new OpenFileDialog
        {
            Title  = "Locate SLDWORKS.exe",
            Filter = "SOLIDWORKS executable (SLDWORKS.exe)|SLDWORKS.exe|All files (*.*)|*.*",
            FileName = _vm.Settings.SolidWorksExe,
        };
        if (TrySetInitialDir(dlg, _vm.Settings.SolidWorksExe))
        {
            // dlg.InitialDirectory set
        }
        if (dlg.ShowDialog(this) == true)
            _vm.Settings.SolidWorksExe = dlg.FileName;
    }

    private void BrowseMacroPath_Click(object sender, RoutedEventArgs e)
    {
        var dlg = new OpenFileDialog
        {
            Title  = "Pick the .swp macro file",
            Filter = "SOLIDWORKS macro (*.swp)|*.swp|All files (*.*)|*.*",
            FileName = _vm.Settings.MacroPath,
        };
        TrySetInitialDir(dlg, _vm.Settings.MacroPath);
        if (dlg.ShowDialog(this) == true)
            _vm.Settings.MacroPath = dlg.FileName;
    }

    private void BrowseStepsFolder_Click(object sender, RoutedEventArgs e)
    {
        var dlg = new OpenFolderDialog
        {
            Title = "Pick the STEPS folder",
            InitialDirectory = SafeDirectory(_vm.Settings.StepsFolder),
        };
        if (dlg.ShowDialog(this) == true)
        {
            _vm.Settings.StepsFolder = dlg.FolderName;
            _vm.UpdateInitialProgress();
            _vm.RefreshSkipList();
        }
    }

    private void BrowseStatusFolder_Click(object sender, RoutedEventArgs e)
    {
        var dlg = new OpenFolderDialog
        {
            Title = "Pick the status folder (where heartbeat / done marker / skip list live)",
            InitialDirectory = SafeDirectory(_vm.Settings.StatusFolder),
        };
        if (dlg.ShowDialog(this) == true)
        {
            _vm.Settings.StatusFolder = dlg.FolderName;
            _vm.RefreshSkipList();
        }
    }

    private void StepsFolder_LostFocus(object sender, RoutedEventArgs e)
    {
        // When the user types a new STEPS folder, re-count progress so the
        // bar and counter reflect the new location without needing to start.
        _vm.UpdateInitialProgress();
    }

    // -- Action buttons --------------------------------------------------

    private void SaveSettings_Click(object sender, RoutedEventArgs e)
    {
        try
        {
            _vm.Settings.Save();
            MessageBox.Show(this, $"Settings saved to:\n{AppSettings.SettingsPath}",
                "Saved", MessageBoxButton.OK, MessageBoxImage.Information);
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, $"Could not save settings:\n{ex.Message}",
                "Save failed", MessageBoxButton.OK, MessageBoxImage.Error);
        }
    }

    private void Start_Click(object sender, RoutedEventArgs e)
    {
        _vm.Start();
    }

    private void Stop_Click(object sender, RoutedEventArgs e)
    {
        var ans = MessageBox.Show(this,
            "Stop the orchestrator and kill SOLIDWORKS?",
            "Confirm stop", MessageBoxButton.YesNo, MessageBoxImage.Question);
        if (ans == MessageBoxResult.Yes) _vm.Stop();
    }

    private void ResetState_Click(object sender, RoutedEventArgs e)
    {
        var ans = MessageBox.Show(this,
            "Clear the heartbeat and done marker files?\n" +
            "(SLDPRT outputs and the skip list are NOT touched.)",
            "Confirm reset", MessageBoxButton.YesNo, MessageBoxImage.Question);
        if (ans == MessageBoxResult.Yes) _vm.ResetState();
    }

    private void OpenStatusFolder_Click(object sender, RoutedEventArgs e)
    {
        var folder = _vm.Settings.StatusFolder;
        try
        {
            if (Directory.Exists(folder))
            {
                Process.Start(new ProcessStartInfo
                {
                    FileName = folder,
                    UseShellExecute = true,
                });
            }
            else
            {
                MessageBox.Show(this, $"Folder does not exist:\n{folder}",
                    "Not found", MessageBoxButton.OK, MessageBoxImage.Warning);
            }
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, $"Could not open folder:\n{ex.Message}",
                "Error", MessageBoxButton.OK, MessageBoxImage.Error);
        }
    }

    // -- Helpers ---------------------------------------------------------

    private static bool TrySetInitialDir(OpenFileDialog dlg, string path)
    {
        var dir = SafeDirectory(path);
        if (dir != null) { dlg.InitialDirectory = dir; return true; }
        return false;
    }

    private static string? SafeDirectory(string path)
    {
        try
        {
            if (string.IsNullOrEmpty(path)) return null;
            if (Directory.Exists(path)) return path;
            var dir = Path.GetDirectoryName(path);
            if (!string.IsNullOrEmpty(dir) && Directory.Exists(dir)) return dir;
        }
        catch { }
        return null;
    }
}
