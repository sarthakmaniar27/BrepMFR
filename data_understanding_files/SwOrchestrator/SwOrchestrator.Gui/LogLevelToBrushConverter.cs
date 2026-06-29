using System.Globalization;
using System.Windows.Data;
using System.Windows.Media;

namespace SwOrchestrator;

/// <summary>
/// Maps <see cref="LogLevel"/> values to a <see cref="Brush"/> for use in
/// XAML bindings. Used by the activity log to colour each row by severity.
/// Lives in the GUI project because it depends on System.Windows.Media,
/// which isn't available in the Core library.
/// </summary>
public sealed class LogLevelToBrushConverter : IValueConverter
{
    public object Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        if (value is LogLevel level)
        {
            return level switch
            {
                LogLevel.Debug => Brushes.Gray,
                LogLevel.Info  => Brushes.Black,
                LogLevel.Warn  => Brushes.DarkOrange,
                LogLevel.Error => Brushes.Crimson,
                _ => Brushes.Black,
            };
        }
        return Brushes.Black;
    }

    public object ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
        => throw new NotSupportedException();
}
