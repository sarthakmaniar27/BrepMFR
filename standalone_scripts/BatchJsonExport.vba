'----------------------------------------------------------------------------
' Pure STEP to JSON Batch (No feature creation, just raw STEP)
'
' IMPORTANT — open timeout:
'   LoadFile4 is BLOCKING. VBA cannot cancel it after 5 seconds from inside
'   the same macro. Instead we use an external watchdog:
'     1. Before open, write the file name to in_progress.txt
'     2. Start Watchdog-StepOpen.ps1 (kills SolidWorks if still in progress
'        after OPEN_TIMEOUT_SEC)
'     3. On next macro start, any leftover in_progress name is added to
'        skip_list.txt so that hanging/crashing part is not retried.
'
'   5 seconds is usually too aggressive for STEP import; default is 60.
'   Set OPEN_TIMEOUT_SEC = 5 only if you truly want that.
'----------------------------------------------------------------------------
Option Explicit

Dim swApp As SldWorks.SldWorks
Dim swAppInternal As SldWorksInternal2

' Seconds the watchdog waits before killing SolidWorks on a stuck open.
' Set to 5 if you want a hard 5s limit (will skip many valid large parts).
Private Const OPEN_TIMEOUT_SEC As Long = 60

' Skip STEP files larger than this (bytes). 0 = no size filter.
' 80 MB is a reasonable guard against pathological files.
Private Const MAX_STEP_BYTES As Long = 80000000

Private Const WATCHDOG_PS1 As String = "C:\jsons\Watchdog-StepOpen.ps1"

Sub main()
    Set swApp = Application.SldWorks
    Set swAppInternal = swApp
    
    Dim folderPath As String
    ' IMPORTANT: trailing backslash required, otherwise Dir becomes
    '   C:\abc_steps_not_in_allowlist*.step  (matches nothing)
    folderPath = "C:\abc_steps_not_in_allowlist\"
    
    Dim outFolder As String
    outFolder = "C:\jsons"
    
    Dim logFolder As String
    logFolder = outFolder & "\batch_logs"
    EnsureFolder logFolder
    
    Dim skipListPath As String
    Dim inProgressPath As String
    Dim logPath As String
    skipListPath = logFolder & "\skip_list.txt"
    inProgressPath = logFolder & "\in_progress.txt"
    logPath = logFolder & "\batch_log.txt"
    
    ' If last run died mid-open, never retry that part automatically.
    PromoteInProgressToSkip skipListPath, inProgressPath, logPath
    
    Dim colFiles As Collection
    Set colFiles = New Collection
    
    Dim fileName As String
    fileName = Dir(folderPath & "*.step")
    Do While Len(fileName) > 0
        colFiles.Add fileName
        fileName = Dir
    Loop
    
    If colFiles.Count = 0 Then
        MsgBox "No .step files found in:" & vbCrLf & folderPath & vbCrLf & vbCrLf & _
               "Check the path ends with a backslash \", vbExclamation
        Exit Sub
    End If
    
    MsgBox "Found " & colFiles.Count & " STEP file(s). Starting batch...", vbInformation
    
    swApp.SetUserPreferenceIntegerValue swStepAP, 214
    
    Dim lErrors As Long, lWarnings As Long
    Dim swModel As SldWorks.ModelDoc2
    
    Dim i As Long
    For i = 1 To colFiles.Count
        fileName = colFiles(i)
        Dim fullPath As String
        fullPath = folderPath & fileName
        
        Dim stemName As String
        Dim dotPos As Long
        dotPos = InStrRev(fileName, ".")
        If dotPos > 0 Then
            stemName = Left(fileName, dotPos - 1)
        Else
            stemName = fileName
        End If
        
        If AnyBodyJsonExists(outFolder, stemName) Then
            GoTo NextFile
        End If
        
        If IsInSkipList(skipListPath, fileName) Then
            AppendLog logPath, "SKIP_LIST: " & fileName
            GoTo NextFile
        End If
        
        If MAX_STEP_BYTES > 0 Then
            Dim fSize As Long
            fSize = FileLenSafe(fullPath)
            If fSize > MAX_STEP_BYTES Then
                AppendSkip skipListPath, fileName
                AppendLog logPath, "SKIP_SIZE (" & fSize & " bytes): " & fileName
                GoTo NextFile
            End If
        End If
        
        ' Mark in-progress BEFORE open so a hang/crash can be detected.
        WriteTextFile inProgressPath, fileName
        StartOpenWatchdog OPEN_TIMEOUT_SEC, inProgressPath, skipListPath
        
        On Error Resume Next
        Set swModel = swApp.LoadFile4(fullPath, "r", Nothing, lErrors)
        Dim openErr As Long
        openErr = Err.Number
        On Error GoTo 0
        
        ' Open finished (success or fail) — clear in-progress so watchdog exits.
        ClearInProgress inProgressPath
        
        If openErr <> 0 Or swModel Is Nothing Then
            AppendSkip skipListPath, fileName
            AppendLog logPath, "SKIP_OPEN_FAIL err=" & openErr & " swErr=" & lErrors & ": " & fileName
            Set swModel = Nothing
            GoTo NextFile
        End If
        
        If Not swModel Is Nothing Then
            If swModel.GetType = swDocPART Then
                
                Dim tempSldPrt As String
                tempSldPrt = outFolder & "\" & stemName & ".SLDPRT"
                
                On Error Resume Next
                swModel.Extension.SaveAs tempSldPrt, 0, 1, Nothing, lErrors, lWarnings
                On Error GoTo 0
                
                On Error Resume Next
                swAppInternal.BaselineOutputCmd 100040, outFolder & "|5"
                On Error GoTo 0
                
                KeepOnlyOneBodyJson outFolder, stemName
                
                On Error Resume Next
                If Dir(tempSldPrt) <> "" Then Kill tempSldPrt
                On Error GoTo 0
                
                If AnyBodyJsonExists(outFolder, stemName) Then
                    AppendLog logPath, "OK: " & fileName
                Else
                    AppendSkip skipListPath, fileName
                    AppendLog logPath, "SKIP_NO_JSON: " & fileName
                End If
            End If
            
            On Error Resume Next
            swApp.CloseDoc swModel.GetTitle
            On Error GoTo 0
            Set swModel = Nothing
        End If
        
NextFile:
    Next i
    
    ClearInProgress inProgressPath
    MsgBox "Batch JSON generation complete (1 JSON per STEP)!", vbInformation
End Sub


Private Function AnyBodyJsonExists(ByVal outFolder As String, ByVal stemName As String) As Boolean
    Dim f As String
    f = Dir(outFolder & "\" & stemName & "_*.json")
    AnyBodyJsonExists = (Len(f) > 0)
End Function


Private Sub KeepOnlyOneBodyJson(ByVal outFolder As String, ByVal stemName As String)
    Dim col As Collection
    Set col = New Collection
    
    Dim f As String
    f = Dir(outFolder & "\" & stemName & "_*.json")
    Do While Len(f) > 0
        col.Add f
        f = Dir
    Loop
    
    If col.Count <= 1 Then Exit Sub
    
    Dim keep As String
    keep = col(1)
    
    Dim j As Long
    For j = 2 To col.Count
        If StrComp(CStr(col(j)), keep, vbTextCompare) < 0 Then
            keep = CStr(col(j))
        End If
    Next j
    
    For j = 1 To col.Count
        If StrComp(CStr(col(j)), keep, vbTextCompare) <> 0 Then
            On Error Resume Next
            Kill outFolder & "\" & CStr(col(j))
            On Error GoTo 0
        End If
    Next j
End Sub


Private Sub EnsureFolder(ByVal folderPath As String)
    If Dir(folderPath, vbDirectory) = "" Then
        MkDir folderPath
    End If
End Sub


Private Function FileLenSafe(ByVal filePath As String) As Long
    On Error Resume Next
    FileLenSafe = FileLen(filePath)
    If Err.Number <> 0 Then
        FileLenSafe = 0
        Err.Clear
    End If
    On Error GoTo 0
End Function


Private Sub WriteTextFile(ByVal filePath As String, ByVal content As String)
    Dim n As Integer
    n = FreeFile
    Open filePath For Output As #n
    Print #n, content
    Close #n
End Sub


Private Sub ClearInProgress(ByVal inProgressPath As String)
    On Error Resume Next
    If Dir(inProgressPath) <> "" Then Kill inProgressPath
    On Error GoTo 0
End Sub


Private Sub AppendLog(ByVal logPath As String, ByVal msg As String)
    Dim n As Integer
    n = FreeFile
    Open logPath For Append As #n
    Print #n, Format$(Now, "yyyy-mm-dd hh:nn:ss") & "  " & msg
    Close #n
End Sub


Private Sub AppendSkip(ByVal skipListPath As String, ByVal fileName As String)
    If IsInSkipList(skipListPath, fileName) Then Exit Sub
    Dim n As Integer
    n = FreeFile
    Open skipListPath For Append As #n
    Print #n, fileName
    Close #n
End Sub


Private Function IsInSkipList(ByVal skipListPath As String, ByVal fileName As String) As Boolean
    IsInSkipList = False
    If Dir(skipListPath) = "" Then Exit Function
    
    Dim n As Integer
    Dim line As String
    n = FreeFile
    Open skipListPath For Input As #n
    Do While Not EOF(n)
        Line Input #n, line
        If StrComp(Trim$(line), fileName, vbTextCompare) = 0 Then
            IsInSkipList = True
            Close #n
            Exit Function
        End If
    Loop
    Close #n
End Function


' If SolidWorks was killed mid-open, in_progress.txt still exists — skip that file forever.
Private Sub PromoteInProgressToSkip(ByVal skipListPath As String, ByVal inProgressPath As String, ByVal logPath As String)
    If Dir(inProgressPath) = "" Then Exit Sub
    
    Dim n As Integer
    Dim line As String
    n = FreeFile
    Open inProgressPath For Input As #n
    If Not EOF(n) Then
        Line Input #n, line
    End If
    Close #n
    
    line = Trim$(line)
    If Len(line) > 0 Then
        AppendSkip skipListPath, line
        AppendLog logPath, "SKIP_TIMEOUT_OR_CRASH (from previous run): " & line
    End If
    
    ClearInProgress inProgressPath
End Sub


' Launch external watchdog. It kills SolidWorks if in_progress.txt still exists
' after TimeoutSec (meaning LoadFile4 is still hung).
Private Sub StartOpenWatchdog(ByVal timeoutSec As Long, ByVal inProgressPath As String, ByVal skipListPath As String)
    On Error Resume Next
    
    Dim cmd As String
    If Dir(WATCHDOG_PS1) = "" Then
        ' Watchdog script missing — continue without hard timeout.
        On Error GoTo 0
        Exit Sub
    End If
    
    cmd = "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File """ & _
          WATCHDOG_PS1 & """" & _
          " -TimeoutSec " & CStr(timeoutSec) & _
          " -InProgressPath """ & inProgressPath & """" & _
          " -SkipListPath """ & skipListPath & """"
    
    Shell cmd, vbHide
    On Error GoTo 0
End Sub
