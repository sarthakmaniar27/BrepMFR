Attribute VB_Name = "JSON_Generator_2027"
' JSON_Generator_2027.bas - SLDPRT -> Brep/UV JSON via BaselineOutputCmd
'
' Orchestrator-ready (same status contract as ThreadCreationScript8):
'   - heartbeat.txt   : timestamp|status|filepath
'   - skip_files.txt  : one basename per line (# comments ok)
'   - batch_done.marker when the full walk finishes
'   - ExitApp on done so the outer babysitter can stop
'
' IMPORTANT: After editing, save as JSON_Generator_2027.swp from SOLIDWORKS
' (Tools -> Macro -> Edit / Save). /m only runs .swp, not .bas.
'
' Paths below match the conversion layout on the SW machines.

Option Explicit

' --- Paths ------------------------------------------------------------------
Private Const SLDPRT_FOLDER      As String = "C:\Threads\conversion\sldprts"
Private Const OUTPUT_JSON_FOLDER As String = "C:\Threads\conversion\jsons"
Private Const UV_JSON_FOLDER     As String = "C:\Threads\conversion\uv_jsons"

' Separate from thread-pipeline status (C:\Threads\status) so jobs never collide.
Private Const STATUS_FOLDER      As String = "C:\Threads\conversion\status"
Private Const HEARTBEAT_FILE     As String = "C:\Threads\conversion\status\heartbeat.txt"
Private Const DONE_MARKER        As String = "C:\Threads\conversion\status\batch_done.marker"
Private Const SKIP_LIST_FILE     As String = "C:\Threads\conversion\status\skip_files.txt"
Private Const IN_PROGRESS_FILE   As String = "C:\Threads\conversion\status\in_progress.txt"

' 0 = no size filter. Set e.g. 1500000000 to auto-skip parts larger than ~1.5 GB.
Private Const MAX_SLDPRT_BYTES   As Long = 0

Private Const EXIT_ON_DONE       As Boolean = True

Dim swApp As SldWorks.SldWorks
Dim swModel As SldWorks.ModelDoc2
Dim swAppInternal As SldWorksInternal2
Dim fso As Object
Dim gSkipSet As Object   ' Scripting.Dictionary

Sub main()
    Set swApp = Application.SldWorks
    Set swAppInternal = swApp
    Set fso = CreateObject("Scripting.FileSystemObject")

    EnsureFolder STATUS_FOLDER
    EnsureFolder OUTPUT_JSON_FOLDER
    EnsureFolder UV_JSON_FOLDER

    ' If last run died mid-open/export, blacklist that part and continue.
    PromoteInProgressToSkip

    LoadSkipList

    Dim folderPath As String
    folderPath = SLDPRT_FOLDER
    If Right$(folderPath, 1) <> "\" Then folderPath = folderPath & "\"

    If Not fso.FolderExists(folderPath) Then
        WriteHeartbeat "", "fatal_no_folder"
        Exit Sub
    End If

    ' Clear stale done marker so the babysitter does not exit immediately.
    On Error Resume Next
    If fso.FileExists(DONE_MARKER) Then fso.DeleteFile DONE_MARKER, True
    On Error GoTo 0

    WriteHeartbeat "", "batch_start"
    ProcessFolder fso.GetFolder(folderPath)

    ClearInProgress
    WriteHeartbeat "", "batch_done"
    WriteDoneMarker

    If EXIT_ON_DONE Then
        On Error Resume Next
        swApp.ExitApp
        On Error GoTo 0
    End If
End Sub

Sub ProcessFolder(fFolder As Object)
    Dim fFile As Object
    Dim fSubFolder As Object
    Dim fileError As Long
    Dim fileWarning As Long
    Dim baseName As String
    Dim matchedFile As String
    Dim outputFolderPath As String

    outputFolderPath = OUTPUT_JSON_FOLDER
    If Right$(outputFolderPath, 1) <> "\" Then outputFolderPath = outputFolderPath & "\"

    For Each fFile In fFolder.Files
        If LCase$(Right$(fFile.Name, 7)) = ".sldprt" Then
            baseName = Left$(fFile.Name, Len(fFile.Name) - 7)

            If IsInSkipList(fFile.Name) Then
                WriteHeartbeat fFile.Path, "skipped_blacklist"
                GoTo NextFile
            End If

            matchedFile = Dir(outputFolderPath & baseName & "*.json")
            If matchedFile <> "" Then
                WriteHeartbeat fFile.Path, "skipped_done"
                GoTo NextFile
            End If

            If MAX_SLDPRT_BYTES > 0 Then
                If fFile.Size > MAX_SLDPRT_BYTES Then
                    AppendToSkipList fFile.Name, "auto-skip oversize"
                    WriteHeartbeat fFile.Path, "skipped_size"
                    GoTo NextFile
                End If
            End If

            ' Active statuses (file_start / opening / exporting) let the
            ' babysitter attribute a crash/hang to this file.
            WriteHeartbeat fFile.Path, "file_start"
            WriteInProgress fFile.Name

            fileError = 0
            fileWarning = 0
            WriteHeartbeat fFile.Path, "opening"
            Set swModel = swApp.OpenDoc6(fFile.Path, swDocPART, swOpenDocOptions_Silent, "", fileError, fileWarning)

            If swModel Is Nothing Then
                AppendToSkipList fFile.Name, "open failed err=" & CStr(fileError)
                WriteHeartbeat fFile.Path, "open_failed"
                ClearInProgress
                GoTo NextFile
            End If

            WriteHeartbeat fFile.Path, "exporting"
            On Error Resume Next
            swAppInternal.BaselineOutputCmd 100040, OUTPUT_JSON_FOLDER & "|1|" & UV_JSON_FOLDER
            On Error GoTo 0

            On Error Resume Next
            swApp.CloseDoc fFile.Name
            On Error GoTo 0
            Set swModel = Nothing

            ClearInProgress

            matchedFile = Dir(outputFolderPath & baseName & "*.json")
            If matchedFile <> "" Then
                WriteHeartbeat fFile.Path, "file_done"
            Else
                ' Export returned but no JSON — do not blacklist forever on first miss;
                ' heartbeat still shows we finished attempting this file.
                WriteHeartbeat fFile.Path, "file_no_json"
            End If
        End If
NextFile:
    Next

    For Each fSubFolder In fFolder.SubFolders
        ProcessFolder fSubFolder
    Next
End Sub

' --- Skip list / heartbeat / markers ---------------------------------------

Private Sub LoadSkipList()
    Set gSkipSet = CreateObject("Scripting.Dictionary")
    gSkipSet.CompareMode = 1

    On Error Resume Next
    If Dir(SKIP_LIST_FILE) = "" Then Exit Sub

    Dim fnum As Integer
    Dim sLine As String
    fnum = FreeFile
    Open SKIP_LIST_FILE For Input As #fnum
    Do While Not EOF(fnum)
        Line Input #fnum, sLine
        sLine = Trim$(sLine)
        If Len(sLine) > 0 And Left$(sLine, 1) <> "#" Then
            If Not gSkipSet.Exists(LCase$(sLine)) Then gSkipSet.Add LCase$(sLine), True
        End If
    Loop
    Close #fnum
    On Error GoTo 0
End Sub

Private Function IsInSkipList(ByVal sName As String) As Boolean
    IsInSkipList = False
    If gSkipSet Is Nothing Then Exit Function
    If gSkipSet.Exists(LCase$(sName)) Then IsInSkipList = True
End Function

Private Sub AppendToSkipList(ByVal sName As String, ByVal reason As String)
    On Error Resume Next
    EnsureFolder STATUS_FOLDER
    Dim fnum As Integer
    fnum = FreeFile
    Open SKIP_LIST_FILE For Append As #fnum
    Print #fnum, "# " & Format$(Now, "yyyy-mm-ddThh:nn:ss") & "  " & reason
    Print #fnum, sName
    Close #fnum
    If Not gSkipSet Is Nothing Then
        If Not gSkipSet.Exists(LCase$(sName)) Then gSkipSet.Add LCase$(sName), True
    End If
    On Error GoTo 0
End Sub

Private Sub PromoteInProgressToSkip()
    On Error Resume Next
    If Dir(IN_PROGRESS_FILE) = "" Then Exit Sub
    Dim fnum As Integer
    Dim sName As String
    fnum = FreeFile
    Open IN_PROGRESS_FILE For Input As #fnum
    If Not EOF(fnum) Then Line Input #fnum, sName
    Close #fnum
    sName = Trim$(sName)
    If Len(sName) > 0 Then
        AppendToSkipList sName, "leftover in_progress from previous crash/hang"
    End If
    Kill IN_PROGRESS_FILE
    On Error GoTo 0
End Sub

Private Sub WriteInProgress(ByVal sName As String)
    On Error Resume Next
    Dim fnum As Integer
    fnum = FreeFile
    Open IN_PROGRESS_FILE For Output As #fnum
    Print #fnum, sName
    Close #fnum
    On Error GoTo 0
End Sub

Private Sub ClearInProgress()
    On Error Resume Next
    If Dir(IN_PROGRESS_FILE) <> "" Then Kill IN_PROGRESS_FILE
    On Error GoTo 0
End Sub

Private Sub WriteHeartbeat(ByVal sPath As String, ByVal sStatus As String)
    On Error Resume Next
    EnsureFolder STATUS_FOLDER
    Dim fnum As Integer
    fnum = FreeFile
    Open HEARTBEAT_FILE For Output As #fnum
    Print #fnum, Format$(Now, "yyyy-mm-dd hh:nn:ss") & "|" & sStatus & "|" & sPath
    Close #fnum
    On Error GoTo 0
End Sub

Private Sub WriteDoneMarker()
    On Error Resume Next
    EnsureFolder STATUS_FOLDER
    Dim fnum As Integer
    fnum = FreeFile
    Open DONE_MARKER For Output As #fnum
    Print #fnum, Format$(Now, "yyyy-mm-dd hh:nn:ss") & "|batch finished cleanly"
    Close #fnum
    On Error GoTo 0
End Sub

Private Sub EnsureFolder(ByVal sPath As String)
    On Error Resume Next
    If Not fso Is Nothing Then
        If Not fso.FolderExists(sPath) Then fso.CreateFolder sPath
    Else
        If Dir(sPath, vbDirectory) = "" Then MkDir sPath
    End If
    On Error GoTo 0
End Sub
