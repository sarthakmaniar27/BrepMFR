Attribute VB_Name = "ThreadCreationScript8"
' ThreadCreationScript8.bas - resume-on-crash version of v7.
'
' CHANGES vs v7
' -------------
' 1. ProcessStepFile skips files whose six variation outputs already exist.
'    Re-running the macro therefore picks up where the previous run stopped.
' 2. The variation loop skips individual variations whose output already
'    exists. A crash mid-file only forces the missing variations to redo,
'    not the whole file.
' 3. A heartbeat file is rewritten before each file and after each variation,
'    so the Python orchestrator can (a) see progress and (b) know which STEP
'    file SolidWorks was on when it crashed.
' 4. A skip list (skip_files.txt - one bare filename per line) is honored if
'    present. The orchestrator writes to this list when a file crashes
'    SolidWorks repeatedly, so the batch can advance past poison inputs.
' 5. When main() finishes the entire batch, it writes batch_done.marker and
'    calls swApp.ExitApp so the orchestrator knows it's done and can stop.
'
' No threading logic or output paths were changed. The six-variation
' Phase-1/Phase-2 persistent-ID flow inside AddFeaturesFromAIEdges is
' identical to v7.

Option Explicit

' --- Paths ------------------------------------------------------------------
' Adjust these three to match your environment if you are not using the
' default C:\ThreadRecognition\ layout.
Private Const STEP_FOLDER             As String = "C:\ThreadRecognition\STEPS"
Private Const STATUS_FOLDER           As String = "C:\ThreadRecognition"
Private Const BREP_JSON_OUT           As String = "C:\ThreadRecognition\BrepJson|5"

Private Const HEARTBEAT_FILE          As String = "C:\ThreadRecognition\heartbeat.txt"
Private Const DONE_MARKER             As String = "C:\ThreadRecognition\batch_done.marker"
Private Const SKIP_LIST_FILE          As String = "C:\ThreadRecognition\skip_files.txt"

' Set to False if you would rather have SolidWorks stay open after the batch
' finishes (e.g. for inspection). The orchestrator only needs DONE_MARKER to
' know the batch is complete - the ExitApp call is just a courtesy.
Private Const EXIT_ON_DONE            As Boolean = True

' --- Original constants (unchanged) -----------------------------------------
Private Const THREAD_FRACTION         As Double = 2# / 3#
Private Const AI_CMD_THREAD_RIM_EDGES As Long = 100046
Private Const NUM_THREAD_VARIATIONS   As Long = 6
Private Const THREAD_SIZE             As String = "M10x1.5"
Private Const THREAD_METHOD           As Long = 1
Private Const DIAGNOSTIC_USE_FILLET    As Boolean = False
Private Const DIAGNOSTIC_FILLET_RADIUS As Double = 0.003

Dim swApp As SldWorks.SldWorks
Dim swAppInternal As SldWorksInternal2

' Lower-cased filenames (no path) loaded from SKIP_LIST_FILE at startup.
Dim gSkipSet As Object   ' Scripting.Dictionary

Sub main()
    Set swApp = Application.SldWorks
    Set swAppInternal = swApp

    LoadSkipList

    Dim sFolder As String
    sFolder = STEP_FOLDER
    If Right$(sFolder, 1) <> "\" Then sFolder = sFolder & "\"

    If Dir(sFolder, vbDirectory) = "" Then
        WriteHeartbeat "", "fatal_no_folder"
        MsgBox "Folder not found: " & sFolder, vbCritical
        Exit Sub
    End If

    swApp.SetUserPreferenceIntegerValue swStepAP, 214

    WriteHeartbeat "", "batch_start"

    ProcessAllStepFilesInFolder sFolder, "*.step"
    ProcessAllStepFilesInFolder sFolder, "*.stp"

    WriteHeartbeat "", "batch_done"
    WriteDoneMarker

    If EXIT_ON_DONE Then
        On Error Resume Next
        swApp.ExitApp
        On Error GoTo 0
    End If
End Sub

' Enumerate pattern into a Collection, then process.
Private Sub ProcessAllStepFilesInFolder(ByVal sFolder As String, ByVal sPattern As String)
    Dim colFiles As Collection
    Set colFiles = New Collection

    Dim sFile As String
    sFile = Dir(sFolder & sPattern)
    Do While Len(sFile) > 0
        colFiles.Add sFile
        sFile = Dir
    Loop

    Dim i As Long
    For i = 1 To colFiles.Count
        ProcessStepFile sFolder & colFiles(i)
    Next i
End Sub

Private Sub ProcessStepFile(ByVal sStepPath As String)
    ' --- NEW: skip-list (orchestrator-managed) -----------------------------
    If IsInSkipList(sStepPath) Then
        WriteHeartbeat sStepPath, "skipped_blacklist"
        Exit Sub
    End If

    ' --- NEW: skip if every variation already exists. ----------------------
    If IsStepFullyProcessed(sStepPath) Then
        WriteHeartbeat sStepPath, "skipped_done"
        Exit Sub
    End If

    WriteHeartbeat sStepPath, "file_start"

    Dim lErrors As Long, lWarnings As Long
    Dim swModel As SldWorks.ModelDoc2

    Dim v As Long
    For v = 1 To NUM_THREAD_VARIATIONS
        Dim sVarPath As String
        sVarPath = PartPathWithVariationSuffix(sStepPath, v)

        ' --- NEW: skip individual variations whose SLDPRT already exists. --
        If FileExists(sVarPath) Then
            WriteHeartbeat sStepPath, "skipped_var_" & CStr(v)
            GoTo NextVar
        End If

        WriteHeartbeat sStepPath, "var_start_" & CStr(v)

        Set swModel = swApp.LoadFile4(sStepPath, "r", Nothing, lErrors)
        If swModel Is Nothing Then GoTo NextVar
        If swModel.GetType <> swDocPART Then
            swApp.CloseDoc swModel.GetTitle
            Set swModel = Nothing
            GoTo NextVar
        End If

        AddFeaturesFromAIEdges swModel, v

        swModel.Extension.SaveAs sVarPath, swSaveAsCurrentVersion, _
            swSaveAsOptions_Silent, Nothing, lErrors, lWarnings
        swAppInternal.BaselineOutputCmd 100040, BREP_JSON_OUT
        swApp.CloseDoc swModel.GetTitle
        Set swModel = Nothing

        WriteHeartbeat sStepPath, "var_done_" & CStr(v)
NextVar:
    Next v

    WriteHeartbeat sStepPath, "file_done"
End Sub

' --- NEW: helpers -----------------------------------------------------------

Private Function IsStepFullyProcessed(ByVal sStepPath As String) As Boolean
    Dim v As Long
    For v = 1 To NUM_THREAD_VARIATIONS
        If Not FileExists(PartPathWithVariationSuffix(sStepPath, v)) Then
            IsStepFullyProcessed = False
            Exit Function
        End If
    Next v
    IsStepFullyProcessed = True
End Function

Private Function FileExists(ByVal sPath As String) As Boolean
    On Error Resume Next
    FileExists = (Len(Dir(sPath)) > 0)
    On Error GoTo 0
End Function

' Read SKIP_LIST_FILE into gSkipSet as lower-cased basenames. Silent on any
' I/O error - an unreadable / missing list just means no skips.
Private Sub LoadSkipList()
    Set gSkipSet = CreateObject("Scripting.Dictionary")
    gSkipSet.CompareMode = 1   ' TextCompare (case-insensitive)

    On Error Resume Next
    If Not FileExists(SKIP_LIST_FILE) Then Exit Sub

    Dim fnum As Integer
    fnum = FreeFile
    Open SKIP_LIST_FILE For Input As #fnum
    Dim sLine As String
    Do While Not EOF(fnum)
        Line Input #fnum, sLine
        sLine = Trim$(sLine)
        If Len(sLine) > 0 And Left$(sLine, 1) <> "#" Then
            If Not gSkipSet.Exists(LCase$(sLine)) Then
                gSkipSet.Add LCase$(sLine), True
            End If
        End If
    Loop
    Close #fnum
    On Error GoTo 0
End Sub

Private Function IsInSkipList(ByVal sStepPath As String) As Boolean
    IsInSkipList = False
    If gSkipSet Is Nothing Then Exit Function
    Dim sName As String
    sName = LCase$(BaseNameFromPath(sStepPath))
    If gSkipSet.Exists(sName) Then IsInSkipList = True
End Function

Private Function BaseNameFromPath(ByVal sPath As String) As String
    Dim p As Long
    p = InStrRev(sPath, "\")
    If p < 1 Then
        BaseNameFromPath = sPath
    Else
        BaseNameFromPath = Mid$(sPath, p + 1)
    End If
End Function

' Overwrite HEARTBEAT_FILE with one line: timestamp|status|stepfile
' Cheap, atomic enough for the orchestrator's purposes.
Private Sub WriteHeartbeat(ByVal sStepPath As String, ByVal sStatus As String)
    On Error Resume Next
    Dim fnum As Integer
    fnum = FreeFile
    Open HEARTBEAT_FILE For Output As #fnum
    Print #fnum, Format$(Now, "yyyy-mm-dd hh:nn:ss") & "|" & sStatus & "|" & sStepPath
    Close #fnum
    On Error GoTo 0
End Sub

Private Sub WriteDoneMarker()
    On Error Resume Next
    Dim fnum As Integer
    fnum = FreeFile
    Open DONE_MARKER For Output As #fnum
    Print #fnum, Format$(Now, "yyyy-mm-dd hh:nn:ss") & "|batch finished cleanly"
    Close #fnum
    On Error GoTo 0
End Sub

' --- Original helpers (unchanged) -------------------------------------------

Private Function BaseSldprtFromStep(ByVal sStepPath As String) As String
    Dim p As Long
    p = InStrRev(sStepPath, ".")
    If p < 1 Then
        BaseSldprtFromStep = sStepPath & ".SLDPRT"
    Else
        BaseSldprtFromStep = Left$(sStepPath, p - 1) & ".SLDPRT"
    End If
End Function

Private Function PartPathWithVariationSuffix(ByVal sStepPath As String, ByVal varIdx As Long) As String
    Dim p As Long
    p = InStrRev(sStepPath, ".")
    If p < 1 Then
        PartPathWithVariationSuffix = sStepPath & "_" & CStr(varIdx) & ".SLDPRT"
    Else
        PartPathWithVariationSuffix = Left$(sStepPath, p - 1) & "_" & CStr(varIdx) & ".SLDPRT"
    End If
End Function

Private Sub AddFeaturesFromAIEdges(ByVal swModel As SldWorks.ModelDoc2, ByVal varIdx As Long)
    If swModel Is Nothing Then Exit Sub
    If swAppInternal Is Nothing Then Exit Sub

    Dim vOut As Variant
    On Error Resume Next
    Dim inArgs As Variant
    inArgs = swModel
    vOut = swAppInternal.AITrainUtils(AI_CMD_THREAD_RIM_EDGES, inArgs)
    On Error GoTo 0

    If IsEmpty(vOut) Then Exit Sub

    Dim swModelDocExt As SldWorks.ModelDocExtension
    Set swModelDocExt = swModel.Extension
    If swModelDocExt Is Nothing Then Exit Sub

    Dim persistIDs As Collection
    Set persistIDs = New Collection

    If VarType(vOut) = vbObject Then
        If vOut Is Nothing Then Exit Sub
        Dim swE0 As SldWorks.Edge
        Set swE0 = vOut
        Dim pid0 As Variant
        pid0 = Empty
        On Error Resume Next
        pid0 = swModelDocExt.GetPersistReference3(swE0)
        On Error GoTo 0
        If Not IsEmpty(pid0) Then persistIDs.Add pid0

    ElseIf IsArray(vOut) Then
        Dim lb As Long, ub As Long
        On Error Resume Next
        lb = LBound(vOut)
        ub = UBound(vOut)
        If Err.Number <> 0 Then
            Err.Clear
            On Error GoTo 0
            Exit Sub
        End If
        On Error GoTo 0

        Dim i As Long
        Dim swEdge As SldWorks.Edge
        Dim pid As Variant
        For i = lb To ub
            Set swEdge = Nothing
            On Error Resume Next
            Set swEdge = vOut(i)
            On Error GoTo 0
            If Not swEdge Is Nothing Then
                pid = Empty
                On Error Resume Next
                pid = swModelDocExt.GetPersistReference3(swEdge)
                On Error GoTo 0
                If Not IsEmpty(pid) Then persistIDs.Add pid
            End If
        Next i
    Else
        Exit Sub
    End If

    vOut = Empty

    If persistIDs.Count = 0 Then Exit Sub

    Dim k As Long
    Dim errCode As Long
    Dim swFreshEdge As SldWorks.Edge
    For k = 1 To persistIDs.Count
        Set swFreshEdge = Nothing
        errCode = 0
        On Error Resume Next
        Set swFreshEdge = swModelDocExt.GetObjectByPersistReference3(persistIDs(k), errCode)
        On Error GoTo 0

        If Not swFreshEdge Is Nothing And errCode = 0 Then
            If DIAGNOSTIC_USE_FILLET Then
                CreateFilletOnEdge swModel, swFreshEdge
            Else
                CreateSweepThreadOnEdge swModel, swFreshEdge, varIdx
            End If
        End If
    Next k
End Sub

Private Function CreateFilletOnEdge(ByVal swModel As SldWorks.ModelDoc2, _
        ByVal swEdge As SldWorks.Edge) As Boolean
    CreateFilletOnEdge = False
    If swModel Is Nothing Or swEdge Is Nothing Then Exit Function

    swModel.ClearSelection2 True

    Dim swSelData As SldWorks.SelectData
    Set swSelData = Nothing
    On Error Resume Next
    Set swSelData = swModel.SelectionManager.CreateSelectData2()
    If swSelData Is Nothing Then
        Err.Clear
        Set swSelData = swModel.SelectionManager.CreateSelectData()
    End If
    On Error GoTo 0
    If Not swSelData Is Nothing Then swSelData.Mark = 1

    If Not swEdge.Select4(False, swSelData) Then Exit Function

    Dim swFillet As SldWorks.Feature
    On Error Resume Next
    Set swFillet = swModel.FeatureManager.FeatureFillet3( _
        0, _
        DIAGNOSTIC_FILLET_RADIUS, _
        0#, _
        0, _
        1, _
        Empty, Empty, Empty)
    On Error GoTo 0

    swModel.ClearSelection2 True
    CreateFilletOnEdge = Not (swFillet Is Nothing)
End Function

Private Sub GetThreadVariation(ByVal varIdx As Long, ByRef threadType As String, _
        ByRef threadFraction As Double, ByRef rightHanded As Boolean, ByRef threadSize As String, ByRef threadMethod As Long)
    Select Case varIdx
        Case 1
            threadType = "inch die":     rightHanded = True:  threadSize = "#8-36":       threadMethod = 1: threadFraction = 0.5
        Case 2
            threadType = "inch die":     rightHanded = False: threadSize = "#1.7500-5":   threadMethod = 1: threadFraction = 0.36
        Case 3
            threadType = "metric tap":   rightHanded = True:  threadSize = "#M12x1.75":   threadMethod = 0: threadFraction = 0.67
        Case 4
            threadType = "metric tap":   rightHanded = False: threadSize = "#M36x2.0":    threadMethod = 0: threadFraction = 0.45
        Case 5
            threadType = "sp4xx bottle": rightHanded = True:  threadSize = "SP410-L-6":   threadMethod = 0: threadFraction = 0.23
        Case 6
            threadType = "sp4xx bottle": rightHanded = False: threadSize = "SP425-L-12":  threadMethod = 0: threadFraction = 0.75
        Case Else
            threadType = "metric tap":   rightHanded = True:  threadSize = "#M36x2.0":    threadMethod = 0: threadFraction = 0.5
    End Select
End Sub

Private Function GetCylinderDataFromThreadEdge(ByVal swEdge As SldWorks.Edge, _
        ByRef r As Double, ByRef dAxialLen As Double) As Boolean
    GetCylinderDataFromThreadEdge = False
    r = 0#
    dAxialLen = 0#
    If swEdge Is Nothing Then Exit Function

    Dim vF As Variant
    vF = swEdge.GetTwoAdjacentFaces2
    If IsEmpty(vF) Then Exit Function

    Dim swF As SldWorks.Face2
    Dim swSurf As SldWorks.Surface
    Dim i As Long
    For i = LBound(vF) To UBound(vF)
        Set swF = Nothing
        Set swSurf = Nothing

        Set swF = vF(i)
        If swF Is Nothing Then GoTo NextI
        Set swSurf = swF.GetSurface
        If swSurf Is Nothing Then GoTo NextI
        If swSurf.IsCylinder Then
            Dim vCyl As Variant
            vCyl = swSurf.CylinderParams
            If IsEmpty(vCyl) Then GoTo NextI
            r = vCyl(6)
            Dim vUV As Variant
            vUV = swF.GetUVBounds
            If IsEmpty(vUV) Then GoTo NextI
            dAxialLen = Abs(vUV(3) - vUV(2))
            GetCylinderDataFromThreadEdge = (r > 0# And dAxialLen > 0#)
            Exit Function
        End If
NextI:
    Next i
End Function

Private Function CreateSweepThreadOnEdge(ByVal swModel As SldWorks.ModelDoc2, _
        ByVal swEdge As SldWorks.Edge, ByVal varIdx As Long) As Boolean
    CreateSweepThreadOnEdge = False
    If swModel Is Nothing Or swEdge Is Nothing Then Exit Function

    Dim r As Double, dAxialLen As Double
    If Not GetCylinderDataFromThreadEdge(swEdge, r, dAxialLen) Then Exit Function

    Dim threadType As String
    Dim threadSize As String
    Dim bRH As Boolean
    Dim threadMethod As Long
    Dim threadFraction As Double
    GetThreadVariation varIdx, threadType, threadFraction, bRH, threadSize, threadMethod

    Dim dThreadLen As Double
    dThreadLen = dAxialLen * threadFraction

    Dim swThreadFeatData As SldWorks.ThreadFeatureData
    Set swThreadFeatData = swModel.FeatureManager.CreateDefinition(swFeatureNameID_e.swFmSweepThread)
    If swThreadFeatData Is Nothing Then Exit Function

    swThreadFeatData.InitializeThreadData

    swThreadFeatData.Type = threadType
    swThreadFeatData.Size = threadSize
    swThreadFeatData.PitchOverride = False
    swThreadFeatData.rightHanded = bRH

    swModel.ClearSelection2 True
    swThreadFeatData.Edge = swEdge

    swThreadFeatData.EndCondition = swThreadEndCondition_e.swThreadEndCondition_Blind
    swThreadFeatData.BlindDepth = dThreadLen
    swThreadFeatData.threadMethod = threadMethod
    swThreadFeatData.Diameter = 2# * r

    Dim swThreadFeat As SldWorks.Feature
    Set swThreadFeat = swModel.FeatureManager.CreateFeature(swThreadFeatData)
    swModel.ClearSelection2 True

    CreateSweepThreadOnEdge = Not (swThreadFeat Is Nothing)
End Function
