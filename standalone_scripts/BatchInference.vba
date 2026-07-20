'----------------------------------------------------------------------------
' Pure STEP Inference Batch (No feature creation, just raw STEP)
'
' Loops through \\GR-SW65551\abc_steps
' For each .step/.stp file:
'   1. Opens the file in SolidWorks (extracting Parasolid geometry)
'   2. Calls command 100050 to run inference natively
'   3. Closes the file without saving
'----------------------------------------------------------------------------
Option Explicit

Dim swApp As SldWorks.SldWorks
Dim swAppInternal As SldWorksInternal2

Const SW_BREP_MFR_INFERENCE_CMD As Long = 100050

Sub main()
    Set swApp = Application.SldWorks
    Set swAppInternal = swApp
    
    Dim folderPath As String
    folderPath = "\\GR-SW65551\abc_steps\"
    
    Dim fileName As String
    fileName = Dir(folderPath & "*.step")
    
    swApp.SetUserPreferenceIntegerValue swStepAP, 214
    
    Dim lErrors As Long, lWarnings As Long
    Dim swModel As SldWorks.ModelDoc2
    
    Do While Len(fileName) > 0
        Dim fullPath As String
        fullPath = folderPath & fileName
        
        ' 1. Open the raw STEP file
        Set swModel = swApp.LoadFile4(fullPath, "r", Nothing, lErrors)
        
        If Not swModel Is Nothing Then
            If swModel.GetType = swDocPART Then
                
                ' 2. Run Inference Command
                Dim inArgs As Variant
                inArgs = swModel
                On Error Resume Next
                ' This triggers your C++ DLL to extract features and run ONNX natively
                swAppInternal.AITrainUtils SW_BREP_MFR_INFERENCE_CMD, inArgs
                On Error GoTo 0
                
            End If
            
            ' 3. Close without saving
            swApp.CloseDoc swModel.GetTitle
            Set swModel = Nothing
        End If
        
        ' Get next file
        fileName = Dir
    Loop
    
    MsgBox "Batch inference complete!", vbInformation
End Sub
