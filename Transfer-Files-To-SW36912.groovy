// ============================================================================
// Job: Transfer-Filtered-SLDPRTs
// ============================================================================
// Runs on GR-SW36912 (destination machine) for fast local writes.
//
// Reads matched_files.json from the network share, scans the NEW sldprt
// source directories, and copies ONLY SLDPRTs whose prefix ID matches.
// All filename variations of a matched prefix are included.
//
// Pre-requisite: matched_files.json must be at:
//   \\DZ4-SMR52-DSA\cadsynth_data\matched_files.json
// ============================================================================

pipeline {
    agent { label 'GR-SW36912' }
    options {
        timestamps()
        disableConcurrentBuilds()
    }

    stages {

        stage('1 — Copy filtered SLDPRTs') {
            steps {
                powershell '''
                    $ErrorActionPreference = "Stop"

                    $jsonPath   = "\\\\DZ4-SMR52-DSA\\cadsynth_data\\matched_files.json"
                    $sourceDirs = @(
                        "\\\\Gr-sw26877\\d\\brepmfr_sldprts\\cadsynth",
                        "\\\\Gr-sw34959\\d\\brepmfr_sldprts\\cadsynth"
                    )
                    $destDir    = "C:\\Threads\\conversion\\sldprts"
                    $batchSize  = 200

                    if (!(Test-Path $destDir)) {
                        New-Item -ItemType Directory -Path $destDir -Force | Out-Null
                    }

                    # --- Load target IDs ---
                    Write-Host "Loading target IDs from $jsonPath ..."
                    $json = Get-Content $jsonPath -Raw | ConvertFrom-Json
                    $targetIds = @{}
                    foreach ($f in $json.matched_files) {
                        $id = [System.IO.Path]::GetFileNameWithoutExtension($f).ToLower()
                        $targetIds[$id] = $true
                    }
                    Write-Host "  Target IDs: $($targetIds.Count)"

                    # --- Scan, filter, and copy per source directory ---
                    foreach ($srcDir in $sourceDirs) {
                        Write-Host ""
                        Write-Host "Source: $srcDir"

                        if (!(Test-Path $srcDir)) {
                            Write-Host "  [!] Cannot access — skipping"
                            continue
                        }

                        $matchedFiles = Get-ChildItem $srcDir -Filter *.SLDPRT -File | Where-Object {
                            $prefix = ($_.BaseName -split '_')[0].ToLower()
                            $targetIds.ContainsKey($prefix)
                        }
                        $fileNames = @($matchedFiles | ForEach-Object { $_.Name })
                        Write-Host "  Matched files: $($fileNames.Count)"

                        if ($fileNames.Count -eq 0) { continue }

                        $totalBatches = [Math]::Ceiling($fileNames.Count / $batchSize)
                        for ($i = 0; $i -lt $fileNames.Count; $i += $batchSize) {
                            $batch = $fileNames[$i..([Math]::Min($i + $batchSize - 1, $fileNames.Count - 1))]
                            $batchNum = [Math]::Floor($i / $batchSize) + 1

                            $args = @($srcDir, $destDir) + $batch + @("/MT:32", "/R:3", "/W:5", "/J", "/NP", "/NDL", "/NJH", "/NJS")
                            $proc = Start-Process -FilePath "robocopy" -ArgumentList $args -Wait -PassThru -NoNewWindow

                            if ($proc.ExitCode -ge 8) {
                                Write-Host "  [!] Batch $batchNum/$totalBatches error code: $($proc.ExitCode)"
                            }
                            if ($batchNum % 10 -eq 0 -or $batchNum -eq $totalBatches) {
                                Write-Host "  [batch $batchNum/$totalBatches] done"
                            }
                        }
                    }
                '''
            }
        }

        stage('2 — Summary') {
            steps {
                powershell '''
                    $count = (Get-ChildItem "C:\\Threads\\conversion\\sldprts" -Filter *.SLDPRT -ErrorAction SilentlyContinue).Count
                    Write-Host ""
                    Write-Host "=================================================="
                    Write-Host "  TRANSFER COMPLETE"
                    Write-Host "=================================================="
                    Write-Host "  SLDPRTs at destination: $count"
                    Write-Host "=================================================="
                '''
            }
        }
    }

    post {
        success { echo 'Filtered SLDPRT transfer complete.' }
        failure { echo 'Transfer failed — check stage output above.' }
    }
}
