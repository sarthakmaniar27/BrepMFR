// ============================================================================
// Job: Distribute-Priority-STEP-Files
// ============================================================================
// Distributes ONLY the remaining priority STEP files (from the filtered list)
// across all VMs into C:\Threads\steps.
//
// Pre-requisite: Run generate_remaining_ids.py to create the filter file at:
//   \\DZ4-SMR52-DSA\cadsynth_data\remaining_step_ids.txt
//
// Each machine reads the filter list from the network share, scans the STEP
// source, picks only matching files, splits them evenly, and copies its chunk.
//
// C:\Threads\steps is cleared on each machine before copying.
// ============================================================================

def NODES = [
    'GR-NAS',
    'GR-SW26859',
    // 'GR-SW26877',
    'GR-SW34959',
    'GR-SW36912',
    'GR-SW41696',
    'GR-SW43226',
    'GR-SW43701',
    'GR-SW53252',
    'GR-SW62363',
    'GR-SW62366',
    'GR-SW65237',
    'GR-SW65700',
    // 'GR-SW66732'
]

def SHARE_IN    = '\\\\DZ4-SMR52-DSA\\cadsynth_data\\orginal_authors\\step'
def FILTER_FILE = '\\\\DZ4-SMR52-DSA\\cadsynth_data\\remaining_step_ids.txt'
def STEPS_DIR   = 'C:\\Threads\\cadsynth\\cad_steps_filtered'

pipeline {
    agent none
    options {
        timestamps()
        disableConcurrentBuilds()
    }

    stages {

        // =====================================================================
        // STAGE 1 — DISTRIBUTE (filtered)
        // Each machine:
        //   1. Reads the filter list from the network share
        //   2. Scans the STEP source and keeps only files whose base ID
        //      (filename without extension) is in the filter list
        //   3. Splits the filtered list evenly using floor-division + remainder
        //   4. Copies its assigned chunk to C:\Threads\steps
        // =====================================================================
        stage('1 — Distribute filtered STEP files') {
            steps {
                script {
                    def nodesCsv = NODES.join("','")
                    def branches = [failFast: false]

                    NODES.each { n ->
                        def nodeName = n

                        branches[nodeName] = {
                            node(nodeName) {
                                powershell """
                                \$ErrorActionPreference = 'Stop'
                                \$sourceDir  = "${SHARE_IN}"
                                \$filterFile = "${FILTER_FILE}"
                                \$stepsDir   = "${STEPS_DIR}"
                                \$machine    = \$env:COMPUTERNAME

                                \$nodes = @('${nodesCsv}')
                                \$M     = \$nodes.Count
                                \$myIdx = [Array]::IndexOf(\$nodes, \$machine)

                                if (\$myIdx -lt 0) {
                                    Write-Error "Machine '\$machine' not found in NODES list"
                                    exit 1
                                }

                                if (!(Test-Path \$stepsDir)) {
                                    Write-Host "[\$machine] Creating \$stepsDir ..."
                                    New-Item -ItemType Directory -Path \$stepsDir -Force | Out-Null
                                }

                                # --- Read the filter list from network share ---
                                Write-Host "[\$machine] Reading filter list from \$filterFile ..."
                                if (!(Test-Path \$filterFile)) {
                                    Write-Error "Filter file not found: \$filterFile — run generate_remaining_ids.py first"
                                    exit 1
                                }
                                \$filterIds = @{}
                                Get-Content \$filterFile | ForEach-Object {
                                    \$id = \$_.Trim().ToLower()
                                    if (\$id -ne '') { \$filterIds[\$id] = \$true }
                                }
                                Write-Host "[\$machine] Filter list contains \$(\$filterIds.Count) IDs"

                                # --- Scan source and filter ---
                                Write-Host "[\$machine] Scanning and filtering source ..."
                                \$filteredFiles = Get-ChildItem \$sourceDir -File |
                                    Where-Object {
                                        \$_.Extension -in @('.step', '.stp') -and
                                        \$filterIds.ContainsKey(\$_.BaseName.ToLower())
                                    } |
                                    Sort-Object Name
                                \$N = \$filteredFiles.Count

                                Write-Host "[\$machine] Filtered STEP files : \$N"
                                Write-Host "[\$machine] Total machines      : \$M"
                                Write-Host "[\$machine] This node index     : \$myIdx"

                                # --- Calculate this machine's slice ---
                                \$base  = [Math]::Floor(\$N / \$M)
                                \$extra = \$N % \$M
                                \$start = \$myIdx * \$base + [Math]::Min(\$myIdx, \$extra)
                                \$count = \$base
                                if (\$myIdx -lt \$extra) { \$count++ }

                                if (\$count -eq 0) {
                                    Write-Host "[\$machine] No files assigned (fewer files than machines)"
                                    exit 0
                                }

                                # --- Clear and copy ---
                                Write-Host "[\$machine] Clearing \$stepsDir ..."
                                Get-ChildItem \$stepsDir -ErrorAction SilentlyContinue |
                                    Remove-Item -Force -Recurse

                                Write-Host "[\$machine] Copying \$count files ..."
                                \$chunk = \$filteredFiles | Select-Object -Skip \$start -First \$count
                                foreach (\$f in \$chunk) {
                                    Copy-Item \$f.FullName \$stepsDir -Force
                                }

                                Write-Host ""
                                Write-Host "[\$machine] Done"
                                Write-Host "[\$machine]   Assigned : \$count files"
                                Write-Host "[\$machine]   Range    : position \$start to \$(\$start + \$count - 1)"
                                Write-Host "[\$machine]   First    : \$(\$chunk[0].Name)"
                                Write-Host "[\$machine]   Last     : \$(\$chunk[-1].Name)"
                                """
                            }
                        }
                    }
                    parallel branches
                }
            }
        }

        // =====================================================================
        // STAGE 2 — SUMMARY
        // =====================================================================
        stage('2 — Summary') {
            steps {
                script {
                    node(NODES[0]) {
                        powershell """
                        \$nodes      = @('${NODES.join("','")}')
                        \$sourceDir  = "${SHARE_IN}"
                        \$filterFile = "${FILTER_FILE}"
                        \$M = \$nodes.Count

                        # Count filtered files
                        \$filterIds = @{}
                        Get-Content \$filterFile | ForEach-Object {
                            \$id = \$_.Trim().ToLower()
                            if (\$id -ne '') { \$filterIds[\$id] = \$true }
                        }
                        \$total = @(Get-ChildItem \$sourceDir -File |
                            Where-Object {
                                \$_.Extension -in @('.step', '.stp') -and
                                \$filterIds.ContainsKey(\$_.BaseName.ToLower())
                            }).Count

                        \$base  = [Math]::Floor(\$total / \$M)
                        \$extra = \$total % \$M

                        \$div  = "=" * 50
                        \$div2 = "-" * 36

                        Write-Host ""
                        Write-Host \$div
                        Write-Host "  PRIORITY DISTRIBUTION COMPLETE"
                        Write-Host \$div
                        Write-Host "  Filter file : \$filterFile"
                        Write-Host "  Filter IDs  : \$(\$filterIds.Count)"
                        Write-Host "  STEP matches: \$total"
                        Write-Host \$div
                        Write-Host ("  {0,-16}  {1,12}" -f "Machine", "Files assigned")
                        Write-Host ("  " + \$div2)
                        for (\$i = 0; \$i -lt \$M; \$i++) {
                            \$c = \$base + (\$i -lt \$extra ? 1 : 0)
                            Write-Host ("  {0,-16}  {1,12}" -f \$nodes[\$i], \$c)
                        }
                        Write-Host ("  " + \$div2)
                        Write-Host ("  {0,-16}  {1,12}" -f "TOTAL", \$total)
                        Write-Host \$div
                        Write-Host ""
                        """
                    }
                }
            }
        }
    }

    post {
        success { echo 'Priority distribution complete — C:\\Threads\\cadsynth\\cad_steps_filtered is ready on all VMs.' }
        failure { echo 'Distribution failed — check individual node console outputs above.' }
    }
}
