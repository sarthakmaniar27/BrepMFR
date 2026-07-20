#!/usr/bin/env python3
"""Generate Jenkinsfile with embedded allowlist (agents cannot reach LP76 share)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
key_lines = [
    ln.strip()
    for ln in (ROOT / "allowed_step_keys.txt").read_text(encoding="utf-8").splitlines()
    if ln.strip() and not ln.strip().startswith("#")
]
if any("'''" in ln for ln in key_lines):
    raise SystemExit("allowlist contains ''' which breaks Groovy string")

key_count = len(key_lines)
# Split into 3 nearly equal chunks for ALLOWLIST_P1 / P2 / P3
n1 = (key_count + 2) // 3
n2 = (key_count + 1) // 3
chunks = [
    key_lines[:n1],
    key_lines[n1 : n1 + n2],
    key_lines[n1 + n2 :],
]

# Also write standalone text files
for i, chunk in enumerate(chunks, start=1):
    path = ROOT / f"allowed_step_keys_p{i}.txt"
    path.write_text("\n".join(chunk) + ("\n" if chunk else ""), encoding="utf-8")
    print(f"Wrote {path.name}: {len(chunk)} keys")

keys = "\n".join(key_lines) + "\n"  # kept for any legacy use

ps_script = r"""
$ErrorActionPreference = 'Stop'

$allowlistPath = Join-Path -Path $env:WORKSPACE -ChildPath 'allowed_step_keys.txt'
$sourceDir = $env:SOURCE_DIR
$destDir = $env:DEST_DIR

Write-Host '============================================================'
Write-Host ("Node        : {0} / {1}" -f $env:TARGET_NODE, $env:COMPUTERNAME)
Write-Host ("Allowlist   : {0}" -f $allowlistPath)
Write-Host ("Source      : {0}" -f $sourceDir)
Write-Host ("Destination : {0}" -f $destDir)
Write-Host '============================================================'

if (-not (Test-Path -LiteralPath $allowlistPath)) {
    throw ("Allowlist not found: {0}" -f $allowlistPath)
}
if (-not (Test-Path -LiteralPath $sourceDir)) {
    throw ("Source directory not found: {0}" -f $sourceDir)
}

$allow = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
Get-Content -LiteralPath $allowlistPath |
    ForEach-Object { $_.Trim() } |
    Where-Object { $_ -and -not $_.StartsWith('#') } |
    ForEach-Object {
        $stem = [System.IO.Path]::GetFileNameWithoutExtension($_)
        if ($stem -match '^(?<key>.+?_step_\d+)') {
            [void]$allow.Add($Matches['key'].ToLowerInvariant())
        }
        else {
            [void]$allow.Add($stem.ToLowerInvariant())
        }
    }

Write-Host ("Allowlist keys loaded: {0}" -f $allow.Count)

if (Test-Path -LiteralPath $destDir) {
    Write-Host ("Clearing destination: {0}" -f $destDir)
    Get-ChildItem -LiteralPath $destDir -File -Force -ErrorAction SilentlyContinue |
        Where-Object { $_.Extension -ieq '.step' -or $_.Extension -ieq '.stp' } |
        Remove-Item -Force -ErrorAction Stop
}
else {
    New-Item -Path $destDir -ItemType Directory -Force | Out-Null
}

$sourceFiles = @(
    Get-ChildItem -LiteralPath $sourceDir -File -ErrorAction Stop |
    Where-Object { $_.Extension -ieq '.step' -or $_.Extension -ieq '.stp' }
)

Write-Host ("Source STEP/STP files: {0}" -f $sourceFiles.Count)

$copied = 0
foreach ($file in $sourceFiles) {
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($file.Name)
    if ($stem -match '^(?<key>.+?_step_\d+)') {
        $key = $Matches['key'].ToLowerInvariant()
        if ($allow.Contains($key)) {
            $destFile = Join-Path -Path $destDir -ChildPath $file.Name
            Copy-Item -LiteralPath $file.FullName -Destination $destFile -Force -ErrorAction Stop
            $copied++
        }
    }
}

$verified = @(
    Get-ChildItem -LiteralPath $destDir -File -ErrorAction Stop |
    Where-Object { $_.Extension -ieq '.step' -or $_.Extension -ieq '.stp' }
).Count

Write-Host '============================================================'
Write-Host ("Matched+copied : {0}" -f $copied)
Write-Host ("Verified in dest: {0}" -f $verified)
Write-Host '============================================================'

if ($copied -ne $verified) {
    throw ("Copy/verify mismatch: copied={0} verified={1}" -f $copied, $verified)
}

& icacls.exe $destDir /grant 'Everyone:(OI)(CI)F' /T /C /Q | Out-Null
""".strip()

# Indent PS script for readability inside Jenkinsfile
ps_indented = "\n".join(
    ("                                            " + line if line else "")
    for line in ps_script.splitlines()
)

parts = []
parts.append("// ============================================================================")
parts.append("// Job: Filter-abc_steps-No-Thread-Text")
parts.append("// ============================================================================")
parts.append("// Allowlist is EMBEDDED below and written to each agent WORKSPACE.")
parts.append("// (The 10 VMs cannot read \\\\LP76-RZA2-DSA\\jsons\\...)")
parts.append("// Then copies matching STEPs: C:\\abc_steps -> C:\\abc_steps_filtered")
parts.append(f"// Allowlist key count: {key_count}")
parts.append("// ============================================================================")
parts.append("")
parts.append("def OLD_NODES = [")
for h in [
    "walswkqa19383",
    "walswkqa19382",
    "walswkqa19381",
    "walswkqa19380",
    "walswkqa19374",
    "walswkqa19437",
    "walswkqa19438",
    "walswkqa19439",
    "walswkqa19440",
    "walswkqa19441",
]:
    parts.append(f"    '{h}',")
parts[-1] = parts[-1].rstrip(",")
parts.append("]")
parts.append("")
parts.append("def SOURCE_DIR = 'C:\\\\abc_steps'")
parts.append("def DEST_DIR = 'C:\\\\abc_steps_filtered'")
parts.append("")
parts.append("// Allowlist split into 3 parts (Jenkins-friendly); joined before writeFile.")
for i, chunk in enumerate(chunks, start=1):
    parts.append(f"def ALLOWLIST_P{i} = '''")
    if chunk:
        parts.append("\n".join(chunk))
    parts.append("'''")
    parts.append("")
parts.append("def ALLOWLIST_TEXT = ALLOWLIST_P1 + '\\n' + ALLOWLIST_P2 + '\\n' + ALLOWLIST_P3")
parts.append("")
parts.append("pipeline {")
parts.append("    agent none")
parts.append("")
parts.append("    options {")
parts.append("        timestamps()")
parts.append("        disableConcurrentBuilds()")
parts.append("        skipDefaultCheckout(true)")
parts.append("    }")
parts.append("")
parts.append("    stages {")
parts.append("        stage('0 - Info') {")
parts.append("            steps {")
parts.append("                echo '================================================================'")
parts.append("                echo 'FILTER abc_steps -> abc_steps_filtered (no confident Thread/Text)'")
parts.append("                echo '================================================================'")
parts.append('                echo "Machines   : ${OLD_NODES.size()}"')
parts.append('                echo "Source     : ${SOURCE_DIR}"')
parts.append('                echo "Dest       : ${DEST_DIR}"')
parts.append(
    f'                echo "Allowlist  : embedded ({key_count} keys) -> WORKSPACE/allowed_step_keys.txt"'
)
parts.append("                echo '================================================================'")
parts.append("            }")
parts.append("        }")
parts.append("")
parts.append("        stage('1 - Filter on each machine') {")
parts.append("            steps {")
parts.append("                script {")
parts.append("                    def branches = [:]")
parts.append("")
parts.append("                    OLD_NODES.each { selectedNode ->")
parts.append("                        def nodeName = selectedNode")
parts.append("                        branches[nodeName] = {")
parts.append("                            catchError(")
parts.append("                                buildResult: 'UNSTABLE',")
parts.append("                                stageResult: 'FAILURE',")
parts.append('                                message: "Filter failed on ${nodeName}"')
parts.append("                            ) {")
parts.append("                                timeout(time: 30, unit: 'MINUTES') {")
parts.append("                                    node(nodeName) {")
parts.append(
            "                                        writeFile file: 'allowed_step_keys.txt', text: ALLOWLIST_TEXT"
        )
parts.append("                                        withEnv([")
parts.append('                                            "SOURCE_DIR=${SOURCE_DIR}",')
parts.append('                                            "DEST_DIR=${DEST_DIR}",')
parts.append('                                            "TARGET_NODE=${nodeName}"')
parts.append("                                        ]) {")
parts.append("                                            powershell '''")
parts.append(ps_indented)
parts.append("                                            '''")
parts.append("                                        }")
parts.append("                                    }")
parts.append("                                }")
parts.append("                            }")
parts.append("                        }")
parts.append("                    }")
parts.append("")
parts.append("                    parallel branches")
parts.append("                }")
parts.append("            }")
parts.append("        }")
parts.append("")
parts.append("        stage('2 - Summary counts') {")
parts.append("            steps {")
parts.append("                script {")
parts.append("                    def rows = []")
parts.append("                    def total = 0")
parts.append("")
parts.append("                    OLD_NODES.each { selectedNode ->")
parts.append("                        def nodeName = selectedNode")
parts.append("                        try {")
parts.append("                            timeout(time: 2, unit: 'MINUTES') {")
parts.append("                                node(nodeName) {")
parts.append('                                    withEnv(["DEST_DIR=${DEST_DIR}"]) {')
parts.append("                                        def countText = powershell(")
parts.append("                                            returnStdout: true,")
parts.append("                                            script: '''")
parts.append("$ErrorActionPreference = 'Stop'")
parts.append("$dir = $env:DEST_DIR")
parts.append("if (-not (Test-Path -LiteralPath $dir)) { Write-Output 0; exit 0 }")
parts.append("$n = @(")
parts.append("  Get-ChildItem -LiteralPath $dir -File |")
parts.append("  Where-Object { $_.Extension -ieq '.step' -or $_.Extension -ieq '.stp' }")
parts.append(").Count")
parts.append("Write-Output $n")
parts.append("                                            '''")
parts.append("                                        ).trim()")
parts.append("                                        def n = countText.toInteger()")
parts.append("                                        total += n")
parts.append("                                        rows << [node: nodeName, count: n, status: 'OK']")
parts.append(
            '                                        echo "[OK] ${nodeName}: ${n} files in abc_steps_filtered"'
        )
parts.append("                                    }")
parts.append("                                }")
parts.append("                            }")
parts.append("                        } catch (Exception e) {")
parts.append("                            rows << [node: nodeName, count: 0, status: 'FAILED']")
parts.append('                            echo "[FAILED] ${nodeName}: ${e.getMessage()}"')
parts.append("                        }")
parts.append("                    }")
parts.append("")
parts.append("                    echo '================================================================'")
parts.append("                    echo 'FILTER SUMMARY'")
parts.append("                    echo '================================================================'")
parts.append("                    rows.each { r ->")
parts.append(
            '                        echo "${r.node.padRight(20)} ${r.count.toString().padLeft(6)}  ${r.status}"'
        )
parts.append("                    }")
parts.append('                    echo "TOTAL filtered STEPs: ${total}"')
parts.append("                    echo '================================================================'")
parts.append("                }")
parts.append("            }")
parts.append("        }")
parts.append("    }")
parts.append("")
parts.append("    post {")
parts.append("        always {")
parts.append("            echo 'Filter-abc-steps-No-Thread-Text finished.'")
parts.append("        }")
parts.append("    }")
parts.append("}")
parts.append("")

out = ROOT / "Jenkinsfile.filter_abc_steps_no_thread_text"
out.write_text("\n".join(parts), encoding="utf-8")
print(f"Wrote {out}")
print(f"Size bytes: {out.stat().st_size}")
print(f"Allowlist keys: {key_count}")

text = out.read_text(encoding="utf-8")
assert "def ALLOWLIST_P1 = '''" in text
assert "def ALLOWLIST_P2 = '''" in text
assert "def ALLOWLIST_P3 = '''" in text
assert "ALLOWLIST_TEXT = ALLOWLIST_P1" in text
assert "writeFile file: 'allowed_step_keys.txt'" in text
assert "00000014_5b1c2f8a8c6f40fdaae1e69d_step_000" in text
assert "_step_\\d+" in text
print("Sanity checks OK")
print(f"Chunk sizes: {[len(c) for c in chunks]}")
