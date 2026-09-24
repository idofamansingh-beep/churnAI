$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

& ".\venv\Scripts\python.exe" ".\scripts\build_database.py"
