$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

& ".\venv\Scripts\python.exe" ".\scripts\train_model.py"
