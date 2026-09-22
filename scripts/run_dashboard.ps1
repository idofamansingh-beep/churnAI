$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

& ".\venv\Scripts\streamlit.exe" run ".\app\dashboard\app.py"
