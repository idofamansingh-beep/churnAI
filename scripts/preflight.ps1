$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

& ".\venv\Scripts\python.exe" -m pip check
& ".\venv\Scripts\python.exe" -m py_compile `
    ".\src\__init__.py" `
    ".\src\churnai_core.py" `
    ".\src\churn_db.py" `
    ".\src\churn_model.py" `
    ".\app\dashboard\app.py" `
    ".\app\aap.py" `
    ".\scripts\build_database.py" `
    ".\scripts\train_model.py" `
    ".\scripts\check_model_quality.py" `
    ".\tests\test_churnai_core.py" `
    ".\tests\test_churn_db.py" `
    ".\tests\test_churn_model.py" `
    ".\tests\test_dashboard_smoke.py"
& ".\venv\Scripts\python.exe" -m unittest discover -s tests
