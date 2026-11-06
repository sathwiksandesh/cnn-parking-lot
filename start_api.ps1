# Start the CNN Parking Lot API using the correct local venv
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ScriptDir ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    Write-Error "venv not found at $Python. Run: python -m venv .venv && .venv\Scripts\pip install -r requirements-api.txt"
    exit 1
}

Write-Host "Using Python: $Python" -ForegroundColor Cyan
& $Python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

