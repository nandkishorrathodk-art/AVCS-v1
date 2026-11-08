# AVCS Ace Desktop Autopilot Launcher (PowerShell)
$ScriptDir = $PSScriptRoot
$VenvPython = Join-Path $ScriptDir ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Error "Virtual environment not found at $VenvPython. Please set up .venv first."
    exit 1
}

& $VenvPython -m avcs.ace.run @args
exit $LASTEXITCODE

