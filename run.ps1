<#
Runs the object monitor and HTTP server, then opens the browser.
#>

param(
    [int]$Port = 8000
)

$ErrorActionPreference = 'Stop'

# Activate virtual environment if present
$venvActivate = Join-Path $PSScriptRoot '.venv\Scripts\Activate.ps1'
if (Test-Path $venvActivate) {
    & $venvActivate
}

Write-Host "Starting simple_monitor in new window…"
Start-Process powershell -ArgumentList @('-NoExit', '-Command', 'python -m clothing_vision.simple_monitor')

Start-Sleep -Seconds 2

Write-Host "Starting HTTP server on port $Port…"
Start-Process powershell -ArgumentList @('-NoExit', '-Command', "uvicorn clothing_vision.http_server:app --port $Port")

Start-Sleep -Seconds 2

Write-Host "Opening browser…"
Start-Process "http://localhost:$Port/"
