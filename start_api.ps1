# CESAROPS Wrecks API — Local Dev Launcher
# Run this from the repo root to start the backend on port 8099.
#
# Usage:
#   .\start_api.ps1              # start on default port 8099
#   .\start_api.ps1 -Port 8099  # explicit port
#
# The Kobold panel and Search panel both call http://localhost:8099
# KoboldCPP itself runs separately on port 5001 (see launch_kobold_*.sh)

param(
    [int]$Port = 8099
)

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════╗"
Write-Host "║  CESAROPS Wrecks API — Local Dev                ║"
Write-Host "╚══════════════════════════════════════════════════╝"
Write-Host ""
Write-Host "  Starting on http://localhost:$Port"
Write-Host "  Kobold endpoints: http://localhost:$Port/kobold/*"
Write-Host "  Search endpoint:  http://localhost:$Port/search"
Write-Host "  Health check:     http://localhost:$Port/health"
Write-Host ""
Write-Host "  Press Ctrl+C to stop."
Write-Host ""

# Activate venv if present
if (Test-Path ".venv\Scripts\Activate.ps1") {
    & .venv\Scripts\Activate.ps1
}

# Set env so the API finds the local DB
$env:DB_PATH = "$PSScriptRoot\db\wrecks.db"
$env:API_BASE_URL = "http://localhost:$Port"

uvicorn wrecks_api.app:app --host 0.0.0.0 --port $Port --reload
