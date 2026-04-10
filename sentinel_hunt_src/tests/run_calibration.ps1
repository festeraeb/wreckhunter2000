# ─── Sentinel-Hunt Calibration Test Runner ───────────────────────────
# Tests the detection pipeline on known wreck sites to validate
# that detectors can see real wrecks before hunting for M&B.
#
# Targets:
#   1. SS Cedarville  — 588ft steel, 32m depth, Straits of Mackinac
#   2. Big Tub Harbor  — Sweepstakes (6m) + City of Grand Rapids (3m)
#
# Usage:  .\tests\run_calibration.ps1
# ──────────────────────────────────────────────────────────────────────

$ErrorActionPreference = "Continue"

# ── Windows SDK env (use 10.0.22621.0, NOT broken 10.0.26100.0) ──────
$env:INCLUDE = "C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\ucrt;C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\um;C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\shared;C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\include"
$env:LIB = "C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\ucrt\x64;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\um\x64;C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\lib\x64"

# ── Paths ─────────────────────────────────────────────────────────────
$root   = "C:\Users\thomf\programming\Bagrecovery\sentinel_hunt"
$bin    = "$root\target\release\sentinel-hunt.exe"
$python = "C:\Users\thomf\miniconda3\envs\wh2k\python.exe"
$outdir = "$root\output\calibration"

Set-Location $root

# ── Verify prerequisites ─────────────────────────────────────────────
if (-not (Test-Path $bin)) {
    Write-Host "Binary not found, building release..." -ForegroundColor Yellow
    cargo build --release
    if ($LASTEXITCODE -ne 0) { Write-Error "Build failed"; exit 1 }
}

if (-not (Test-Path "db\wrecks.db")) {
    Write-Error "Wrecks database not found at db\wrecks.db"; exit 1
}

New-Item -ItemType Directory -Force -Path $outdir | Out-Null
New-Item -ItemType Directory -Force -Path "cache" | Out-Null

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║         SENTINEL-HUNT CALIBRATION TEST SUITE                ║" -ForegroundColor Cyan
Write-Host "╠══════════════════════════════════════════════════════════════╣" -ForegroundColor Cyan
Write-Host "║  Target 1: SS Cedarville (45.65N, 84.33W) — 32m, steel     ║" -ForegroundColor Cyan
Write-Host "║  Target 2: Big Tub Harbor (45.26N, 81.66W) — 3-6m, mixed   ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# ═════════════════════════════════════════════════════════════════════
# TEST 1: STAC Search — verify we can find Sentinel scenes
# ═════════════════════════════════════════════════════════════════════

Write-Host "━━━ TEST 1: STAC Scene Search ━━━" -ForegroundColor Green

Write-Host "`n[1a] Searching for SAR scenes over Cedarville..." -ForegroundColor Yellow
& $bin search --lake huron --method dark-spot --start 2024-07-01 --end 2024-09-30 --max-scenes 3 --python $python 2>&1
Write-Host "  Exit code: $LASTEXITCODE" -ForegroundColor DarkGray

Write-Host "`n[1b] Searching for optical scenes over Tobermory..." -ForegroundColor Yellow
& $bin search --lake huron --method clear-hole --start 2024-07-01 --end 2024-09-30 --max-scenes 3 --python $python 2>&1
Write-Host "  Exit code: $LASTEXITCODE" -ForegroundColor DarkGray

# ═════════════════════════════════════════════════════════════════════
# TEST 2: Detection on Cedarville (deep steel wreck, SAR optimal)
# ═════════════════════════════════════════════════════════════════════

Write-Host "`n━━━ TEST 2: Cedarville Dark Spot Detection (SAR) ━━━" -ForegroundColor Green
Write-Host "  ROI: [-84.40, 45.60, -84.26, 45.70] (~10km x 11km)" -ForegroundColor DarkGray
Write-Host "  Expected: laminar wake / dark spot at 45.65, -84.33" -ForegroundColor DarkGray

& $bin detect `
    --lake huron `
    --method dark-spot `
    --start 2024-07-01 `
    --end 2024-09-30 `
    --max-scenes 5 `
    --bbox -84.40,45.60,-84.26,45.70 `
    --output "$outdir\cedarville_dark_spot.json" `
    --python $python 2>&1

$cedarDarkExit = $LASTEXITCODE
Write-Host "  Exit code: $cedarDarkExit" -ForegroundColor DarkGray

# ═════════════════════════════════════════════════════════════════════
# TEST 3: Clear Hole on Cedarville (optical — mussel filtration)
# ═════════════════════════════════════════════════════════════════════

Write-Host "`n━━━ TEST 3: Cedarville Clear Hole Detection (Optical) ━━━" -ForegroundColor Green
Write-Host "  Looking for invasive mussel clarity anomaly over wreck" -ForegroundColor DarkGray

& $bin detect `
    --lake huron `
    --method clear-hole `
    --start 2024-07-01 `
    --end 2024-09-30 `
    --max-scenes 5 `
    --bbox -84.40,45.60,-84.26,45.70 `
    --output "$outdir\cedarville_clear_hole.json" `
    --python $python 2>&1

$cedarClearExit = $LASTEXITCODE
Write-Host "  Exit code: $cedarClearExit" -ForegroundColor DarkGray

# ═════════════════════════════════════════════════════════════════════
# TEST 4: Big Tub Harbor — Shallow wrecks (optical optimal)
# ═════════════════════════════════════════════════════════════════════

Write-Host "`n━━━ TEST 4: Big Tub Harbor Clear Hole Detection ━━━" -ForegroundColor Green
Write-Host "  ROI: [-81.72, 45.22, -81.62, 45.29] (~8km x 7km)" -ForegroundColor DarkGray
Write-Host "  Expected: Sweepstakes (45.2555, -81.6648, 6m)" -ForegroundColor DarkGray
Write-Host "  Expected: City of Grand Rapids (45.2550, -81.6640, 3m)" -ForegroundColor DarkGray

& $bin detect `
    --lake huron `
    --method clear-hole `
    --start 2024-07-01 `
    --end 2024-09-30 `
    --max-scenes 5 `
    --bbox -81.72,45.22,-81.62,45.29 `
    --output "$outdir\tobermory_clear_hole.json" `
    --python $python 2>&1

$tobClearExit = $LASTEXITCODE
Write-Host "  Exit code: $tobClearExit" -ForegroundColor DarkGray

# ═════════════════════════════════════════════════════════════════════
# TEST 5: Sediment Trap on Big Tub (turbidity wake)
# ═════════════════════════════════════════════════════════════════════

Write-Host "`n━━━ TEST 5: Big Tub Sediment Trap Detection ━━━" -ForegroundColor Green
Write-Host "  Looking for persistent turbid anomalies from current deflection" -ForegroundColor DarkGray

& $bin detect `
    --lake huron `
    --method sediment-trap `
    --start 2024-04-01 `
    --end 2024-05-31 `
    --max-scenes 5 `
    --bbox -81.72,45.22,-81.62,45.29 `
    --output "$outdir\tobermory_sediment_trap.json" `
    --python $python 2>&1

$tobSedExit = $LASTEXITCODE
Write-Host "  Exit code: $tobSedExit" -ForegroundColor DarkGray

# ═════════════════════════════════════════════════════════════════════
# TEST 6: Validation against known wrecks
# ═════════════════════════════════════════════════════════════════════

Write-Host "`n━━━ TEST 6: Validation Against Known Wrecks ━━━" -ForegroundColor Green

$resultFiles = Get-ChildItem "$outdir\*.json" -ErrorAction SilentlyContinue
foreach ($f in $resultFiles) {
    Write-Host "`n  Validating $($f.Name)..." -ForegroundColor Yellow
    & $bin validate --results $f.FullName --lake huron --python $python 2>&1
}

# ═════════════════════════════════════════════════════════════════════
# TEST 7: KMZ Export for Google Earth
# ═════════════════════════════════════════════════════════════════════

Write-Host "`n━━━ TEST 7: KMZ Export ━━━" -ForegroundColor Green

foreach ($f in $resultFiles) {
    $kmzName = $f.BaseName + ".kmz"
    $kmzPath = "$outdir\$kmzName"
    Write-Host "  Exporting $($f.Name) → $kmzName" -ForegroundColor Yellow
    & $bin kml --results $f.FullName --output $kmzPath 2>&1
}

# ═════════════════════════════════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════════════════════════════════

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║                    CALIBRATION SUMMARY                      ║" -ForegroundColor Cyan
Write-Host "╠══════════════════════════════════════════════════════════════╣" -ForegroundColor Cyan

$tests = @(
    @{Name="Cedarville Dark Spot (SAR)"; Exit=$cedarDarkExit; File="cedarville_dark_spot.json"},
    @{Name="Cedarville Clear Hole (Opt)"; Exit=$cedarClearExit; File="cedarville_clear_hole.json"},
    @{Name="Tobermory Clear Hole (Opt)"; Exit=$tobClearExit; File="tobermory_clear_hole.json"},
    @{Name="Tobermory Sediment Trap"; Exit=$tobSedExit; File="tobermory_sediment_trap.json"}
)

foreach ($t in $tests) {
    $statusColor = if ($t.Exit -eq 0) { "Green" } else { "Red" }
    $statusIcon  = if ($t.Exit -eq 0) { "PASS" } else { "FAIL" }
    $detCount = 0
    $jsonFile = "$outdir\$($t.File)"
    if (Test-Path $jsonFile) {
        try {
            $json = Get-Content $jsonFile -Raw | ConvertFrom-Json
            foreach ($scene in $json) {
                $detCount += $scene.detections.Count
            }
        } catch {}
    }
    Write-Host ("║  [{0}] {1,-30} {2,4} detections" -f $statusIcon, $t.Name, $detCount) -ForegroundColor $statusColor
}

Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan

# Files produced
Write-Host "`nOutput files:" -ForegroundColor Yellow
Get-ChildItem $outdir -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host ("  {0,-45} {1,8:N0} bytes" -f $_.Name, $_.Length)
}

Write-Host "`nDone. Check output\calibration\ for results." -ForegroundColor Green
