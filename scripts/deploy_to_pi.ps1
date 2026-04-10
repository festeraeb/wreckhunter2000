#Requires -Version 5.1
<#
.SYNOPSIS
    Push local changes to GitHub and pull them on the Pi (cesarops-node).

.DESCRIPTION
    1. Commits any staged/unstaged changes (with an optional message)
    2. Pushes wreckhuntertools to origin (GitHub)
    3. SSH's into the Pi and pulls the same branch

    Run from the repo root:
        .\scripts\deploy_to_pi.ps1
        .\scripts\deploy_to_pi.ps1 -Message "add satellite package"
        .\scripts\deploy_to_pi.ps1 -PiHost 10.0.0.226 -PiUser pi
        .\scripts\deploy_to_pi.ps1 -PushOnly        # skip Pi pull
        .\scripts\deploy_to_pi.ps1 -PullOnly        # skip git push, Pi pull only

.NOTES
    First run: set up SSH key auth to avoid password prompts every time.
        ssh-keygen -t ed25519 -C "wreckhunter-deploy"
        type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh pi@10.0.0.226 "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
#>

param(
    [string]$Message    = "",
    [string]$Branch     = "wreckhuntertools",
    [string]$PiHost     = "10.0.0.226",
    [string]$PiUser     = "pi",
    [string]$PiRepoPath = "",        # leave blank to auto-detect on first run
    [switch]$PushOnly,
    [switch]$PullOnly,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Repo = Split-Path -Parent $PSScriptRoot

function Say { param($Msg, $Color = "Cyan") Write-Host $Msg -ForegroundColor $Color }
function Warn { param($Msg) Write-Host "  WARN $Msg" -ForegroundColor Yellow }
function Die  { param($Msg) Write-Host "  FAIL $Msg" -ForegroundColor Red; exit 1 }
function Dry  { param($Cmd) if ($DryRun) { Say "[DRY] $Cmd" "DarkGray"; return $true } return $false }

# ── 1. Push to GitHub ────────────────────────────────────────────────────────
if (-not $PullOnly) {
    Say "`n[1/3] Checking for uncommitted changes..."
    Push-Location $Repo

    $status = git status --porcelain 2>&1
    if ($status) {
        Say "  $($status.Count) change(s) found — staging all and committing."
        if (-not (Dry "git add -A")) { git add -A }

        $commitMsg = if ($Message) { $Message } else {
            "deploy: $(Get-Date -Format 'yyyy-MM-dd HH:mm') auto-commit"
        }
        if (-not (Dry "git commit -m '$commitMsg'")) {
            git commit -m $commitMsg
        }
    } else {
        Say "  Working tree clean — nothing to commit."
    }

    Say "`n[2/3] Pushing $Branch to origin..."
    if (-not (Dry "git push origin $Branch")) {
        git push origin $Branch 2>&1 | ForEach-Object { Say "  $_" }
        if ($LASTEXITCODE -ne 0) { Die "git push failed (exit $LASTEXITCODE)" }
    }

    Pop-Location
    Say "  Push OK." "Green"
} else {
    Say "[1-2/3] Skipped (--PullOnly)."
}

# ── 2. Pull on the Pi ────────────────────────────────────────────────────────
if (-not $PushOnly) {
    $pi = "${PiUser}@${PiHost}"

    # Auto-detect repo path on Pi if not provided
    if (-not $PiRepoPath) {
        Say "`n[3/3] Locating repo on Pi..."
        $findCmd = 'find /home /opt -maxdepth 5 -name ".git" -type d 2>/dev/null | head -5'
        if (Dry "ssh $pi '$findCmd'") { exit 0 }
        try {
            $found = ssh -o ConnectTimeout=8 $pi $findCmd 2>&1
        } catch {
            Die "SSH to $pi failed. Check host, user, and key auth.`n  $_"
        }

        # Filter to repo roots that contain wreckhunter2000 in the path or have matching remote
        $repoRoot = $null
        foreach ($gitDir in $found -split "`n" | Where-Object { $_ -match "\.git$" }) {
            $dir = $gitDir -replace "/\.git$", ""
            $remote = ssh -o ConnectTimeout=5 $pi "git -C '$dir' remote get-url origin 2>/dev/null" 2>&1
            if ($remote -match "wreckhunter2000") {
                $repoRoot = $dir
                break
            }
        }

        if (-not $repoRoot) {
            Warn "Repo not found on Pi. Clone it first:"
            Warn "  ssh ${pi}"
            Warn "  git clone https://github.com/festeraeb/wreckhunter2000.git ~/wreckhunter2000"
            Warn "  cd ~/wreckhunter2000 && git checkout $Branch"
            exit 0
        }
        Say "  Found repo at: $repoRoot"
    } else {
        $repoRoot = $PiRepoPath
    }

    Say "`n[3/3] Pulling $Branch on Pi ($pi : $repoRoot)..."
    $pullCmd = "cd '$repoRoot' && git fetch origin && git checkout $Branch && git pull origin $Branch"
    if (-not (Dry "ssh $pi '$pullCmd'")) {
        $out = ssh -o ConnectTimeout=10 $pi $pullCmd 2>&1
        $out | ForEach-Object { Say "  $_" }
        if ($LASTEXITCODE -ne 0) { Die "git pull on Pi failed (exit $LASTEXITCODE)" }
    }

    Say "  Pi is up to date." "Green"
} else {
    Say "[3/3] Skipped (--PushOnly)."
}

Say "`nDone." "Green"
