#Requires -Version 5.1
<#
.SYNOPSIS
    Push local changes to GitHub, then trigger pull-based update on all nodes.

.DESCRIPTION
    1. Commits any staged/unstaged changes (with an optional message)
    2. Pushes the branch to origin (GitHub)
    3. SSHs into each node and runs node_update.sh (git pull + pip install if
       requirements.txt changed). Xeon gets --gpu flag automatically.

    Run from the repo root:
        .\scripts\deploy_to_pi.ps1
        .\scripts\deploy_to_pi.ps1 -Message "add satellite package"
        .\scripts\deploy_to_pi.ps1 -PushOnly          # push to GitHub only, skip nodes
        .\scripts\deploy_to_pi.ps1 -PullOnly           # skip git push, update nodes only
        .\scripts\deploy_to_pi.ps1 -Nodes pi           # update Pi only
        .\scripts\deploy_to_pi.ps1 -Nodes xeon         # update Xeon only
        .\scripts\deploy_to_pi.ps1 -DryRun             # preview without making changes

.NOTES
    First run: set up SSH key auth to avoid password prompts every time.
        ssh-keygen -t ed25519 -C "wreckhunter-deploy"
        type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh pi@10.0.0.226 "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
        type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh cesarops@10.0.0.55 "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
#>

param(
    [string]$Message    = "",
    [string]$Branch     = "wreckhuntertools",
    # Comma-separated node names to update. Default: all configured nodes.
    [string]$Nodes      = "pi,xeon",
    [switch]$PushOnly,
    [switch]$PullOnly,
    [switch]$DryRun
)

# ── Node definitions ──────────────────────────────────────────────────────────
# Override via .env: PI_HOST, PI_USER, XENON_HOST, XENON_USER
$NodeDefs = @{
    pi    = @{ host = "10.0.0.226"; user = "pi";        repoDir = "/home/pi/cesarops-core/wreckhunter2000";         gpu = $false }
    xeon  = @{ host = "10.0.0.55";  user = "cesarops";  repoDir = "/home/cesarops/cesarops/cesarops-core";           gpu = $true  }
}

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Repo = Split-Path -Parent $PSScriptRoot

# ── Load .env ────────────────────────────────────────────────────────────────
$dotenv = @{}
$dotenvPath = Join-Path $Repo ".env"
if (Test-Path $dotenvPath) {
    Get-Content $dotenvPath | Where-Object { $_ -match "^\s*[^#].*=" } | ForEach-Object {
        $k, $v = $_ -split "=", 2
        $dotenv[$k.Trim()] = $v.Trim()
    }
}
function Env([string]$Key) {
    $ev = [System.Environment]::GetEnvironmentVariable($Key)
    if ($ev) { return $ev }
    if ($dotenv.ContainsKey($Key)) { return $dotenv[$Key] }
    return $null
}

# Apply .env overrides to node definitions
$h = Env "PI_HOST";    if ($h) { $NodeDefs.pi.host    = $h }
$u = Env "PI_USER";    if ($u) { $NodeDefs.pi.user    = $u }
$h = Env "XENON_HOST"; if ($h) { $NodeDefs.xeon.host = $h }
$u = Env "XENON_USER"; if ($u) { $NodeDefs.xeon.user = $u }

$GitHubPat = Env "GITHUB_PAT"

function Say  { param($Msg, $Color = "Cyan")   Write-Host $Msg -ForegroundColor $Color }
function Warn { param($Msg)                     Write-Host "  WARN $Msg" -ForegroundColor Yellow }
function Die  { param($Msg)                     Write-Host "  FAIL $Msg" -ForegroundColor Red; exit 1 }
function Dry  { param($Cmd) if ($DryRun) { Say "[DRY] $Cmd" "DarkGray"; return $true }; return $false }

# Parse -Nodes list into a set
$targetNodes = @($Nodes -split "[,\s]+" | Where-Object { $_ } | ForEach-Object { $_.ToLower() })

# ── 1 & 2. Push to GitHub ────────────────────────────────────────────────────
if (-not $PullOnly) {
    Say "`n[1/2] Checking for uncommitted changes..."
    Push-Location $Repo

    $status = git status --porcelain 2>&1
    if ($status) {
        $changeCount = @($status).Count
        Say "  $changeCount change(s) found — staging all and committing."
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

    Say "`n[2/2] Pushing $Branch to origin..."
    if (-not (Dry "git push origin $Branch")) {
        git push origin $Branch 2>&1 | ForEach-Object { Say "  $_" }
        if ($LASTEXITCODE -ne 0) { Die "git push failed (exit $LASTEXITCODE)" }
    }

    Pop-Location
    Say "  Push OK." "Green"
} else {
    Say "[1-2] Skipped (--PullOnly)."
}

# ── 3. Update each target node via node_update.sh ────────────────────────────
if (-not $PushOnly) {
    $nodeIndex = 0
    foreach ($nodeName in $targetNodes) {
        $nodeIndex++
        if (-not $NodeDefs.ContainsKey($nodeName)) {
            Warn "Unknown node '$nodeName' — skipping."
            continue
        }
        $nd       = $NodeDefs[$nodeName]
        $nodeHost = $nd.host
        $nodeUser = $nd.user
        $dir      = $nd.repoDir
        $gpu      = $nd.gpu
        $target   = "${nodeUser}@${nodeHost}"

        Say "`n[node $nodeIndex/$($targetNodes.Count)] Updating $nodeName ($target)..."

        # Build the remote command
        $scriptPath  = "$dir/scripts/node_update.sh"
        $gpuFlag     = if ($gpu) { "--gpu" } else { "" }
        $dryRunFlag  = if ($DryRun) { "--dry-run" } else { "" }
        $remoteCmd   = "bash '$scriptPath' $gpuFlag $dryRunFlag".Trim()

        # If node_update.sh might not exist yet (fresh node), upload it first
        $uploadScript = {
            Say "  Bootstrapping: uploading node_update.sh to $target..."
            $localScript = Join-Path $Repo "scripts\node_update.sh"
            if (-not (Test-Path $localScript)) {
                Warn "scripts\node_update.sh not found locally — cannot bootstrap $nodeName."
                return
            }
            # Ensure remote dir exists and upload
            ssh -o ConnectTimeout=10 $target "mkdir -p '$dir/scripts'"
            scp -o ConnectTimeout=10 $localScript "${target}:${dir}/scripts/node_update.sh"
            ssh -o ConnectTimeout=10 $target "chmod +x '$dir/scripts/node_update.sh'"
        }

        if (Dry "ssh $target '$remoteCmd'") { continue }

        try {
            # Check if node_update.sh exists on the remote node first
            $exists = ssh -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new `
                          $target "test -f '$scriptPath' && echo yes || echo no" 2>&1
            if ($exists -notmatch "yes") {
                & $uploadScript
            }

            # Run node_update.sh
            $out = ssh -o ConnectTimeout=8 -o BatchMode=no $target $remoteCmd 2>&1
            $out | ForEach-Object { Say "  $_" }

            if ($LASTEXITCODE -ne 0) {
                Warn "$nodeName node_update.sh exited with code $LASTEXITCODE."
            } else {
                Say "  $nodeName is up to date." "Green"
            }
        } catch {
            Warn "$nodeName ($nodeHost) unreachable or update failed — skipping."
            Warn "  $_"
        }
    }
} else {
    Say "[nodes] Skipped (--PushOnly)."
}

Say "`nDone." "Green"
