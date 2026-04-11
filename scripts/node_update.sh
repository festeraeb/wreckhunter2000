#!/usr/bin/env bash
# scripts/node_update.sh — Pull-based deploy for CESARops/WreckHunter nodes
#
# Run on any node to self-update from GitHub:
#   bash ~/cesarops-core/wreckhunter2000/scripts/node_update.sh
#   bash ~/cesarops-core/wreckhunter2000/scripts/node_update.sh --gpu   # Xenon
#   bash ~/cesarops-core/wreckhunter2000/scripts/node_update.sh --dry-run
#
# Called remotely by deploy_to_pi.ps1 / deploy_and_scan.py via SSH.
# Reads GITHUB_PAT from ~/.env or environment for private repo access.
#
# Environment variables (optional, read from ~/.env or env):
#   GITHUB_PAT       — GitHub personal access token for private repo pull
#   CESAROPS_BRANCH  — branch to track (default: wreckhuntertools)
#   CESAROPS_REPO    — git remote URL (default: https://github.com/festeraeb/wreckhunter2000.git)
#   CESAROPS_DIR     — local repo path (auto-detected if not set)
#   VENV_DIR         — path to Python venv (uses system Python if not set)

set -euo pipefail

# ── Args ────────────────────────────────────────────────────────────────────
GPU=0
DRY=0
for arg in "$@"; do
    case "$arg" in
        --gpu)     GPU=1 ;;
        --dry-run) DRY=1 ;;
        *) echo "Unknown arg: $arg" >&2; exit 1 ;;
    esac
done

# ── Load ~/.env if present ───────────────────────────────────────────────────
if [[ -f "$HOME/.env" ]]; then
    # shellcheck disable=SC1091
    set -a; source "$HOME/.env"; set +a
fi

# ── Config ───────────────────────────────────────────────────────────────────
BRANCH="${CESAROPS_BRANCH:-wreckhuntertools}"
REMOTE_URL="${CESAROPS_REPO:-https://github.com/festeraeb/wreckhunter2000.git}"

# Embed PAT for private repo access if available
if [[ -n "${GITHUB_PAT:-}" ]]; then
    AUTHED_URL="${REMOTE_URL/https:\/\//https://${GITHUB_PAT}@}"
else
    AUTHED_URL="$REMOTE_URL"
fi

# Auto-detect repo dir: prefer CESAROPS_DIR env, then walk common paths
if [[ -n "${CESAROPS_DIR:-}" ]]; then
    REPO_DIR="$CESAROPS_DIR"
else
    CANDIDATES=(
        "$HOME/cesarops-core/wreckhunter2000"
        "$HOME/cesarops/cesarops-core"
        "$HOME/wreckhunter2000"
    )
    REPO_DIR=""
    for c in "${CANDIDATES[@]}"; do
        if [[ -d "$c/.git" ]]; then
            REPO_DIR="$c"
            break
        fi
    done
fi

# ── Clone if repo not found ──────────────────────────────────────────────────
if [[ -z "$REPO_DIR" || ! -d "$REPO_DIR/.git" ]]; then
    REPO_DIR="$HOME/cesarops-core/wreckhunter2000"
    echo "[node_update] Repo not found — cloning to $REPO_DIR"
    if [[ $DRY -eq 1 ]]; then
        echo "[DRY] git clone -b $BRANCH $REMOTE_URL $REPO_DIR"
    else
        mkdir -p "$(dirname "$REPO_DIR")"
        git clone -b "$BRANCH" "$AUTHED_URL" "$REPO_DIR"
        # Remove PAT from stored remote URL
        git -C "$REPO_DIR" remote set-url origin "$REMOTE_URL"
    fi
fi

echo "[node_update] Repo: $REPO_DIR  Branch: $BRANCH"

# ── Git pull ─────────────────────────────────────────────────────────────────
PREV_HASH=$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || echo "none")
PREV_REQ_HASH=$(sha256sum "$REPO_DIR/requirements.txt" 2>/dev/null | cut -d' ' -f1 || echo "none")

if [[ $DRY -eq 1 ]]; then
    echo "[DRY] git -C $REPO_DIR fetch origin && git checkout $BRANCH && git pull"
else
    # Temporarily set authed URL, fetch, restore clean URL
    git -C "$REPO_DIR" remote set-url origin "$AUTHED_URL"
    git -C "$REPO_DIR" fetch origin
    git -C "$REPO_DIR" checkout "$BRANCH"
    git -C "$REPO_DIR" pull origin "$BRANCH"
    git -C "$REPO_DIR" remote set-url origin "$REMOTE_URL"
fi

NEW_HASH=$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || echo "none")
NEW_REQ_HASH=$(sha256sum "$REPO_DIR/requirements.txt" 2>/dev/null | cut -d' ' -f1 || echo "none")

if [[ "$PREV_HASH" == "$NEW_HASH" ]]; then
    echo "[node_update] Already up to date ($NEW_HASH)"
else
    echo "[node_update] Updated $PREV_HASH → $NEW_HASH"
fi

# ── Pip install (only if requirements.txt changed or new install) ─────────────
REQ_CHANGED=0
if [[ "$PREV_REQ_HASH" != "$NEW_REQ_HASH" ]]; then
    REQ_CHANGED=1
    echo "[node_update] requirements.txt changed — running pip install"
fi
if [[ "$PREV_HASH" == "none" ]]; then
    REQ_CHANGED=1
    echo "[node_update] Fresh install — running pip install"
fi

if [[ $REQ_CHANGED -eq 1 ]]; then
    # Resolve python: prefer venv if set, else system python3
    if [[ -n "${VENV_DIR:-}" && -f "$VENV_DIR/bin/pip" ]]; then
        PIP="$VENV_DIR/bin/pip"
        PYTHON="$VENV_DIR/bin/python"
    elif [[ -f "$HOME/cesarops/venv/bin/pip" ]]; then
        PIP="$HOME/cesarops/venv/bin/pip"
        PYTHON="$HOME/cesarops/venv/bin/python"
    else
        PIP="python3 -m pip"
        PYTHON="python3"
        # PEP 668 — Raspberry Pi OS requires --break-system-packages for system Python
        PIPFLAGS="--break-system-packages --quiet"
    fi
    PIPFLAGS="${PIPFLAGS:---quiet}"

    if [[ $DRY -eq 1 ]]; then
        echo "[DRY] $PIP install $PIPFLAGS -r $REPO_DIR/requirements.txt"
        [[ $GPU -eq 1 ]] && echo "[DRY] $PIP install $PIPFLAGS -r $REPO_DIR/requirements-gpu.txt"
    else
        # shellcheck disable=SC2086
        $PIP install $PIPFLAGS -r "$REPO_DIR/requirements.txt"
        if [[ $GPU -eq 1 ]]; then
            echo "[node_update] Installing GPU extras..."
            # shellcheck disable=SC2086
            $PIP install $PIPFLAGS -r "$REPO_DIR/requirements-gpu.txt"
        fi
        echo "[node_update] pip install complete"
    fi
else
    echo "[node_update] requirements.txt unchanged — skipping pip install"
fi

echo "[node_update] Done on $(hostname)"
