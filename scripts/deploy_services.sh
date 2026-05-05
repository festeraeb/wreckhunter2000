#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# CESAROPS Service Installer
# Installs and enables wrecks-api, koboldcpp, scan-worker, and watchdog
# as systemd services that start on boot and restart on crash.
#
# Run on the target Linux node (i7, T440, P1000):
#   bash scripts/deploy_services.sh
#
# Options:
#   --host USER@HOST   — deploy via SSH instead of running locally
#   --restart          — restart all services (don't reinstall)
#   --status           — show service status
#   --logs             — tail all service logs
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_DIR="/home/cesarops/wreckhunter2000-1"
VENV="/home/cesarops/tpu-venv"
SERVICES=(wrecks-api koboldcpp scan-worker cesarops-watchdog)

# ── Parse args ────────────────────────────────────────────────────────────────
HOST=""
ACTION="install"
for arg in "$@"; do
    case "$arg" in
        --host=*) HOST="${arg#--host=}" ;;
        --host)   shift; HOST="$1" ;;
        --restart) ACTION="restart" ;;
        --status)  ACTION="status" ;;
        --logs)    ACTION="logs" ;;
    esac
done

# ── SSH wrapper ───────────────────────────────────────────────────────────────
run() {
    if [ -n "$HOST" ]; then
        ssh -o ConnectTimeout=10 "$HOST" "$@"
    else
        bash -c "$@"
    fi
}

scp_file() {
    local src="$1" dst="$2"
    if [ -n "$HOST" ]; then
        scp "$src" "$HOST:$dst"
    else
        cp "$src" "$dst"
    fi
}

# ── Status / logs shortcuts ───────────────────────────────────────────────────
if [ "$ACTION" = "status" ]; then
    for svc in "${SERVICES[@]}"; do
        echo "── $svc ──────────────────────────────────────────"
        run "systemctl status $svc --no-pager -l 2>/dev/null || echo '  not installed'"
    done
    exit 0
fi

if [ "$ACTION" = "logs" ]; then
    run "journalctl -u wrecks-api -u koboldcpp -u scan-worker -u cesarops-watchdog -f --no-pager"
    exit 0
fi

if [ "$ACTION" = "restart" ]; then
    echo "Restarting all CESAROPS services…"
    for svc in "${SERVICES[@]}"; do
        run "sudo systemctl restart $svc 2>/dev/null || true"
        echo "  ✓ $svc restarted"
    done
    sleep 2
    for svc in "${SERVICES[@]}"; do
        STATUS=$(run "systemctl is-active $svc 2>/dev/null || echo unknown")
        echo "  $svc: $STATUS"
    done
    exit 0
fi

# ── Full install ──────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  CESAROPS Service Installer                                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
[ -n "$HOST" ] && echo "  Target: $HOST" || echo "  Target: localhost"
echo ""

# 1. Sync service files
echo "[1/5] Syncing service files…"
for svc in wrecks-api koboldcpp scan-worker cesarops-watchdog; do
    src="$REPO_DIR/scripts/${svc}.service"
    if [ -f "$src" ]; then
        scp_file "$src" "/tmp/${svc}.service"
        run "sudo mv /tmp/${svc}.service /etc/systemd/system/${svc}.service"
        echo "  ✓ /etc/systemd/system/${svc}.service"
    fi
done

# 2. Sync Python files
echo "[2/5] Syncing Python files…"
for f in watchdog.py scan_worker.py scan_queue.py; do
    if [ -f "$REPO_DIR/$f" ]; then
        scp_file "$REPO_DIR/$f" "$REMOTE_DIR/$f"
        echo "  ✓ $f"
    fi
done

# 3. Sync wrecks_api
echo "[3/5] Syncing wrecks_api…"
if [ -n "$HOST" ]; then
    scp "$REPO_DIR/wrecks_api/app.py" "$HOST:$REMOTE_DIR/wrecks_api/app.py"
else
    cp "$REPO_DIR/wrecks_api/app.py" "$REMOTE_DIR/wrecks_api/app.py" 2>/dev/null || true
fi
echo "  ✓ wrecks_api/app.py"

# 4. Install Python deps
echo "[4/5] Installing Python dependencies…"
run "$VENV/bin/pip install uvicorn[standard] fastapi psutil --quiet" || true
echo "  ✓ deps installed"

# 5. Enable and start services
echo "[5/5] Enabling and starting services…"
run "sudo systemctl daemon-reload"

# Start in dependency order
for svc in wrecks-api koboldcpp scan-worker cesarops-watchdog; do
    run "sudo systemctl enable $svc"
    run "sudo systemctl restart $svc" || true
    sleep 1
    STATUS=$(run "systemctl is-active $svc 2>/dev/null || echo unknown")
    ICON="✓"
    [ "$STATUS" != "active" ] && ICON="⚠"
    echo "  $ICON $svc: $STATUS"
done

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Done. Services will now start on boot and restart on crash. ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║                                                              ║"
echo "║  Check status:  bash scripts/deploy_services.sh --status    ║"
echo "║  Tail logs:     bash scripts/deploy_services.sh --logs      ║"
echo "║  Restart all:   bash scripts/deploy_services.sh --restart   ║"
echo "║                                                              ║"
echo "║  Remote deploy: bash scripts/deploy_services.sh --host      ║"
echo "║                 cesarops@10.0.0.56                          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
