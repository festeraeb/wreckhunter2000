#!/usr/bin/env python3
"""
CESAROPS Service Watchdog
=========================
Keeps the wrecks API and KoboldCPP alive on Linux nodes.

What it does:
  - Starts wrecks_api (uvicorn) on port 8099 if not running
  - Starts KoboldCPP on port 5001 if a model is found and it's not running
  - Health-checks both every POLL_SECS seconds
  - Restarts either service if it stops responding
  - Writes status to db/watchdog_state.json (read by the frontend)
  - Logs to logs/watchdog.log

Run as a systemd service (see scripts/cesarops-watchdog.service) or directly:
    python watchdog.py

Environment variables (or .env):
    WATCHDOG_POLL_SECS      — check interval in seconds (default 30)
    WATCHDOG_API_PORT       — wrecks API port (default 8099)
    WATCHDOG_KOBOLD_PORT    — KoboldCPP port (default 5001)
    WATCHDOG_KOBOLD_MODEL   — path to .gguf model (auto-detected if unset)
    WATCHDOG_KOBOLD_LAYERS  — GPU layers for KoboldCPP (default 28)
    WATCHDOG_VENV           — path to venv (default: auto-detect)
"""

import json
import logging
import os
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent
LOG_FILE = REPO / "logs" / "watchdog.log"
STATE_FILE = REPO / "db" / "watchdog_state.json"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(LOG_FILE), encoding="utf-8"),
    ],
)
log = logging.getLogger("watchdog")


# ── Config ────────────────────────────────────────────────────────────────────
def _load_env() -> dict:
    env = {}
    p = REPO / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip()
    return env

_ENV = _load_env()

def _cfg(key: str, default: str = "") -> str:
    return os.environ.get(key, _ENV.get(key, default))

POLL_SECS      = int(_cfg("WATCHDOG_POLL_SECS", "30"))
API_PORT       = int(_cfg("WATCHDOG_API_PORT", "8099"))
KOBOLD_PORT    = int(_cfg("WATCHDOG_KOBOLD_PORT", "5001"))
KOBOLD_LAYERS  = int(_cfg("WATCHDOG_KOBOLD_LAYERS", "28"))
KOBOLD_MODEL   = _cfg("WATCHDOG_KOBOLD_MODEL", "")  # auto-detect if empty

# ── Find Python / venv ────────────────────────────────────────────────────────
def _find_python() -> str:
    venv = _cfg("WATCHDOG_VENV", "")
    if venv:
        p = Path(venv) / "bin" / "python"
        if p.exists():
            return str(p)
    # Common venv locations on the nodes
    for candidate in [
        REPO / ".venv" / "bin" / "python",
        Path.home() / "tpu-venv" / "bin" / "python",
        Path.home() / ".venv" / "bin" / "python",
        Path("/usr/bin/python3"),
        Path("/usr/local/bin/python3"),
    ]:
        if candidate.exists():
            return str(candidate)
    return sys.executable

PYTHON = _find_python()

# ── Find KoboldCPP binary ─────────────────────────────────────────────────────
def _find_kobold_bin() -> str | None:
    home = Path.home()
    candidates = [
        home / "ai_coding" / "koboldcpp",
        home / "koboldcpp" / "koboldcpp",
        home / "koboldcpp",
        Path("/opt/koboldcpp/koboldcpp-linux-x64"),
        Path("/opt/koboldcpp/koboldcpp"),
        Path("/usr/local/bin/koboldcpp"),
        Path("/usr/bin/koboldcpp"),
        REPO / "koboldcpp",
        REPO / "koboldcpp-linux-x64",
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return str(p)
    return None

# ── Find a model file ─────────────────────────────────────────────────────────
def _find_model() -> str | None:
    if KOBOLD_MODEL and Path(KOBOLD_MODEL).exists():
        return KOBOLD_MODEL
    search_dirs = [
        Path("/mnt/garmour/models"),
        Path("/mnt/data/models"),
        Path.home() / "models",
        Path.home() / "ai_coding" / "models",
        Path("/models"),
        REPO / "models",
    ]
    for d in search_dirs:
        if d.exists():
            gguf_files = sorted(d.glob("*.gguf"), key=lambda f: f.stat().st_size, reverse=True)
            if gguf_files:
                return str(gguf_files[0])
    return None


# ── Health check ──────────────────────────────────────────────────────────────
def _is_up(port: int, path: str = "/health", timeout: int = 4) -> bool:
    try:
        urllib.request.urlopen(f"http://localhost:{port}{path}", timeout=timeout)
        return True
    except Exception:
        return False


# ── Process tracking ──────────────────────────────────────────────────────────
_procs: dict[str, subprocess.Popen] = {}


def _start_api() -> bool:
    """Start uvicorn wrecks_api on API_PORT."""
    log.info(f"Starting wrecks API on port {API_PORT}…")
    env = dict(os.environ)
    env["DB_PATH"] = str(REPO / "db" / "wrecks.db")
    env["API_BASE_URL"] = f"http://localhost:{API_PORT}"
    try:
        proc = subprocess.Popen(
            [PYTHON, "-m", "uvicorn", "wrecks_api.app:app",
             "--host", "0.0.0.0", "--port", str(API_PORT),
             "--workers", "2", "--log-level", "warning"],
            cwd=str(REPO),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        _procs["api"] = proc
        log.info(f"  API started (PID {proc.pid})")
        return True
    except Exception as e:
        log.error(f"  Failed to start API: {e}")
        return False


def _start_kobold() -> bool:
    """Start KoboldCPP on KOBOLD_PORT if a binary and model are available."""
    kobold_bin = _find_kobold_bin()
    if not kobold_bin:
        log.warning("KoboldCPP binary not found — skipping auto-start")
        return False

    model = _find_model()
    if not model:
        log.warning("No .gguf model found — skipping KoboldCPP auto-start")
        return False

    log.info(f"Starting KoboldCPP on port {KOBOLD_PORT} with model {Path(model).name}…")
    try:
        proc = subprocess.Popen(
            [kobold_bin,
             "--model", model,
             "--port", str(KOBOLD_PORT),
             "--gpulayers", str(KOBOLD_LAYERS),
             "--contextsize", "4096",
             "--threads", "8"],
            cwd=str(REPO),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        _procs["kobold"] = proc
        log.info(f"  KoboldCPP started (PID {proc.pid})")
        return True
    except Exception as e:
        log.error(f"  Failed to start KoboldCPP: {e}")
        return False


def _ensure_running(name: str, port: int, health_path: str, start_fn) -> str:
    """
    Check if service is healthy. Restart if dead.
    Returns: "ok" | "restarted" | "failed"
    """
    # Check if existing tracked process died
    proc = _procs.get(name)
    if proc and proc.poll() is not None:
        log.warning(f"{name} process (PID {proc.pid}) exited with code {proc.returncode}")
        del _procs[name]

    if _is_up(port, health_path):
        return "ok"

    log.warning(f"{name} not responding on port {port} — restarting…")
    ok = start_fn()
    if ok:
        # Give it a moment to bind
        time.sleep(3)
        if _is_up(port, health_path):
            log.info(f"  {name} is back up ✓")
            return "restarted"
        else:
            log.error(f"  {name} started but still not responding")
            return "failed"
    return "failed"


# ── State file ────────────────────────────────────────────────────────────────
def _write_state(api: str, kobold: str, restarts: dict):
    STATE_FILE.write_text(json.dumps({
        "running": True,
        "worker_id": "watchdog",
        "api_status": api,
        "kobold_status": kobold,
        "api_port": API_PORT,
        "kobold_port": KOBOLD_PORT,
        "api_pid": _procs.get("api", None) and _procs["api"].pid,
        "kobold_pid": _procs.get("kobold", None) and _procs["kobold"].pid,
        "restarts": restarts,
        "updated": datetime.now(timezone.utc).isoformat(),
        "poll_secs": POLL_SECS,
    }, indent=2))


# ── Graceful shutdown ─────────────────────────────────────────────────────────
_STOP = False

def _sig_handler(signum, frame):
    global _STOP
    log.info(f"Signal {signum} — shutting down watchdog")
    _STOP = True

signal.signal(signal.SIGTERM, _sig_handler)
signal.signal(signal.SIGINT, _sig_handler)


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info("CESAROPS Watchdog starting")
    log.info(f"  Python:      {PYTHON}")
    log.info(f"  API port:    {API_PORT}")
    log.info(f"  Kobold port: {KOBOLD_PORT}")
    log.info(f"  Poll every:  {POLL_SECS}s")
    log.info("=" * 60)

    restarts = {"api": 0, "kobold": 0}

    # Initial start
    if not _is_up(API_PORT, "/health"):
        if _start_api():
            time.sleep(4)  # wait for uvicorn to bind
    if not _is_up(KOBOLD_PORT, "/api/v1/model"):
        _start_kobold()
        time.sleep(3)

    while not _STOP:
        api_status    = _ensure_running("api",    API_PORT,    "/health",        _start_api)
        kobold_status = _ensure_running("kobold", KOBOLD_PORT, "/api/v1/model",  _start_kobold)

        if api_status    == "restarted": restarts["api"]    += 1
        if kobold_status == "restarted": restarts["kobold"] += 1

        _write_state(api_status, kobold_status, restarts)

        if api_status != "ok" or kobold_status != "ok":
            log.info(f"Status — API: {api_status}  Kobold: {kobold_status}  "
                     f"Restarts: api={restarts['api']} kobold={restarts['kobold']}")

        time.sleep(POLL_SECS)

    # Clean shutdown — don't kill child processes, let systemd handle them
    _write_state("stopped", "stopped", restarts)
    log.info("Watchdog stopped")


if __name__ == "__main__":
    main()
