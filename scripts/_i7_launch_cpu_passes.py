#!/usr/bin/env python3
"""
Launch i7 CPU specialized scan (Pass 2/3/4 only — no GPU required).

i7 node: 10.0.0.56 / cesarops / cesarops
  - No Nvidia GPU (Intel HD 4000; P1000 moved to Xeon)
  - Edge TPU (Coral) on :5001 — tpu_server already running
  - TIFs: ~/downloads/{michigan,superior,huron,erie,...}

Launches i7_cpu_passes.py with CESAROPS_DATA_DIR=/home/cesarops/downloads
"""

import paramiko
import time

HOST = "10.0.0.56"
USER = "cesarops"
PASS = "cesarops"
REPO = "/home/cesarops/wreckhunter2000-1"
DATA = "/home/cesarops/downloads"
LOG  = "/home/cesarops/i7_cpu_passes.log"


def run(c, cmd, timeout=60):
    _, o, e = c.exec_command(cmd, timeout=timeout)
    try:
        out = o.read().decode(errors='replace').strip()
        err = e.read().decode(errors='replace').strip()
    except Exception:
        out, err = '', ''
    return out or err


def main():
    print(f"[*] Connecting to i7 ({HOST})...")
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=PASS, timeout=10)
    print("[+] Connected")

    # Confirm no GPU (expected)
    gpu = run(c, "lspci | grep -i nvidia || echo 'no nvidia'")
    print(f"[gpu] {gpu}")

    # Pull latest (picks up i7_cpu_passes.py)
    print("[git] Pulling latest...")
    print(run(c, f"cd {REPO} && git pull origin wreckhuntertools 2>&1 | tail -3", 60))

    # Install scipy if missing (needed for NauticUVs Pass 4)
    print("[dep] Checking scipy...")
    result = run(c, f"{REPO}/.venv/bin/python -c 'import scipy' 2>&1 || echo MISSING")
    if 'MISSING' in result or 'No module' in result:
        print("[dep] Installing scipy...")
        print(run(c, f"{REPO}/.venv/bin/pip install scipy -q 2>&1 | tail -3", 120))
    else:
        print("[dep] scipy OK")

    # Kill any old i7_cpu_passes run
    run(c, "pkill -f i7_cpu_passes 2>/dev/null || true", 5)
    time.sleep(1)

    # Check TIF count
    tif_count = run(c, f"find {DATA} -name '*.tif' | wc -l")
    print(f"[tifs] {tif_count} TIFs in {DATA}")

    # Launch
    launch_cmd = (
        f"cd {REPO} && "
        f"CESAROPS_DATA_DIR={DATA} "
        f"nohup .venv/bin/python i7_cpu_passes.py > {LOG} 2>&1 &"
    )
    run(c, launch_cmd, timeout=5)
    time.sleep(4)

    ps = run(c, "pgrep -la python | grep i7_cpu_passes || echo not_running")
    print(f"[ps] {ps}")

    log_tail = run(c, f"head -25 {LOG} 2>/dev/null || echo no_log")
    print(f"\n[log]\n{log_tail}")

    # Confirm TPU server still up
    tpu = run(c, "pgrep -la python | grep tpu_server || echo tpu_not_running")
    print(f"\n[tpu] {tpu}")

    c.close()
    print("\n[+] Done — i7 specialized scan launched")
    print(f"     Monitor: ssh {USER}@{HOST} 'tail -f {LOG}'")


if __name__ == "__main__":
    main()
