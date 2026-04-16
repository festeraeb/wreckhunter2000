#!/usr/bin/env python3
"""
Deploy code to remote nodes and run a comprehensive multi-sensor scan.
Uses paramiko for SSH with key-based auth (falls back to .env passwords).
"""

import paramiko
import time
import os
import json
from pathlib import Path


def _load_env(path: Path) -> dict:
    env = {}
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, _, v = line.partition('=')
                env[k.strip()] = v.strip()
    return env

_dotenv = _load_env(Path(__file__).parent / ".env")

# Node configurations — credentials from environment / .env, never hardcoded
NODES = {
    'pi': {
        'host': os.environ.get("PI_HOST", _dotenv.get("PI_HOST", "10.0.0.226")),
        'user': os.environ.get("PI_USER", _dotenv.get("PI_USER", "pi")),
        'password': os.environ.get("PI_PASS", _dotenv.get("PI_PASS", "")),
        'work_dir': '/home/pi/cesarops',
    },
    'xeon': {
        'host': os.environ.get("XENON_HOST", _dotenv.get("XENON_HOST", "10.0.0.129")),  # cesarops2 at .129
        'user': os.environ.get("XENON_USER", _dotenv.get("XENON_USER", "cesarops1")),
        'password': os.environ.get("XENON_PASS", _dotenv.get("XENON_PASS", "cesarops1")),
        'work_dir': '/home/cesarops1/wreckhunter2000-1',
    }
}

# Area definition for the scan
SCAN_AREA = {
    'name': 'Northern Great Lakes Comprehensive Scan',
    'description': 'Northern Lake Huron + Michigan + Straits + 45th parallel to UP + North Channel + Georgian Bay + Green Bay',
    'bbox': {
        'lat_min': 44.5,
        'lon_min': -92.0,  # Western Green Bay
        'lat_max': 47.0,   # UP boundary
        'lon_max': -80.0,  # Eastern Georgian Bay/North Channel
    },
    'dates': {
        'start': '2024-06-01',  # Post ice-melt
        'end': '2025-09-30',    # Pre freeze-up
    },
    'sensors': ['thermal', 'optical', 'sar', 'swot'],
}

def ssh_connect(node_config):
    """Connect to a node with key-based auth, falling back to password from .env."""
    client = paramiko.SSHClient()
    # Load known hosts to prevent MITM — reject unknown hosts
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    try:
        client.load_system_host_keys()
    except FileNotFoundError:
        pass

    try:
        # Try key-based auth first (ed25519 keys deployed via SSH)
        client.connect(
            node_config['host'],
            username=node_config['user'],
            timeout=10,
            allow_agent=True,
            look_for_keys=True,
        )
        return client
    except Exception as e:
        if not node_config.get('password'):
            print(f"  Key auth failed and no password configured: {e}")
            return None
        print(f"  Key auth failed ({e}), trying password from .env...")
        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.RejectPolicy())
            try:
                client.load_system_host_keys()
            except FileNotFoundError:
                pass
            client.connect(
                node_config['host'],
                username=node_config['user'],
                password=node_config['password'],
                timeout=10,
                allow_agent=False,
                look_for_keys=False,
            )
            return client
        except Exception as e2:
            print(f"  Password auth also failed: {e2}")
            return None

def run_command(client, cmd, timeout=300):
    """Run a command on remote node and return output."""
    try:
        stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
        exit_code = stdout.channel.recv_exit_status()
        out = stdout.read().decode('utf-8', errors='replace')
        err = stderr.read().decode('utf-8', errors='replace')
        return {
            'exit_code': exit_code,
            'stdout': out,
            'stderr': err,
        }
    except Exception as e:
        return {
            'exit_code': -1,
            'stdout': '',
            'stderr': str(e),
        }

def deploy_code():
    """Trigger pull-based update on each node via node_update.sh."""
    print("\n" + "="*60)
    print("TRIGGERING PULL-BASED UPDATE ON REMOTE NODES")
    print("="*60)

    # node_update.sh lives inside the repo; nodes run it from their local clone.
    # If the script isn't on the node yet, SCP it once as a bootstrap step.
    core_dir = Path(__file__).parent
    update_script_local = core_dir / 'scripts' / 'node_update.sh'

    for node_name, node_cfg in NODES.items():
        print(f"\n--- Updating {node_name} ({node_cfg['host']}) ---")
        client = ssh_connect(node_cfg)
        if not client:
            print(f"  ✗ Could not connect to {node_name} — skipping")
            continue

        print(f"  ✓ Connected")

        # ── Bootstrap: push node_update.sh if the repo clone doesn't exist yet ──
        # Check whether the repo is already present on the node
        check = run_command(client,
            "find $HOME -maxdepth 6 -name 'node_update.sh' -path '*/scripts/*' 2>/dev/null | head -1")
        script_path = check['stdout'].strip()

        if not script_path:
            print("  Repo not found on node — bootstrapping node_update.sh...")
            sftp = client.open_sftp()
            try:
                sftp.put(str(update_script_local), '/tmp/node_update_bootstrap.sh')
                print("  ✓ Bootstrap script uploaded")
            except Exception as e:
                print(f"  ✗ Could not upload bootstrap script: {e}")
                client.close()
                continue
            finally:
                sftp.close()
            script_path = '/tmp/node_update_bootstrap.sh'

        # ── Run node_update.sh ───────────────────────────────────────────────
        gpu_flag = '--gpu' if node_name == 'xenon' else ''
        update_cmd = f"bash {script_path} {gpu_flag}".strip()
        print(f"  Running: {update_cmd}")
        result = run_command(client, update_cmd, timeout=300)

        output = (result['stdout'] + result['stderr']).strip()
        for line in output.splitlines():
            print(f"    {line}")

        if result['exit_code'] == 0:
            print(f"  ✓ {node_name} updated successfully")
        else:
            print(f"  ✗ {node_name} update failed (exit {result['exit_code']})")

        client.close()

def run_comprehensive_scan():
    """Run the full multi-sensor scan on the defined area."""
    print("\n" + "="*60)
    print("RUNNING COMPREHENSIVE MULTI-SENSOR SCAN")
    print("="*60)
    print(f"Area: {SCAN_AREA['name']}")
    print(f"BBOX: {SCAN_AREA['bbox']}")
    print(f"Dates: {SCAN_AREA['dates']}")
    print(f"Sensors: {', '.join(SCAN_AREA['sensors'])}")
    
    # Connect to Xenon for processing
    xenon_cfg = NODES['xenon']
    client = ssh_connect(xenon_cfg)
    if not client:
        print("✗ Could not connect to Xenon for scan execution")
        return
    
    print(f"\n✓ Connected to Xenon ({xenon_cfg['host']})")
    
    # Set up environment variables for the scan
    bbox = SCAN_AREA['bbox']
    dates = SCAN_AREA['dates']
    
    # Run the comprehensive scan
    scan_cmd = f"""
    source ~/cesarops/venv/bin/activate
    cd {xenon_cfg['work_dir']}
    
    # Set data directory
    export CESAROPS_DATA_DIR=/home/cesarops/cesarops/Sync
    
    echo "Starting comprehensive scan..."
    echo "Area: {SCAN_AREA['name']}"
    echo "BBOX: {bbox['lat_min']},{bbox['lon_min']},{bbox['lat_max']},{bbox['lon_max']}"
    echo "Date range: {dates['start']} to {dates['end']}"
    echo "Sensors: {', '.join(SCAN_AREA['sensors'])}"
    echo ""
    
    # Run AI Director with all sensors
    python ai_director.py \\
        --bbox {bbox['lat_min']},{bbox['lon_min']},{bbox['lat_max']},{bbox['lon_max']} \\
        --tools thermal,optical,sar,swot \\
        --sensitivity 1.0 \\
        --execute \\
        --no-llm \\
        --output outputs/probes/comprehensive_scan_{dates['start'].replace('-', '')}.json 2>&1
    """
    
    print(f"\nExecuting scan command...")
    result = run_command(client, scan_cmd, timeout=7200)  # 2 hour timeout
    
    print(f"\n{'='*60}")
    print("SCAN RESULTS")
    print(f"{'='*60}")
    
    if result['stdout']:
        print(result['stdout'])
    
    if result['stderr']:
        print(f"\nSTDERR:\n{result['stderr']}")
    
    print(f"\nExit code: {result['exit_code']}")
    
    # Save results locally
    timestamp = dates['start'].replace('-', '')
    output_file = Path(__file__).parent / 'outputs' / 'probes' / f'comprehensive_scan_{timestamp}.json'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    results_data = {
        'timestamp': time.time(),
        'area': SCAN_AREA['name'],
        'bbox': SCAN_AREA['bbox'],
        'dates': SCAN_AREA['dates'],
        'sensors': SCAN_AREA['sensors'],
        'remote_execution': {
            'host': xenon_cfg['host'],
            'exit_code': result['exit_code'],
            'stdout': result['stdout'],
            'stderr': result['stderr'],
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    client.close()

def main():
    print("CESAROPS Multi-Node Deployment & Scan System")
    print("="*60)
    
    # Step 1: Deploy code
    deploy_code()
    
    # Step 2: Run scan
    run_comprehensive_scan()
    
    print("\n" + "="*60)
    print("ALL TASKS COMPLETE")
    print("="*60)

if __name__ == '__main__':
    main()
