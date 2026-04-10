import os
import sys
import time
import json
from pathlib import Path

# Setup paths
ROOT = Path(r"C:\Users\thomf\programming\Bagrecovery")
sys.path.insert(0, str(ROOT))

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CompareScanners")

from bag_processor.rust_bag_scanner_runner import run_advanced_scan as run_rust
from bag_processor.advanced_bag_scanner_runner import run_advanced_scan as run_python

def main():
    bag_files = [
        r"C:\Users\thomf\Downloads\H13255_MB_50cm_LWD_1of6.bag",
        r"C:\Users\thomf\Downloads\H13255_MB_50cm_LWD_2of6.bag",
        r"C:\Users\thomf\Downloads\H13255_MB_50cm_LWD_3of6.bag",
        r"C:\Users\thomf\Downloads\H13255_MB_1m_LWD_4of6.bag",
        r"C:\Users\thomf\Downloads\H13255_MB_1m_LWD_5of6.bag",
        r"C:\Users\thomf\Downloads\H13255_MB_4m_LWD_6of6.bag"
    ]

    import os
    bag_files = [f for f in bag_files if os.path.exists(f)]
    logger.info(f"Found {len(bag_files)} files to scan.")

    config = {
        'min_wreck_size_sq_ft': 30.0,
        'max_wreck_size_sq_ft': 500000.0, # massive upper limit
        'min_confidence': 0.3,
        'anomaly_threshold': 2.5,
        'skip_pattern': [5],
        'output_kml': False,
        'output_kmz': False
    }

    out_dir = str(ROOT / "bag_processor" / "comparison_out")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n--- Running Rust Scanner ({len(bag_files)} files) ---")
    t0 = time.time()
    res_rust = run_rust(bag_files, out_dir, config)
    t_rust = time.time() - t0
    print(f"Rust finished in {t_rust:.2f} seconds.")
    
    print(f"\n--- Running Old Python Scanner ({len(bag_files)} files) ---")
    print("Warning: This might take significantly longer!")
    t0 = time.time()
    res_python = run_python(bag_files, out_dir, config)
    t_python = time.time() - t0
    print(f"Python finished in {t_python:.2f} seconds.")

    rust_cands = res_rust.get("total_candidates", 0)
    py_cands = res_python.get("total_candidates", 0)

    print("\n" + "="*50)
    print("RESULTS COMPARISON")
    print("="*50)
    print(f"RUST Scanner:   {rust_cands} candidates found in {res_rust.get('processing_time_ms', 0)/1000:.2f} seconds.")
    print(f"PYTHON Scanner: {py_cands} candidates found in {res_python.get('processing_time_ms', 0)/1000:.2f} seconds.")
    
    with open(os.path.join(out_dir, "rust_results.json"), "w") as f:
        json.dump(res_rust, f, indent=2, default=str)
    
    with open(os.path.join(out_dir, "python_results.json"), "w") as f:
        json.dump(res_python, f, indent=2, default=str)

if __name__ == "__main__":
    main()
