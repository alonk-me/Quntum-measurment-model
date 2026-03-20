#!/usr/bin/env python3
"""
Non-blocking monitor for z2 GPU calibration scans.
Periodically checks CSV files and reports progress without hanging.
"""

import os
import time
import sys
from pathlib import Path
from datetime import datetime

WORKSPACE = Path("/home/alon/Documents/VS_code/Quntum-measurment-model")
RESULTS_DIR = WORKSPACE / "results" / "z2_scan"
POLL_INTERVAL = 10  # seconds

def get_csv_stats(csv_file):
    """Get stats from a CSV file without blocking."""
    try:
        if not csv_file.exists():
            return None, None
        
        # Quick non-blocking read
        with open(csv_file, 'r') as f:
            lines = f.readlines()
        
        headers = lines[0] if lines else None
        data_lines = len(lines) - 1  # Subtract header
        
        # Try to get last entry
        last_entry = ""
        if len(lines) > 1:
            parts = lines[-1].strip().split(',')
            # Extract L, gamma, z2_plus_one, n_trajectories
            if len(parts) >= 5:
                last_entry = f"L={parts[0]}, γ={parts[1]}, z²+1={parts[2]}, n_traj={parts[4]}"
        
        return data_lines, last_entry
    except Exception as e:
        return f"Error: {e}", None

def monitor_loop():
    """Main monitoring loop."""
    print(f"Z2 Calibration Monitor Started")
    print(f"Workspace: {WORKSPACE}")
    print(f"Poll interval: {POLL_INTERVAL}s")
    print(f"Watching: {RESULTS_DIR}")
    print("-" * 60)
    
    csv_files = {
        "L=3": RESULTS_DIR / "z2_gpu_L3_calib_16traj.csv",
        "L=9": RESULTS_DIR / "z2_gpu_L9_calib_16traj.csv",
        "L=17": RESULTS_DIR / "z2_gpu_L17_calib_16traj.csv",
    }
    
    last_counts = {}
    
    try:
        iteration = 0
        while True:
            iteration += 1
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            print(f"\n[{timestamp}] Poll #{iteration}")
            
            any_active = False
            for label, csv_file in csv_files.items():
                count, last_entry = get_csv_stats(csv_file)
                
                if isinstance(count, int):
                    # File exists and has data
                    any_active = True
                    is_new = count != last_counts.get(label)
                    status = "NEW DATA" if is_new else "unchanged"
                    
                    print(f"  {label}: {count} data points [{status}]")
                    if last_entry:
                        print(f"    Last: {last_entry}")
                    
                    last_counts[label] = count
                elif count is None:
                    # File doesn't exist yet
                    print(f"  {label}: waiting...")
                else:
                    # Error
                    print(f"  {label}: {count}")
            
            if not any_active:
                print("  No active scans yet...")
            
            # Sleep before next poll (non-blocking)
            time.sleep(POLL_INTERVAL)
    
    except KeyboardInterrupt:
        print("\nMonitor stopped by user")
        return 0
    except Exception as e:
        print(f"\nMonitor error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(monitor_loop())
