# Susceptibility Analysis System - Deployment Guide

**Version**: 1.0  
**Date**: January 10, 2026  
**Target Environment**: Linux/macOS with Python 3.8+

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Running Production Scans](#running-production-scans)
5. [Monitoring Progress](#monitoring-progress)
6. [Post-Analysis](#post-analysis)
7. [Troubleshooting](#troubleshooting)
8. [Remote Deployment](#remote-deployment)

---

## Prerequisites

### System Requirements

**Hardware**:
- CPU: Multi-core recommended (4+ cores)
- RAM: 8 GB minimum, 16 GB recommended
- Storage: 10 GB free space for cache and results
- OS: Linux or macOS (Windows with WSL)

**Software**:
- Python 3.8 or higher
- Git (for cloning repository)
- Optional: tmux or screen for long-running jobs

### Python Environment

Recommended: Use a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/alonk-me/Quntum-measurment-model.git
cd Quntum-measurment-model
```

### 2. Install Dependencies

```bash
pip install -e .
```

This installs the package in editable mode with all dependencies:
- numpy
- scipy
- pandas
- matplotlib
- h5py

### 3. Verify Installation

```bash
python -c "from quantum_measurement.analysis import compute_n_inf; print('✓ Installation successful')"
```

---

## Quick Start

### Test Run (Small Grid)

Run a small test to verify everything works:

```bash
python scripts/run_susceptibility_scan.py \
    --L 9 17 \
    --gamma-min 3.0 \
    --gamma-max 5.0 \
    --n-gamma 10 \
    --output-dir results/test_scan \
    --verbose
```

**Expected output**:
- Creates `results/test_scan/` directory
- Generates `n_inf_cache.h5` (HDF5 cache)
- Produces `chi_n_results_TIMESTAMP.csv`
- Prints progress for 20 total points (10 γ × 2 L)

**Runtime**: ~5-10 minutes

### Analyze Results

```bash
# Extract γc
python scripts/estimate_gamma_c.py results/test_scan/chi_n_results_*.csv

# Generate plots
python scripts/plot_susceptibility.py results/test_scan/chi_n_results_*.csv
```

**Output**:
- `chi_n_vs_gamma.png`: Main susceptibility curves
- `chi_n_critical_zoom.png`: Zoomed critical region
- `finite_size_scaling.png`: Extrapolation to L→∞
- `error_analysis.png`: Diagnostic panels

---

## Running Production Scans

### Standard Production Configuration

For publishable results, use:

```bash
python scripts/run_susceptibility_scan.py \
    --L 9 17 33 65 129 \
    --gamma-min 2.0 \
    --gamma-max 6.0 \
    --n-gamma 100 \
    --gamma-spacing linear \
    --output-dir results/production_scan \
    --tolerance 1e-4 \
    --dt 0.001 \
    --verbose
```

**Parameters**:
- `--L`: System sizes (odd values for proper momentum grid)
- `--gamma-min/max`: Range covering critical region (γc ≈ 4.0)
- `--n-gamma`: Number of points (100-200 recommended)
- `--tolerance`: Convergence threshold (1e-4 is good default)
- `--dt`: Time step (0.001 is safe, 0.01 for faster/less accurate)

**Computational Cost**:
- Total points: 100 γ × 5 L = 500
- Estimated runtime: 6-8 hours (single core)
- Storage: ~500 MB (cache) + ~5 MB (CSV)

### Dense Critical Region Scan

For high-resolution near γc:

```bash
python scripts/run_susceptibility_scan.py \
    --L 17 33 65 129 \
    --gamma-min 3.5 \
    --gamma-max 4.5 \
    --n-gamma 200 \
    --output-dir results/critical_scan \
    --verbose
```

**Use case**: Precise γc determination after initial scan

**Runtime**: ~4-5 hours

---

## Monitoring Progress

### Real-Time Monitoring

#### Option 1: Verbose Output

Run with `--verbose` flag to see progress:

```
[12:34:56] Progress: 10/500 (2.0%) | L=9, γ=2.1234
[12:35:23] Progress: 11/500 (2.2%) | L=9, γ=2.2345
...
```

#### Option 2: Log to File

```bash
python scripts/run_susceptibility_scan.py \
    --L 9 17 33 --n-gamma 50 --verbose \
    > logs/scan_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Monitor in real-time
tail -f logs/scan_*.log
```

#### Option 3: Check CSV File

The CSV file is updated incrementally:

```bash
# Count completed points
wc -l results/production_scan/chi_n_results_*.csv

# View latest results
tail results/production_scan/chi_n_results_*.csv
```

### Checking Cache

```bash
# List cache contents
python -c "
from quantum_measurement.utilities.cache import load_cache
cache = load_cache('results/production_scan/n_inf_cache.h5')
print(f'Cache has {len(cache.get_all_keys())} entries')
print(f'L values: {cache.get_L_values()}')
cache.close()
"
```

---

## Post-Analysis

### 1. Extract Critical Point

```bash
python scripts/estimate_gamma_c.py \
    results/production_scan/chi_n_results_*.csv \
    --L-min 17 \
    --model linear \
    --output results/production_scan/gamma_c.json \
    --plot
```

**Outputs**:
- `gamma_c.json`: JSON with γc estimate and fit parameters
- `gamma_c.png`: Diagnostic plots (if --plot used)

**Terminal output**:
```
γc = 4.0234 ± 0.0567
gc = 1.0059 ± 0.0142

Fit quality (R²): 0.9876
Model: linear
```

### 2. Generate Publication Figures

```bash
python scripts/plot_susceptibility.py \
    results/production_scan/chi_n_results_*.csv \
    --output-dir results/production_scan/figures \
    --dpi 300 \
    --format pdf
```

**Outputs** (in `results/production_scan/figures/`):
- `chi_n_vs_gamma.pdf`
- `chi_n_critical_zoom.pdf`
- `finite_size_scaling.pdf`
- `error_analysis.pdf`

### 3. Data Inspection

```python
import pandas as pd

# Load results
df = pd.read_csv('results/production_scan/chi_n_results_TIMESTAMP.csv')

# Summary statistics
print(df.describe())

# Check convergence
print(f"Convergence rate: {df['converged_all'].mean()*100:.1f}%")

# Find peaks
for L in df['L'].unique():
    df_L = df[df['L'] == L]
    peak_idx = df_L['chi_n'].abs().idxmax()
    gamma_peak = df_L.loc[peak_idx, 'gamma']
    print(f"L={L}: γ_peak ≈ {gamma_peak:.4f}")
```

---

## Troubleshooting

### Common Issues

#### 1. Memory Errors for Large L

**Symptom**: Process killed or "MemoryError"

**Solution**: Reduce maximum L or increase swap space

```bash
# Check available memory
free -h

# If L=256 fails, the code automatically falls back to L=129
# Or manually limit:
python scripts/run_susceptibility_scan.py --L 9 17 33 65 129  # Skip 256
```

#### 2. Slow Convergence

**Symptom**: Many points with `converged_all=False`

**Solutions**:
- Increase tolerance: `--tolerance 2e-4`
- Decrease dt: `--dt 0.0005` (more accurate but slower)
- Check specific (γ, L) pairs causing issues

```python
import pandas as pd
df = pd.read_csv('results/scan/chi_n_results.csv')
failed = df[df['converged_all'] == False]
print(failed[['L', 'gamma', 'converged_all']])
```

#### 3. Cache Corruption

**Symptom**: HDF5 errors when loading cache

**Solution**: Delete and rebuild cache

```bash
rm results/production_scan/n_inf_cache.h5
# Re-run scan (will recompute from scratch)
```

#### 4. Import Errors

**Symptom**: `ModuleNotFoundError: No module named 'quantum_measurement'`

**Solutions**:
```bash
# Ensure package is installed
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

---

## Remote Deployment

### Using tmux (Recommended)

For long-running scans on remote servers:

```bash
# Start tmux session
tmux new -s susceptibility_scan

# Run scan
python scripts/run_susceptibility_scan.py \
    --L 9 17 33 65 129 \
    --gamma-min 2.0 --gamma-max 6.0 --n-gamma 100 \
    --output-dir results/production_scan \
    --verbose

# Detach from session: Ctrl+B, then D
# Reattach later: tmux attach -t susceptibility_scan
# Kill session: tmux kill-session -t susceptibility_scan
```

### Using nohup

Alternative to tmux:

```bash
nohup python scripts/run_susceptibility_scan.py \
    --L 9 17 33 65 129 \
    --gamma-min 2.0 --gamma-max 6.0 --n-gamma 100 \
    --output-dir results/production_scan \
    --verbose \
    > logs/production_scan.log 2>&1 &

# Save process ID
echo $! > logs/scan.pid

# Check progress
tail -f logs/production_scan.log

# Kill if needed
kill $(cat logs/scan.pid)
```

### Automated Workflow Script

Create `run_production.sh`:

```bash
#!/bin/bash
# Production Susceptibility Scan Workflow

set -e  # Exit on error

# Configuration
OUTPUT_DIR="results/production_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs"

mkdir -p $OUTPUT_DIR $LOG_DIR

echo "Starting production susceptibility scan..."
echo "Output: $OUTPUT_DIR"

# Step 1: Run scan
python scripts/run_susceptibility_scan.py \
    --L 9 17 33 65 129 \
    --gamma-min 2.0 --gamma-max 6.0 --n-gamma 100 \
    --output-dir $OUTPUT_DIR \
    --verbose \
    | tee $LOG_DIR/scan_$(date +%Y%m%d_%H%M%S).log

# Step 2: Estimate γc
echo "Estimating critical point..."
python scripts/estimate_gamma_c.py \
    $OUTPUT_DIR/chi_n_results_*.csv \
    --L-min 17 --model linear --plot

# Step 3: Generate plots
echo "Generating publication figures..."
python scripts/plot_susceptibility.py \
    $OUTPUT_DIR/chi_n_results_*.csv \
    --output-dir $OUTPUT_DIR/figures \
    --dpi 300

echo "✓ Production scan complete!"
echo "Results: $OUTPUT_DIR"
```

Run with:

```bash
chmod +x run_production.sh
./run_production.sh
```

---

## Best Practices

### 1. Incremental Approach

Start small, then scale up:

```bash
# Day 1: Test (30 min)
python scripts/run_susceptibility_scan.py --L 9 17 --n-gamma 20

# Day 2: Medium (2 hours)
python scripts/run_susceptibility_scan.py --L 9 17 33 --n-gamma 50

# Day 3: Production (8 hours)
python scripts/run_susceptibility_scan.py --L 9 17 33 65 129 --n-gamma 100
```

### 2. Cache Management

- **Keep cache files**: Enables rapid reanalysis
- **Backup periodically**: Copy `.h5` files to safe location
- **Version control metadata**: Track `metadata.json` with git

### 3. Result Organization

Recommended directory structure:

```
results/
├── test_scan_20260110/          # Test runs
│   ├── n_inf_cache.h5
│   └── chi_n_results_*.csv
├── production_scan_20260110/    # Main production
│   ├── n_inf_cache.h5
│   ├── chi_n_results_*.csv
│   ├── gamma_c.json
│   ├── metadata.json
│   └── figures/
│       ├── chi_n_vs_gamma.pdf
│       └── ...
└── critical_scan_20260111/      # Focused scans
    └── ...
```

### 4. Documentation

Keep a lab notebook documenting:
- Scan parameters used
- Runtime and convergence statistics
- γc estimates from each run
- Any issues encountered

---

## Performance Tuning

### Faster Scans (Lower Accuracy)

```bash
# Relaxed convergence
--tolerance 2e-4

# Larger time step
--dt 0.002

# Coarser grid
--n-gamma 50
```

### Higher Accuracy (Slower)

```bash
# Tighter convergence
--tolerance 5e-5

# Smaller time step
--dt 0.0005

# Denser grid
--n-gamma 200
```

---

## Next Steps

After successful deployment:

1. **Validate**: Compare small-L results with analytical predictions
2. **Optimize**: Tune convergence parameters for your system
3. **Parallelize**: Implement multi-core processing (future work)
4. **Publish**: Use generated figures in papers/presentations

---

## Support & Contact

For issues or questions:
1. Check troubleshooting section above
2. Review module docstrings: `python -c "import quantum_measurement.analysis.steady_state; help(steady_state.compute_n_inf)"`
3. Consult SUSCEPTIBILITY_REPORT.md for implementation details

---

**Document version**: 1.0  
**Last updated**: January 10, 2026  
**Status**: Production Ready
