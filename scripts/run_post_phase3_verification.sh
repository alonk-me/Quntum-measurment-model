#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
STAMP="$(date +%Y%m%d_%H%M%S)"

OUT_DIR="results/test_scan"
GPU_CSV="$OUT_DIR/post_phase3_gpu_${STAMP}.csv"
CPU_CSV="$OUT_DIR/post_phase3_cpu_${STAMP}.csv"
BENCH_CSV="$OUT_DIR/post_phase3_benchmark_${STAMP}.csv"

mkdir -p "$OUT_DIR"

echo "[1/5] Running focused tests"
"$PYTHON_BIN" -m pytest tests/test_gpu_backend.py tests/test_sweep_executor.py -q

echo "[2/5] Running benchmark"
"$PYTHON_BIN" scripts/benchmark_gpu_speedup.py \
  --simulator l_qubit \
  --L 32 \
  --n-steps 200 \
  --n-trajectories 128 \
  --batch-sizes 1 2 4 8 16 32 \
  --repeats 3 \
  --csv "$BENCH_CSV"

echo "[3/5] Running small GPU sweep"
"$PYTHON_BIN" scripts/run_ninf_scan.py \
  --device gpu \
  --parallel-backend sequential \
  --l-values 9 17 \
  --gamma-values 0.4 1.0 4.0 \
  --skip-plots \
  --output-csv "$GPU_CSV"

echo "[4/5] Running matching CPU baseline sweep"
"$PYTHON_BIN" scripts/run_ninf_scan.py \
  --device cpu \
  --parallel-backend sequential \
  --l-values 9 17 \
  --gamma-values 0.4 1.0 4.0 \
  --skip-plots \
  --output-csv "$CPU_CSV"

echo "[5/5] Comparing GPU vs CPU results"
export GPU_CSV CPU_CSV
"$PYTHON_BIN" - <<'PY'
import os
import sys
import numpy as np
import pandas as pd

gpu_csv = os.environ["GPU_CSV"]
cpu_csv = os.environ["CPU_CSV"]

gpu = pd.read_csv(gpu_csv)
cpu = pd.read_csv(cpu_csv)
merged = gpu.merge(cpu, on=["L", "gamma"], suffixes=("_gpu", "_cpu"))

if merged.empty:
    print("No overlapping rows between GPU and CPU outputs.")
    sys.exit(2)

num = np.abs(merged["n_inf_sim_gpu"] - merged["n_inf_sim_cpu"])
den = np.maximum(np.abs(merged["n_inf_sim_cpu"]), 1e-12)
rel = num / den

max_rel = float(np.max(rel))
mean_rel = float(np.mean(rel))
ok = bool(np.all(rel <= 0.01))

print(f"rows={len(merged)}")
print(f"max_rel={max_rel:.6e}")
print(f"mean_rel={mean_rel:.6e}")
print(f"threshold_pass={ok}")

if not ok:
    worst = int(np.argmax(rel.values))
    row = merged.iloc[worst]
    print(
        "worst_case:",
        {
            "L": int(row["L"]),
            "gamma": float(row["gamma"]),
            "n_inf_gpu": float(row["n_inf_sim_gpu"]),
            "n_inf_cpu": float(row["n_inf_sim_cpu"]),
            "rel": float(rel.iloc[worst]),
        },
    )
    sys.exit(1)
PY

echo "Verification complete"
echo "  benchmark: $BENCH_CSV"
echo "  gpu sweep: $GPU_CSV"
echo "  cpu sweep: $CPU_CSV"
