#!/usr/bin/env bash
set -euo pipefail

REPO="/home/alon/Documents/VS_code/Quntum-measurment-model"
PY="$REPO/.venv/bin/python"
CSV="$REPO/results/z2_scan/l48_nightly_20260324/z2_l48_nightly.csv"
SMOKE_GAMMAS=$(cat "$REPO/results/z2_scan/l48_nightly_20260324/gamma_smoke_10.txt")
FULL_GAMMAS=$(cat "$REPO/results/z2_scan/l48_nightly_20260324/gamma_full_200.txt")

echo "[$(date -Is)] START l48 orchestrator"
echo "[$(date -Is)] Smoke phase: 10 points, 10 cores, fail_on_nan"
"$PY" "$REPO/scripts/run_z2_scan.py" \
  --executor multi_cpu \
  --device cpu \
  --n-workers 10 \
  --csv "$CSV" \
  --l-values 48 \
  --gamma-values $SMOKE_GAMMAS \
  --n-trajectories-per-point 1 \
  --use-stable-integrator \
  --nan-mode fail_on_nan \
  --no-resume

echo "[$(date -Is)] Smoke phase completed successfully"
echo "[$(date -Is)] Full phase: 200 points, 16 cores, finish_full_sweep (resume)"
"$PY" "$REPO/scripts/run_z2_scan.py" \
  --executor multi_cpu \
  --device cpu \
  --n-workers 16 \
  --csv "$CSV" \
  --l-values 48 \
  --gamma-values $FULL_GAMMAS \
  --n-trajectories-per-point 1 \
  --use-stable-integrator \
  --nan-mode finish_full_sweep

echo "[$(date -Is)] FULL phase completed"
