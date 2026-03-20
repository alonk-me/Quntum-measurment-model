#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="$PYTHON_BIN"
elif [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  PYTHON_BIN="python"
fi
STAMP="$(date +%Y%m%d_%H%M%S)"

OUT_DIR="results/test_scan"
LOG_DIR="logs"
mkdir -p "$OUT_DIR" "$LOG_DIR"

CSV_PATH="${CSV_PATH:-$OUT_DIR/long_gpu_benchmark_${STAMP}.csv}"
LOG_PATH="${LOG_PATH:-$LOG_DIR/long_gpu_benchmark_${STAMP}.log}"
PID_PATH="${PID_PATH:-$LOG_DIR/long_gpu_benchmark.pid}"
STATE_PATH="${STATE_PATH:-$LOG_DIR/long_gpu_benchmark.state}"

compute_expected_rows() {
  if ! "$PYTHON_BIN" scripts/run_long_gpu_benchmark_campaign.py --print-expected-rows "$@" 2>/dev/null; then
    echo ""
  fi
}

ln -sfn "$(basename "$CSV_PATH")" "$OUT_DIR/long_gpu_benchmark.latest.csv"
ln -sfn "$(basename "$LOG_PATH")" "$LOG_DIR/long_gpu_benchmark.latest.log"

cat >"$STATE_PATH" <<EOF
CSV_PATH=$CSV_PATH
LOG_PATH=$LOG_PATH
PID_PATH=$PID_PATH
STAMP=$STAMP
EOF

CMD=(
  "$PYTHON_BIN" scripts/run_long_gpu_benchmark_campaign.py
  --csv "$CSV_PATH"
)

if [[ "${1:-}" == "--background" ]]; then
  shift
  EXPECTED_ROWS="$(compute_expected_rows "$@")"
  CMD+=("$@")

  cat >"$STATE_PATH" <<EOF
CSV_PATH=$CSV_PATH
LOG_PATH=$LOG_PATH
PID_PATH=$PID_PATH
STAMP=$STAMP
EXPECTED_ROWS=${EXPECTED_ROWS}
EOF

  echo "Starting long GPU benchmark campaign in background"
  echo "  csv: $CSV_PATH"
  echo "  log: $LOG_PATH"
  if [[ -n "${EXPECTED_ROWS}" ]]; then
    echo "  expected_rows: $EXPECTED_ROWS"
  fi
  nohup "${CMD[@]}" >"$LOG_PATH" 2>&1 &
  echo $! >"$PID_PATH"
  echo "  pid: $(cat "$PID_PATH")"
  echo "  state: $STATE_PATH"
  echo "  latest csv symlink: $OUT_DIR/long_gpu_benchmark.latest.csv"
  echo "  latest log symlink: $LOG_DIR/long_gpu_benchmark.latest.log"
  echo "Use: tail -f $LOG_PATH"
  exit 0
fi

CMD+=("$@")
EXPECTED_ROWS="$(compute_expected_rows "$@")"

cat >"$STATE_PATH" <<EOF
CSV_PATH=$CSV_PATH
LOG_PATH=$LOG_PATH
PID_PATH=$PID_PATH
STAMP=$STAMP
EXPECTED_ROWS=${EXPECTED_ROWS}
EOF

echo "Starting long GPU benchmark campaign in foreground"
echo "  csv: $CSV_PATH"
echo "  log: $LOG_PATH"
if [[ -n "${EXPECTED_ROWS}" ]]; then
  echo "  expected_rows: $EXPECTED_ROWS"
fi
"${CMD[@]}" | tee "$LOG_PATH"
