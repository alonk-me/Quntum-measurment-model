#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="logs"
OUT_DIR="results/test_scan"
STATE_PATH="$LOG_DIR/long_gpu_benchmark.state"
TAIL_PID_PATH="$LOG_DIR/long_gpu_benchmark_tail.pid"
PLOT_PID_PATH="$LOG_DIR/long_gpu_benchmark_plot.pid"
PLOT_OUT_PATH="$LOG_DIR/long_gpu_benchmark_plot.out"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="$PYTHON_BIN"
elif [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  PYTHON_BIN="python"
fi

usage() {
  cat <<EOF
Usage: ./scripts/monitor_long_gpu_benchmark.sh <command> [args]

Commands:
  start [campaign args...]      Start campaign in background
  status                         Show run and monitor process status
  logs                           Print last 80 log lines
  log-on                         Start background tail -f monitor
  log-off                        Stop background tail monitor
  view                           Show CSV progress snapshot
  plot [--interval SEC]          Start live PNG progress plot refresher
  plot-stop                      Stop live PNG progress plot refresher
  stop                           Stop running campaign process

Examples:
  ./scripts/monitor_long_gpu_benchmark.sh start --L-values 64 128 --n-steps-values 10000 30000
  ./scripts/monitor_long_gpu_benchmark.sh status
  ./scripts/monitor_long_gpu_benchmark.sh logs
  ./scripts/monitor_long_gpu_benchmark.sh log-on
  ./scripts/monitor_long_gpu_benchmark.sh log-off
  ./scripts/monitor_long_gpu_benchmark.sh view
  ./scripts/monitor_long_gpu_benchmark.sh plot --interval 30
  ./scripts/monitor_long_gpu_benchmark.sh plot-stop
  ./scripts/monitor_long_gpu_benchmark.sh stop
EOF
}

load_state() {
  if [[ ! -f "$STATE_PATH" ]]; then
    echo "No state file found at $STATE_PATH. Start a run first."
    exit 1
  fi
  # shellcheck disable=SC1090
  source "$STATE_PATH"
}

is_pid_alive() {
  local pid="$1"
  [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null
}

cmd_start() {
  ./scripts/start_long_gpu_benchmark.sh --background "$@"
}

cmd_status() {
  load_state
  local run_pid=""
  if [[ -f "$PID_PATH" ]]; then
    run_pid="$(cat "$PID_PATH" 2>/dev/null || true)"
  fi

  echo "state: $STATE_PATH"
  echo "csv:   $CSV_PATH"
  echo "log:   $LOG_PATH"

  if is_pid_alive "$run_pid"; then
    echo "run:   ACTIVE (pid=$run_pid)"
  else
    echo "run:   INACTIVE"
  fi

  local tail_pid=""
  if [[ -f "$TAIL_PID_PATH" ]]; then
    tail_pid="$(cat "$TAIL_PID_PATH" 2>/dev/null || true)"
  fi
  if is_pid_alive "$tail_pid"; then
    echo "tail:  ACTIVE (pid=$tail_pid)"
  else
    echo "tail:  INACTIVE"
  fi

  local plot_pid=""
  if [[ -f "$PLOT_PID_PATH" ]]; then
    plot_pid="$(cat "$PLOT_PID_PATH" 2>/dev/null || true)"
  fi
  if is_pid_alive "$plot_pid"; then
    echo "plot:  ACTIVE (pid=$plot_pid)"
  else
    echo "plot:  INACTIVE"
  fi
}

cmd_logs() {
  load_state
  if [[ -f "$LOG_PATH" ]]; then
    tail -n 80 "$LOG_PATH"
  else
    echo "Log file not found yet: $LOG_PATH"
  fi
}

cmd_log_on() {
  load_state
  if [[ ! -f "$LOG_PATH" ]]; then
    echo "Log file not found yet: $LOG_PATH"
    exit 1
  fi

  if [[ -f "$TAIL_PID_PATH" ]] && is_pid_alive "$(cat "$TAIL_PID_PATH")"; then
    echo "Tail monitor already active (pid=$(cat "$TAIL_PID_PATH"))."
    exit 0
  fi

  nohup tail -f "$LOG_PATH" >"$LOG_DIR/long_gpu_benchmark_tail.out" 2>&1 &
  echo $! >"$TAIL_PID_PATH"
  echo "Tail monitor started (pid=$(cat "$TAIL_PID_PATH"))."
  echo "Output: $LOG_DIR/long_gpu_benchmark_tail.out"
}

cmd_log_off() {
  if [[ ! -f "$TAIL_PID_PATH" ]]; then
    echo "No tail monitor pid file found."
    exit 0
  fi

  local pid
  pid="$(cat "$TAIL_PID_PATH" 2>/dev/null || true)"
  if is_pid_alive "$pid"; then
    kill "$pid"
    echo "Tail monitor stopped (pid=$pid)."
  else
    echo "Tail monitor pid not active."
  fi
  rm -f "$TAIL_PID_PATH"
}

cmd_view() {
  load_state
  if [[ ! -f "$CSV_PATH" ]]; then
    echo "CSV not found yet: $CSV_PATH"
    exit 1
  fi

  STATE_PATH_ENV="$STATE_PATH" "$PYTHON_BIN" - <<'PY'
import os
import math
import pandas as pd

state_path = os.environ['STATE_PATH_ENV']
state = {}
with open(state_path, 'r', encoding='utf-8') as f:
    for line in f:
        if '=' in line:
            k, v = line.strip().split('=', 1)
            state[k] = v

csv_path = state['CSV_PATH']
df = pd.read_csv(csv_path)
expected_rows_raw = state.get('EXPECTED_ROWS', '').strip()
expected_rows = int(expected_rows_raw) if expected_rows_raw.isdigit() else None

rows = len(df)
print(f"rows: {rows}")
print(f"last_timestamp: {df['timestamp'].iloc[-1] if rows else 'n/a'}")

if rows:
    runtime = float(df['campaign_runtime_sec'].iloc[-1]) if 'campaign_runtime_sec' in df.columns else float('nan')
    avg_row_sec = runtime / rows if runtime and runtime > 0 else float('nan')
    rows_per_hour = 3600.0 / avg_row_sec if avg_row_sec and avg_row_sec > 0 else float('nan')
    print(f"elapsed_sec: {runtime:.2f}" if math.isfinite(runtime) else "elapsed_sec: n/a")
    print(f"avg_row_sec: {avg_row_sec:.2f}" if math.isfinite(avg_row_sec) else "avg_row_sec: n/a")
    print(f"rows_per_hour: {rows_per_hour:.2f}" if math.isfinite(rows_per_hour) else "rows_per_hour: n/a")

    if expected_rows is not None and expected_rows > 0 and math.isfinite(avg_row_sec):
        remaining = max(expected_rows - rows, 0)
        eta_sec = remaining * avg_row_sec
        print(f"expected_rows: {expected_rows}")
        print(f"remaining_rows: {remaining}")
        print(f"eta_hours: {eta_sec / 3600.0:.2f}")
    else:
        print("expected_rows: n/a")
        print("eta_hours: n/a")

    grouped = df.groupby(['device', 'L'])['throughput_traj_per_sec'].agg(['count', 'mean', 'max']).reset_index()
    print("\nprogress by device/L:")
    print(grouped.to_string(index=False))

    if {'device', 'L', 'batch_size', 'n_steps', 'n_trajectories'}.issubset(df.columns):
        u = df[['device', 'L', 'batch_size', 'n_steps', 'n_trajectories']].drop_duplicates()
        print(f"\nunique tuples completed: {len(u)}")

    last = df.iloc[-1]
    print("\nlatest tuple:")
    print(
        "device={device} L={L} n_steps={n_steps} n_traj={n_trajectories} "
        "batch={batch_size} gamma={gamma} tps={tps:.3f}".format(
            device=last.get('device', 'n/a'),
            L=last.get('L', 'n/a'),
            n_steps=last.get('n_steps', 'n/a'),
            n_trajectories=last.get('n_trajectories', 'n/a'),
            batch_size=last.get('batch_size', 'n/a'),
            gamma=last.get('gamma', 'n/a'),
            tps=float(last.get('throughput_traj_per_sec', float('nan'))),
        )
    )
PY
}

cmd_plot() {
  load_state

  if [[ -f "$PLOT_PID_PATH" ]] && is_pid_alive "$(cat "$PLOT_PID_PATH")"; then
    echo "Plot monitor already active (pid=$(cat "$PLOT_PID_PATH"))."
    exit 0
  fi

  local interval="30"
  if [[ "${1:-}" == "--interval" ]]; then
    interval="${2:-30}"
  fi

  nohup "$PYTHON_BIN" scripts/plot_gpu_benchmark_progress.py --state "$STATE_PATH" --interval "$interval" >"$PLOT_OUT_PATH" 2>&1 &
  echo $! >"$PLOT_PID_PATH"
  echo "Plot monitor started (pid=$(cat "$PLOT_PID_PATH"))."
  echo "Output: $PLOT_OUT_PATH"
  echo "Image: $OUT_DIR/gpu_benchmark_live_progress.png"
}

cmd_plot_stop() {
  if [[ ! -f "$PLOT_PID_PATH" ]]; then
    echo "No plot monitor pid file found."
    exit 0
  fi

  local pid
  pid="$(cat "$PLOT_PID_PATH" 2>/dev/null || true)"
  if is_pid_alive "$pid"; then
    kill "$pid"
    echo "Plot monitor stopped (pid=$pid)."
  else
    echo "Plot monitor pid not active."
  fi
  rm -f "$PLOT_PID_PATH"
}

cmd_stop() {
  load_state
  if [[ ! -f "$PID_PATH" ]]; then
    echo "No campaign pid file found: $PID_PATH"
    exit 0
  fi

  local pid
  pid="$(cat "$PID_PATH" 2>/dev/null || true)"
  if is_pid_alive "$pid"; then
    kill "$pid"
    echo "Stopped campaign process (pid=$pid)."
  else
    echo "Campaign pid not active."
  fi
}

main() {
  local cmd="${1:-}"
  case "$cmd" in
    start)
      shift
      cmd_start "$@"
      ;;
    status)
      cmd_status
      ;;
    logs)
      cmd_logs
      ;;
    log-on)
      cmd_log_on
      ;;
    log-off)
      cmd_log_off
      ;;
    view)
      cmd_view
      ;;
    plot)
      shift
      cmd_plot "$@"
      ;;
    plot-stop)
      cmd_plot_stop
      ;;
    stop)
      cmd_stop
      ;;
    *)
      usage
      exit 1
      ;;
  esac
}

main "$@"
