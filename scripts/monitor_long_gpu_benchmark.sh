#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="logs"
OUT_DIR="results/test_scan"
STATE_PATH="$LOG_DIR/long_gpu_benchmark.state"
TAIL_PID_PATH="$LOG_DIR/long_gpu_benchmark_tail.pid"

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
  stop                           Stop running campaign process

Examples:
  ./scripts/monitor_long_gpu_benchmark.sh start --L-values 64 128 --n-steps-values 10000 30000
  ./scripts/monitor_long_gpu_benchmark.sh status
  ./scripts/monitor_long_gpu_benchmark.sh logs
  ./scripts/monitor_long_gpu_benchmark.sh log-on
  ./scripts/monitor_long_gpu_benchmark.sh log-off
  ./scripts/monitor_long_gpu_benchmark.sh view
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

  /home/alon/Documents/VS_code/Quntum-measurment-model/.venv/bin/python - <<'PY'
import os
import pandas as pd

state_path = os.path.join('logs', 'long_gpu_benchmark.state')
state = {}
with open(state_path, 'r', encoding='utf-8') as f:
    for line in f:
        if '=' in line:
            k, v = line.strip().split('=', 1)
            state[k] = v

csv_path = state['CSV_PATH']
df = pd.read_csv(csv_path)

print(f"rows: {len(df)}")
print(f"last_timestamp: {df['timestamp'].iloc[-1] if len(df) else 'n/a'}")

if len(df):
    grouped = df.groupby(['device', 'L'])['throughput_traj_per_sec'].agg(['count', 'mean', 'max']).reset_index()
    print("\nprogress by device/L:")
    print(grouped.to_string(index=False))

    if {'device', 'L', 'batch_size', 'n_steps', 'n_trajectories'}.issubset(df.columns):
        u = df[['device', 'L', 'batch_size', 'n_steps', 'n_trajectories']].drop_duplicates()
        print(f"\nunique tuples completed: {len(u)}")
PY
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
