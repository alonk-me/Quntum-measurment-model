#!/bin/bash
# Start z2 scan with live monitor in tmux

set -e

cd "$(dirname "$0")/.."

# Determine CSV filename
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
CSV_FILE="results/z2_scan/z2_data_${TIMESTAMP}.csv"

echo "========================================"
echo "Starting z2 scan with live monitor"
echo "========================================"
echo "CSV output: $CSV_FILE"
echo ""
echo "Tmux session: z2_scan"
echo "Left pane: scan runner"
echo "Right pane: live monitor"
echo ""
echo "Detach: Ctrl+b, then d"
echo "Reattach: tmux attach -t z2_scan"
echo "========================================"
echo ""

# Create tmux session with two panes
tmux new-session -d -s z2_scan -n scan

# Pane 0: Run the main scan
tmux send-keys -t z2_scan:scan.0 "cd $(pwd)" C-m
tmux send-keys -t z2_scan:scan.0 "python scripts/run_z2_scan.py --csv $CSV_FILE" C-m

# Split vertically and create pane 1
tmux split-window -h -t z2_scan:scan

# Pane 1: Wait for CSV to exist, then start monitor
tmux send-keys -t z2_scan:scan.1 "cd $(pwd)" C-m
tmux send-keys -t z2_scan:scan.1 "echo 'Waiting for scan to start...'" C-m
tmux send-keys -t z2_scan:scan.1 "while [ ! -f $CSV_FILE ]; do sleep 2; done" C-m
tmux send-keys -t z2_scan:scan.1 "python scripts/plot_z2_progress.py --csv $CSV_FILE" C-m

# Attach to session
echo "Attaching to tmux session..."
sleep 1
tmux attach-t z2_scan
