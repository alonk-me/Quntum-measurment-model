#!/bin/bash
# Quick start script for n_∞(γ) verification workflow
# Activates venv and provides menu options

cd "$(dirname "$0")"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Creating venv..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -e .
    echo "✓ Virtual environment created and dependencies installed"
else
    source venv/bin/activate
    echo "✓ Virtual environment activated"
fi

echo ""
echo "=========================================="
echo "  n_∞(γ) Verification Workflow"
echo "=========================================="
echo ""
echo "Choose an option:"
echo ""
echo "  1) Run smoke test notebook (Jupyter)"
echo "  2) Run production sweep (background)"
echo "  3) Start live monitor"
echo "  4) Generate analysis plots"
echo "  5) View latest results"
echo "  6) Launch tmux session (recommended for remote)"
echo "  7) Python shell (with venv)"
echo "  q) Quit"
echo ""
read -p "Enter choice [1-7 or q]: " choice

case $choice in
    1)
        echo "Launching Jupyter notebook..."
        jupyter notebook notebooks/smoke_test_ninf.ipynb
        ;;
    2)
        echo "Starting production sweep in background..."
        nohup python scripts/run_ninf_scan.py > logs/ninf_scan.log 2>&1 &
        PID=$!
        echo $PID > logs/ninf_scan.pid
        echo "✓ Started with PID: $PID"
        echo "  Monitor: tail -f logs/ninf_scan.log"
        echo "  Stop: kill $PID"
        ;;
    3)
        echo "Starting live progress monitor..."
        echo "Press Ctrl+C to stop"
        python scripts/plot_progress.py
        ;;
    4)
        echo "Generating analysis plots..."
        python scripts/generate_analysis_plots.py
        ;;
    5)
        echo "Latest CSV files:"
        ls -lht results/ninf_scan/*.csv 2>/dev/null | head -3
        echo ""
        echo "Latest plots:"
        ls -lht results/ninf_scan/*.png 2>/dev/null | head -5
        ;;
    6)
        echo "Launching tmux session 'ninf_scan'..."
        tmux new-session -s ninf_scan \; \
            send-keys "cd $(pwd) && source venv/bin/activate && python scripts/run_ninf_scan.py" C-m \; \
            split-window -h \; \
            send-keys "cd $(pwd) && source venv/bin/activate && python scripts/plot_progress.py" C-m \; \
            select-pane -t 0
        ;;
    7)
        echo "Starting Python shell with venv..."
        python
        ;;
    q)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac
