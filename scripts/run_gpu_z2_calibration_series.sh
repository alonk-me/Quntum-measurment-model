#!/bin/bash
# GPU Z2 Calibration Series Runner
# Runs L=3, L=9, L=17 sequentially with 16 trajectories each
# Each scan writes to a separate CSV file

set -e

WORKSPACE="/home/alon/Documents/VS_code/Quntum-measurment-model"
cd "$WORKSPACE"

# Common settings
DEVICE="gpu"
TRAJECTORIES=16
BATCH_SIZE=16
COMPUTE_UNCERTAINTY="--compute-uncertainty"
PARALLEL_BACKEND="sequential"

# L values to scan
L_VALUES=(3 9 17)

# Function to run a single L scan
run_l_scan() {
    local L=$1
    local CSV_FILE="results/z2_scan/z2_gpu_L${L}_calib_16traj.csv"
    
    echo "=========================================="
    echo "Starting L=$L GPU calibration scan"
    echo "CSV output: $CSV_FILE"
    echo "Timestamp: $(date)"
    echo "=========================================="
    
    python scripts/run_z2_scan.py \
        --device $DEVICE \
        --l-values $L \
        --n-trajectories-per-point $TRAJECTORIES \
        --batch-size-per-point $BATCH_SIZE \
        $COMPUTE_UNCERTAINTY \
        --csv "$CSV_FILE" \
        --parallel-backend $PARALLEL_BACKEND \
        --no-resume
    
    local EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ L=$L scan completed successfully"
        
        # Extract summary statistics
        if [ -f "$CSV_FILE" ]; then
            local POINTS=$(tail -n +2 "$CSV_FILE" | wc -l)
            echo "  Data points written: $POINTS"
            echo "  Last entry:"
            tail -1 "$CSV_FILE" | cut -d',' -f1-5
        fi
    else
        echo "✗ L=$L scan failed with exit code $EXIT_CODE"
        return $EXIT_CODE
    fi
    
    echo ""
}

# Main execution loop
START_TIME=$(date +%s)
echo "=== GPU Z2 Calibration Series ==="
echo "Starting at $(date)"
echo "L values to scan: ${L_VALUES[@]}"
echo ""

for L in "${L_VALUES[@]}"; do
    run_l_scan $L
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "=========================================="
echo "All scans completed!"
echo "Total runtime: $(($ELAPSED / 3600))h $(($ELAPSED % 3600 / 60))m $(($ELAPSED % 60))s"
echo "Results in: results/z2_scan/z2_gpu_L*_calib_16traj.csv"
echo "=========================================="

# Summary analysis
echo ""
echo "=== Calibration Summary ==="
for L in "${L_VALUES[@]}"; do
    CSV="results/z2_scan/z2_gpu_L${L}_calib_16traj.csv"
    if [ -f "$CSV" ]; then
        POINTS=$(tail -n +2 "$CSV" | wc -l)
        echo "L=$L: $POINTS points"
    fi
done
