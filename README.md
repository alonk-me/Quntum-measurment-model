# Single Qubit Weak Measurement Simulation

This project contains a Monte Carlo simulation of a single qubit undergoing repeated weak measurements, as described in the Arrow of Time and related papers. The simulation uses diagonal Kraus operators and tracks the stochastic evolution of the qubit state and entropy production $Q$.

## Contents
- `qubit_measurement_simulation.py`: Simulation code and analysis functions
- `qubit_simulation_analysis.ipynb`: Jupyter notebook for running simulations and plotting results
- `requirements.txt`: List of required Python packages

## Getting Started
1. Create and activate a Python virtual environment (optional but recommended).
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Open `qubit_simulation_analysis.ipynb` in Jupyter or VS Code to explore the simulations and results.

## Requirements
- Python 3.8+
- numpy
- matplotlib
- scipy

## Description
- Simulates single qubit weak measurement trajectories
- Computes and analyzes entropy production $Q$
- Compares empirical results to analytical predictions

## License
This project is for academic and research use.
