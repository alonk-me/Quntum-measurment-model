# Single-Qubit Stochastic Schrödinger Equation Simulation

This project implements the stochastic Schrödinger equation (SSE) for a single
qubit monitored in $\sigma_z$ using the Stratonovich interpretation.  The
simulation follows Turkeshi *et al.* and uses symmetric Kraus operators to
reproduce the continuous-measurement dynamics.  The main observable is the
trajectory entropy production $Q$ defined in the accompanying SRS, which is
compared against the analytical distribution of Dressel *et al.* (Eq. 14).

## Contents
- `src/quantum_measurements/`: Python package containing the simulation code
- `qubit_simulation_analysis.ipynb`: Jupyter notebook for running simulations and plotting results
- `sse_qubit.ipynb`: Notebook exploring the stochastic Schrödinger equation model
- `requirements.txt`: List of required Python packages

## Getting Started
1. Create and activate a Python virtual environment (optional but recommended).
2. Install dependencies and the local package (editable install recommended for development):
   ```
   pip install -r requirements.txt
   ```
   This installs the `quantum_measurements` package so it can be imported from anywhere in your environment.
3. Run the command-line interface to generate Monte Carlo statistics (optional):
   ```
   python -m quantum_measurements --num-traj 200 --steps 100 --epsilon 0.05 --save-plot results/q_hist.png
   ```
   The module prints summary statistics for the entropy production and optionally saves a histogram with the Eq. (14) fit.
4. Open `qubit_simulation_analysis.ipynb` in Jupyter or VS Code to explore the simulations and results.

## Requirements
- Python 3.8+
- numpy
- matplotlib
- scipy
- tqdm (optional, for progress bars)

## Description
- Integrates single-qubit SSE trajectories with symmetric Kraus operators
- Computes entropy production according to the Stratonovich midpoint rule
- Generates histograms of $Q$ and overlays the Dressel Eq. (14) prediction with
  the substitution $\theta = 2N\epsilon$
- Saves Monte-Carlo data for reproducibility

## License
This project is for academic and research use.
