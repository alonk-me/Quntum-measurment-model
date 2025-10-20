"""
Command‑line utility to run ensembles of stochastic Schrödinger trajectories
and plot their entropy production statistics.

This script uses the :class:`SingleQubitSSE` from ``sse.py`` to simulate
a large number of trajectories, compute the trajectory‑dependent
entropy production Q for each, and display the resulting histogram
alongside the analytical distribution from Eq. (14).  The gamma,
dt, T and number of trajectories can be specified via the command
line or modified in the code.

Usage (from the repository root)::

    python3 -m sse_simulation.run_simulation --gamma 0.5 --dt 0.001 --T 1.0 --n 5000

The resulting plot is saved as ``entropy_production_histogram.png`` in
the current working directory.
"""

from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt

from .sse import SingleQubitSSE, eq14_distribution


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SSE simulations and plot entropy production histogram.")
    parser.add_argument("--gamma", type=float, default=0.5, help="Measurement strength γ")
    parser.add_argument("--dt", type=float, default=0.001, help="Integration timestep dt")
    parser.add_argument("--T", type=float, default=1.0, help="Total simulation time T")
    parser.add_argument("--n", type=int, default=2000, help="Number of trajectories")
    parser.add_argument("--state", type=str, default="plus", choices=["0","1","plus","minus"],
                        help="Initial qubit state: |0>, |1>, |+> (x+), or |-> (x-) ")
    parser.add_argument("--bins", type=int, default=100, help="Number of histogram bins")
    parser.add_argument("--out", type=str, default="entropy_production_histogram.png", help="Output filename for the plot")
    args = parser.parse_args()

    # Construct initial state
    if args.state == "0":
        psi0 = np.array([1.0, 0.0], dtype=complex)
    elif args.state == "1":
        psi0 = np.array([0.0, 1.0], dtype=complex)
    elif args.state == "plus":
        psi0 = (1/np.sqrt(2)) * np.array([1.0, 1.0], dtype=complex)
    elif args.state == "minus":
        psi0 = (1/np.sqrt(2)) * np.array([1.0, -1.0], dtype=complex)
    else:
        raise ValueError(f"Unknown state {args.state}")

    # Initialise simulator
    sim = SingleQubitSSE(gamma=args.gamma, dt=args.dt, T=args.T)

    # Run ensemble
    print(f"Running {args.n} trajectories...")
    _, Q_vals = sim.run_ensemble(psi0, args.n)
    print("Simulation complete.")

    # Plot histogram of Q
    fig, ax = plt.subplots(figsize=(8, 5))
    counts, edges, _ = ax.hist(Q_vals, bins=args.bins, density=True, alpha=0.6, label="Simulation")
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Evaluate analytical distribution on the same range
    x_min, x_max = edges[0], edges[-1]
    x_grid = np.linspace(x_min, x_max, 1000)
    P = eq14_distribution(x_grid, gamma=args.gamma, T=args.T)
    # Renormalise P numerically to unit area
    area = np.trapz(P, x_grid)
    if area > 0:
        P /= area
    ax.plot(x_grid, P, 'r-', lw=2, label="Eq. 14 (scaled)")

    ax.set_xlabel("Q = ln R")
    ax.set_ylabel("Probability density")
    ax.set_title(f"Entropy production distribution (γ={args.gamma}, T={args.T}, dt={args.dt}, N={args.n})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Plot saved as {args.out}")


if __name__ == "__main__":
    main()
