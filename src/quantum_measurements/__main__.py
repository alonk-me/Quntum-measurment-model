"""Command-line entry point for the :mod:`quantum_measurements` package."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .qubit_measurement_simulation import (
    fit_eq14,
    plot_Q_fit,
    simulate_Q_distribution,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run weak-measurement Monte Carlo simulations for a single qubit "
            "and report summary statistics for the entropy production Q."
        )
    )
    parser.add_argument(
        "--num-traj",
        type=int,
        default=200,
        help="Number of trajectories to simulate (default: 200).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of measurement steps per trajectory (default: 100).",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.05,
        help="Weak measurement strength parameter ε (default: 0.05).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for the random number generator (default: None).",
    )
    parser.add_argument(
        "--save-plot",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "If provided, save a histogram of Q with the Eq. 14 fit to the given "
            "file path (e.g. results/q_distribution.png)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    Q_values = simulate_Q_distribution(
        num_traj=args.num_traj, N=args.steps, epsilon=args.epsilon, seed=args.seed
    )
    theta_hat, theta_err = fit_eq14(Q_values)

    print("Simulated entropy production statistics")
    print("-------------------------------------")
    print(f"Trajectories: {args.num_traj}")
    print(f"Steps per trajectory: {args.steps}")
    print(f"Measurement strength ε: {args.epsilon}")
    print(f"Mean(Q): {Q_values.mean():.3f}")
    print(f"Std(Q): {Q_values.std(ddof=1):.3f}")
    print(f"θ̂ (Eq. 14 fit): {theta_hat:.3f} ± {theta_err:.3f}")

    if args.save_plot is not None:
        args.save_plot.parent.mkdir(parents=True, exist_ok=True)
        plot_Q_fit(Q_values, theta_hat, str(args.save_plot))
        print(f"Saved histogram with fit to {args.save_plot}")

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
