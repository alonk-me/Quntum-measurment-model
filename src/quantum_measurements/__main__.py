"""Command-line entry point for the :mod:`quantum_measurements` package."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .qubit_measurement_simulation import (
    plot_histogram_with_theory,
    save_ensemble,
    simulate_ensemble,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate the stochastic Schrödinger equation for a single qubit "
            "under continuous measurement and report entropy production "
            "statistics."
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
        help=(
            "Dimensionless measurement strength ε (default: 0.05).  The "
            "Stratonovich implementation requires |ε| < 1."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for the random number generator (default: None).",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=1.0,
        help="Physical time step associated with each measurement (default: 1.0).",
    )
    parser.add_argument(
        "--save-plot",
        type=Path,
        default=None,
        metavar="PATH",
        help="Save a histogram of Q with the Dressel Eq. 14 overlay to this path.",
    )
    parser.add_argument(
        "--save-data",
        type=Path,
        default=None,
        metavar="PATH",
        help="Persist entropy production values and metadata to a .npz file.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = simulate_ensemble(
        num_trajectories=args.num_traj,
        steps=args.steps,
        epsilon=args.epsilon,
        dt=args.dt,
        seed=args.seed,
    )

    print("Simulated entropy production statistics")
    print("-------------------------------------")
    print(f"Trajectories: {args.num_traj}")
    print(f"Steps per trajectory: {args.steps}")
    print(f"Measurement strength ε: {args.epsilon}")
    print(f"Time step dt: {args.dt}")
    print(f"Mean(Q): {result.Q_values.mean():.3f}")
    print(f"Std(Q): {result.Q_values.std(ddof=1):.3f}")
    print(f"θ = 2Nε: {result.theta:.3f}")

    if args.save_plot is not None:
        plot_histogram_with_theory(result.Q_values, args.epsilon, args.steps, args.save_plot)
        print(f"Saved histogram with theoretical overlay to {args.save_plot}")

    if args.save_data is not None:
        save_ensemble(result, args.save_data)
        print(f"Saved Q values to {args.save_data}")

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
