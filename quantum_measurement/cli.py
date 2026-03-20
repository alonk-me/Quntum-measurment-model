"""Command-line interface for the Quantum Measurement simulation toolkit.

Entry point: ``qmeasure``  (registered via ``[project.scripts]`` in
``pyproject.toml``).

Sub-commands
------------
run krauss
    Run a Kraus-operator Monte-Carlo experiment and print summary statistics.
run sse
    Run an SSE ensemble experiment and print summary statistics.

Examples
--------
::

    # Kraus experiment with 500 trajectories, N=10000, epsilon=0.01
    qmeasure run krauss --num-traj 500 --N 10000 --epsilon 0.01

    # SSE experiment
    qmeasure run sse --n-traj 200 --N-steps 500 --epsilon 0.1

    # Save Q distribution plot
    qmeasure run krauss --num-traj 500 --N 10000 --epsilon 0.01 --plot results/Q_dist.png

    # Enable INFO logging
    qmeasure --log-level INFO run sse --n-traj 100 --N-steps 200 --epsilon 0.05
"""

from __future__ import annotations

import argparse
import sys


# ---------------------------------------------------------------------------
# Sub-command implementations
# ---------------------------------------------------------------------------


def _cmd_run_krauss(args: argparse.Namespace) -> None:
    """Execute the Kraus-operator experiment sub-command."""
    from quantum_measurement.experiments import KraussExperimentConfig, run_krauss_experiment
    from quantum_measurement.analysis import compute_statistics, fit_arrow_of_time, plot_Q_distribution

    cfg = KraussExperimentConfig(
        num_traj=args.num_traj,
        N=args.N,
        epsilon=args.epsilon,
        omega_dt=args.omega_dt,
        seed=args.seed,
    )

    print(f"Running Kraus experiment (num_traj={cfg.num_traj}, N={cfg.N}, "
          f"ε={cfg.epsilon}, θ={cfg.theta:.4f}) …")

    result = run_krauss_experiment(cfg, progress=args.progress)
    stats = compute_statistics(result.Q_values)

    print(f"\nResults")
    print(f"  θ (N·ε²)        = {result.theta:.4f}")
    print(f"  ⟨Q⟩ (empirical) = {result.mean_Q:.4f}")
    print(f"  std(Q)          = {result.std_Q:.4f}")
    print(f"  median(Q)       = {stats.median:.4f}")
    print(f"  skewness(Q)     = {stats.skewness:.4f}")

    if args.fit:
        theta_hat, theta_err = fit_arrow_of_time(result.Q_values)
        print(f"\nArrow-of-Time fit")
        print(f"  θ_fit           = {theta_hat:.4f} ± {theta_err:.4f}")

    if args.plot:
        theta_hat = None
        if args.fit:
            theta_hat, _ = fit_arrow_of_time(result.Q_values)
        plot_Q_distribution(
            result.Q_values,
            theta_hat=theta_hat,
            title=f"Kraus Q distribution (θ={result.theta:.3f})",
            filename=args.plot,
        )
        print(f"\nPlot saved to {args.plot}")


def _cmd_run_sse(args: argparse.Namespace) -> None:
    """Execute the SSE experiment sub-command."""
    from quantum_measurement.experiments import SSEExperimentConfig, run_sse_experiment
    from quantum_measurement.analysis import compute_statistics, fit_arrow_of_time, plot_Q_distribution

    cfg = SSEExperimentConfig(
        n_trajectories=args.n_traj,
        epsilon=args.epsilon,
        N_steps=args.N_steps,
        J=args.J,
        initial_state=args.initial_state,
        seed=args.seed,
    )

    print(f"Running SSE experiment (n_traj={cfg.n_trajectories}, N_steps={cfg.N_steps}, "
          f"ε={cfg.epsilon}, J={cfg.J}, θ={cfg.theta:.4f}) …")

    result = run_sse_experiment(cfg, progress=args.progress)
    stats = compute_statistics(result.Q_values)

    print(f"\nResults")
    print(f"  θ (N·ε²)          = {result.config.theta:.4f}")
    print(f"  ⟨Q⟩ (empirical)   = {result.mean_Q:.4f}")
    print(f"  ⟨Q⟩ (theoretical) = {result.theoretical_mean_Q:.4f}")
    print(f"  std(Q)            = {result.std_Q:.4f}")
    print(f"  median(Q)         = {stats.median:.4f}")
    print(f"  skewness(Q)       = {stats.skewness:.4f}")

    if args.fit:
        theta_hat, theta_err = fit_arrow_of_time(result.Q_values)
        print(f"\nArrow-of-Time fit")
        print(f"  θ_fit             = {theta_hat:.4f} ± {theta_err:.4f}")

    if args.plot:
        theta_hat = None
        if args.fit:
            theta_hat, _ = fit_arrow_of_time(result.Q_values)
        plot_Q_distribution(
            result.Q_values,
            theta_hat=theta_hat,
            title=f"SSE Q distribution (θ={cfg.theta:.3f})",
            filename=args.plot,
        )
        print(f"\nPlot saved to {args.plot}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qmeasure",
        description="Quantum Measurement simulation toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: WARNING)",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        metavar="PATH",
        help="Optional file to write log output to",
    )

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # -- run ---------------------------------------------------------------
    run_parser = sub.add_parser("run", help="Run a simulation experiment")
    run_sub = run_parser.add_subparsers(dest="experiment", metavar="EXPERIMENT")
    run_sub.required = True

    # -- run krauss --------------------------------------------------------
    krauss_p = run_sub.add_parser(
        "krauss",
        help="Kraus-operator Monte-Carlo experiment",
        description="Run a Kraus-operator Monte-Carlo simulation of repeated weak measurements.",
    )
    krauss_p.add_argument("--num-traj", type=int, default=500,
                          help="Number of trajectories (default: 500)")
    krauss_p.add_argument("--N", type=int, default=10_000,
                          help="Measurement steps per trajectory (default: 10000)")
    krauss_p.add_argument("--epsilon", type=float, default=0.01,
                          help="Measurement strength ε (default: 0.01)")
    krauss_p.add_argument("--omega-dt", type=float, default=0.0,
                          help="Coherent rotation frequency × dt (default: 0.0)")
    krauss_p.add_argument("--seed", type=int, default=None,
                          help="Random seed for reproducibility")
    krauss_p.add_argument("--fit", action="store_true",
                          help="Fit Q distribution to Arrow-of-Time density")
    krauss_p.add_argument("--plot", metavar="FILE", default=None,
                          help="Save Q distribution plot to FILE")
    krauss_p.add_argument("--progress", action="store_true",
                          help="Show tqdm progress bar")
    krauss_p.set_defaults(func=_cmd_run_krauss)

    # -- run sse -----------------------------------------------------------
    sse_p = run_sub.add_parser(
        "sse",
        help="Stochastic Schrödinger Equation ensemble experiment",
        description="Run an SSE wavefunction ensemble simulation.",
    )
    sse_p.add_argument("--n-traj", type=int, default=200,
                       help="Number of trajectories (default: 200)")
    sse_p.add_argument("--N-steps", type=int, default=200,
                       help="Measurement steps per trajectory (default: 200)")
    sse_p.add_argument("--epsilon", type=float, default=0.1,
                       help="Measurement strength ε (default: 0.1)")
    sse_p.add_argument("--J", type=float, default=0.0,
                       help="Hamiltonian coupling J (default: 0.0)")
    sse_p.add_argument("--initial-state", default="bloch_equator",
                       choices=["bloch_equator", "up", "down", "plus_y", "minus_y", "custom"],
                       help="Initial quantum state preset (default: bloch_equator)")
    sse_p.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility")
    sse_p.add_argument("--fit", action="store_true",
                       help="Fit Q distribution to Arrow-of-Time density")
    sse_p.add_argument("--plot", metavar="FILE", default=None,
                       help="Save Q distribution plot to FILE")
    sse_p.add_argument("--progress", action="store_true",
                       help="Show tqdm progress bar")
    sse_p.set_defaults(func=_cmd_run_sse)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv=None) -> None:
    """Main entry point for the ``qmeasure`` CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Configure logging early
    from quantum_measurement.monitoring import configure_logging
    configure_logging(level=args.log_level, log_file=args.log_file)

    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        import traceback
        logging.getLogger("quantum_measurement").debug(traceback.format_exc())
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
