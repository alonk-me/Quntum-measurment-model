#!/usr/bin/env python3
"""Run a paired-noise validation of the single-qubit AoT tangent method.

Example
-------
MPLCONFIGDIR=/tmp/mpl-aot .venv/bin/python \
    scripts/validate_aot_single_qubit.py \
    --g-values 0.3 1.0 4.0 \
    --h-values 0.05 0.005 0.0001 0.00001 \
    --n-trajectories 512 --n-steps 20000 --burn-in 800
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

# Make direct execution from ``scripts/`` behave like the repository's tests.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from quantum_measurement.aot_single_qubit import (
    mean_and_sem,
    paired_derivative_validation,
    paired_summary,
    rademacher_noise,
    simulate_ensemble,
    stationary_aot_density,
    stationary_aot_susceptibility,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g-values", nargs="+", type=float, default=[0.3, 1.0, 4.0])
    parser.add_argument(
        "--h-values",
        nargs="+",
        type=float,
        default=[5.0e-2, 5.0e-3, 1.0e-4, 1.0e-5],
    )
    parser.add_argument("--n-trajectories", type=int, default=512)
    parser.add_argument("--n-steps", type=int, default=20000)
    parser.add_argument("--burn-in", type=int, default=800)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--reference-grid", type=int, default=4096)
    parser.add_argument("--reference-h", type=float, default=0.01)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/aot_single_qubit_validation"),
    )
    return parser.parse_args()


def running_means(values: np.ndarray) -> list[dict[str, float]]:
    sizes = []
    size = 16
    while size < values.size:
        sizes.append(size)
        size *= 2
    sizes.append(values.size)
    rows = []
    for size in sorted(set(sizes)):
        mean, sem = mean_and_sem(values[:size])
        rows.append({"n": int(size), "mean": mean, "sem": sem})
    return rows


def trajectory_diagnostics(values: np.ndarray) -> dict[str, object]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "sem": float(np.std(values, ddof=1) / np.sqrt(values.size)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "quantiles": {
            str(q): float(value)
            for q, value in zip(
                [0.01, 0.1, 0.5, 0.9, 0.99],
                np.quantile(values, [0.01, 0.1, 0.5, 0.9, 0.99]),
            )
        },
        "running_means": running_means(values),
    }


def main() -> None:
    args = parse_args()
    if args.burn_in >= args.n_steps:
        raise ValueError("burn-in must be smaller than n-steps")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    all_points = []
    for index, g in enumerate(args.g_values):
        noise = rademacher_noise(
            args.n_trajectories,
            args.n_steps,
            args.seed + index,
        )
        start = perf_counter()
        central, _ = simulate_ensemble(
            4.0 * args.J * g,
            noise,
            J=args.J,
            dt=args.dt,
            burn_in=args.burn_in,
        )
        tangent_seconds = perf_counter() - start
        q_mean, q_sem = mean_and_sem(central.q)
        chi_mean, chi_sem = mean_and_sem(central.chi_q)
        reference_q = stationary_aot_density(
            g,
            n_grid=args.reference_grid,
            J=args.J,
            dt=args.dt,
        )
        reference_chi = stationary_aot_susceptibility(
            g,
            h=args.reference_h,
            n_grid=args.reference_grid,
            J=args.J,
            dt=args.dt,
        )

        record = {
            "g": g,
            "gamma": 4.0 * args.J * g,
            "q_mean": q_mean,
            "q_sem": q_sem,
            "chi_q_tangent_mean": chi_mean,
            "chi_q_tangent_sem": chi_sem,
            "reference_q": reference_q,
            "reference_chi_q": reference_chi,
            "tangent_bias_from_reference": chi_mean - reference_chi,
            "tangent_reference_z_score": (
                (chi_mean - reference_chi) / chi_sem if chi_sem > 0 else 0.0
            ),
            "tangent_seconds": tangent_seconds,
            "max_tangent_norm": float(np.max(central.max_tangent_norm)),
            "max_norm_error": central.max_norm_error,
            "max_gauge_error": central.max_gauge_error,
            "tangent_distribution": trajectory_diagnostics(central.chi_q),
            "step_sizes": [],
        }

        for h in args.h_values:
            start = perf_counter()
            paired, _ = paired_derivative_validation(
                g,
                h,
                noise,
                J=args.J,
                dt=args.dt,
                burn_in=args.burn_in,
                central=central,
            )
            finite_difference_seconds = perf_counter() - start
            summary = paired_summary(paired)
            summary["finite_difference_bias_from_reference"] = (
                summary["finite_difference_mean"] - reference_chi
            )
            summary["finite_difference_reference_z_score"] = (
                summary["finite_difference_bias_from_reference"]
                / summary["finite_difference_sem"]
                if summary["finite_difference_sem"] > 0
                else 0.0
            )
            summary["finite_difference_seconds"] = finite_difference_seconds
            summary["fd_to_tangent_variance_ratio"] = (
                summary["finite_difference_std"] ** 2
                / summary["tangent_std"] ** 2
                if summary["tangent_std"] > 0
                else 0.0
            )
            record["step_sizes"].append(summary)
            all_points.append({"g": g, **summary})

        all_records.append(record)
        coarse = record["step_sizes"][0]
        local = record["step_sizes"][-1]
        print(
            f"g={g:g}: tangent={chi_mean:+.6g} +/- {chi_sem:.3g}; "
            f"reference={reference_chi:+.6g}; "
            f"FD(h={coarse['h']:g})={coarse['finite_difference_mean']:+.6g} "
            f"+/- {coarse['finite_difference_sem']:.3g}; "
            f"paired RMS(h={local['h']:g})={local['difference_rms']:.3g}; "
            f"max|eta|={record['max_tangent_norm']:.3g}"
        )

    configuration = {
        "g_values": args.g_values,
        "h_values": args.h_values,
        "n_trajectories": args.n_trajectories,
        "n_steps": args.n_steps,
        "burn_in": args.burn_in,
        "dt": args.dt,
        "J": args.J,
        "seed": args.seed,
        "reference_grid": args.reference_grid,
        "reference_h": args.reference_h,
    }
    json_path = args.output_dir / "validation_results.json"
    json_path.write_text(
        json.dumps({"configuration": configuration, "results": all_records}, indent=2)
        + "\n"
    )

    csv_path = args.output_dir / "step_size_summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_points[0]))
        writer.writeheader()
        writer.writerows(all_points)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for record in all_records:
        h = np.array([row["h"] for row in record["step_sizes"]])
        rms = np.array([row["difference_rms"] for row in record["step_sizes"]])
        fd_std = np.array(
            [row["finite_difference_std"] for row in record["step_sizes"]]
        )
        axes[0].loglog(h, rms, "o-", label=f"g={record['g']:g}")
        axes[1].loglog(h, fd_std, "o-", label=f"FD, g={record['g']:g}")
        axes[1].axhline(
            record["tangent_distribution"]["std"],
            linestyle="--",
            linewidth=1,
            label=f"tangent, g={record['g']:g}",
        )
    axes[0].set_xlabel(r"central-difference step $h$ in $\ln g$")
    axes[0].set_ylabel(r"paired RMS $\|\chi_q-D_hq\|$")
    axes[0].set_title("Pathwise convergence to the tangent")
    axes[1].set_xlabel(r"central-difference step $h$ in $\ln g$")
    axes[1].set_ylabel("trajectory-estimator standard deviation")
    axes[1].set_title("Finite-difference regularization and variance")
    for axis in axes:
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output_dir / "step_size_and_variance.png", dpi=180)
    plt.close(fig)

    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
