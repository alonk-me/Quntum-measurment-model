#!/usr/bin/env python3
from __future__ import annotations

"""Reproduce Figure 2 style QSD entanglement growth curves.

This script uses the existing free-fermion trajectory worker engine that already
returns trajectory-level bipartite entanglement entropy ``S_A(t)``.

Outputs:
- CSV (long format): one row per (L, gamma, time_step)
- NPZ: compact arrays for plotting/reload
- PNG: multi-panel Figure-2 style plot (panels by gamma, curves by L)
"""

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from quantum_measurement.backends import MultiCpuBackend, MultiCpuBackendConfig
from quantum_measurement.parallel.trajectory_worker import (
    TrajectoryConfig,
    TrajectoryTask,
    run_single_trajectory,
)


def parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def get_time_params(gamma: float, t_multiplier: float, t_min: float, dt_ratio: float, dt_max: float) -> tuple[float, float, int]:
    tau = 1.0 / gamma
    T = max(tau * t_multiplier, t_min)
    dt = min(tau * dt_ratio, dt_max)
    n_steps = int(round(T / dt))
    return float(T), float(dt), int(n_steps)


def resolve_subsystem_sites(L: int, subsystem_sites: int | None, subsystem_fraction: float) -> int:
    if subsystem_sites is not None:
        n_sites = int(subsystem_sites)
    else:
        n_sites = int(round(float(L) * float(subsystem_fraction)))
    n_sites = max(1, n_sites)
    n_sites = min(int(L), n_sites)
    return n_sites


def apply_profile_overrides(
    profile: str,
    *,
    t_multiplier: float,
    t_min: float,
    dt_ratio: float,
    dt_max: float,
) -> tuple[float, float, float, float]:
    if profile == "diagnostic-fast":
        # Fast checkpoint profile intended for launch/health validation runs.
        return 1.0, 5.0, 5e-3, 1e-3
    return float(t_multiplier), float(t_min), float(dt_ratio), float(dt_max)


def run_serial_trajectories(total_trajectories: int, config: TrajectoryConfig, master_seed: int) -> list[tuple[int, np.ndarray]]:
    seed_seq = np.random.SeedSequence(master_seed)
    children = seed_seq.spawn(total_trajectories)
    out: list[tuple[int, np.ndarray]] = []
    for i in range(total_trajectories):
        child_seed = int(children[i].generate_state(1)[0])
        task = TrajectoryTask(traj_id=i, child_seed=child_seed)
        out.append(run_single_trajectory(task, config))
    out.sort(key=lambda x: x[0])
    return out


@dataclass
class PointResult:
    L: int
    gamma: float
    dt: float
    n_steps: int
    t_grid: np.ndarray
    s_mean: np.ndarray
    s_std: np.ndarray


def point_key(L: int, gamma: float) -> str:
    gamma_token = format(float(gamma), ".12g").replace("-", "m").replace(".", "p")
    return f"L{int(L)}_g{gamma_token}"


def checkpoint_file_for(checkpoint_dir: Path, L: int, gamma: float) -> Path:
    return checkpoint_dir / f"{point_key(L, gamma)}.npz"


def partial_checkpoint_file_for(checkpoint_dir: Path, L: int, gamma: float) -> Path:
    return checkpoint_dir / f"{point_key(L, gamma)}.partial.npz"


def _atomic_savez(path: Path, payload: dict[str, np.ndarray]) -> None:
    tmp_path = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp_path, **payload)
    tmp_path.replace(path)


def save_point_checkpoint(
    checkpoint_dir: Path,
    result: PointResult,
    *,
    n_trajectories: int,
    mode: str,
    closed_boundary: bool,
    J: float,
    subsystem_sites: int,
    subsystem_start: int,
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    out = checkpoint_file_for(checkpoint_dir, result.L, result.gamma)
    payload: dict[str, np.ndarray] = {
        "L": np.asarray(result.L, dtype=int),
        "gamma": np.asarray(result.gamma, dtype=float),
        "dt": np.asarray(result.dt, dtype=float),
        "n_steps": np.asarray(result.n_steps, dtype=int),
        "t_grid": np.asarray(result.t_grid, dtype=float),
        "s_mean": np.asarray(result.s_mean, dtype=float),
        "s_std": np.asarray(result.s_std, dtype=float),
        "n_trajectories": np.asarray(n_trajectories, dtype=int),
        "mode": np.asarray(mode, dtype=str),
        "closed_boundary": np.asarray(int(bool(closed_boundary)), dtype=int),
        "J": np.asarray(float(J), dtype=float),
        "subsystem_sites": np.asarray(int(subsystem_sites), dtype=int),
        "subsystem_start": np.asarray(int(subsystem_start), dtype=int),
    }
    _atomic_savez(out, payload)
    partial = partial_checkpoint_file_for(checkpoint_dir, result.L, result.gamma)
    if partial.exists():
        partial.unlink()
    return out


def save_point_partial_checkpoint(
    checkpoint_dir: Path,
    *,
    L: int,
    gamma: float,
    dt: float,
    n_steps: int,
    t_grid: np.ndarray,
    s_mean_partial: np.ndarray,
    s_std_partial: np.ndarray,
    n_completed: int,
    n_total: int,
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    out = partial_checkpoint_file_for(checkpoint_dir, L, gamma)
    payload: dict[str, np.ndarray] = {
        "L": np.asarray(int(L), dtype=int),
        "gamma": np.asarray(float(gamma), dtype=float),
        "dt": np.asarray(float(dt), dtype=float),
        "n_steps": np.asarray(int(n_steps), dtype=int),
        "t_grid": np.asarray(t_grid, dtype=float),
        "s_mean_partial": np.asarray(s_mean_partial, dtype=float),
        "s_std_partial": np.asarray(s_std_partial, dtype=float),
        "n_completed": np.asarray(int(n_completed), dtype=int),
        "n_total": np.asarray(int(n_total), dtype=int),
        "updated_at": np.asarray(datetime.now().isoformat(), dtype=str),
    }
    _atomic_savez(out, payload)
    return out


def load_point_checkpoint(
    checkpoint_file: Path,
    *,
    expected_n_trajectories: int,
    expected_mode: str,
    expected_closed_boundary: bool,
    expected_J: float,
    expected_subsystem_sites: int,
    expected_subsystem_start: int,
) -> PointResult:
    with np.load(checkpoint_file, allow_pickle=False) as data:
        n_trajectories = int(data["n_trajectories"])
        mode = str(data["mode"])  # numpy string scalar
        closed_boundary = bool(int(data["closed_boundary"]))
        J = float(data["J"])
        subsystem_sites = int(data["subsystem_sites"]) if "subsystem_sites" in data else None
        subsystem_start = int(data["subsystem_start"]) if "subsystem_start" in data else None

        if n_trajectories != int(expected_n_trajectories):
            raise ValueError("checkpoint n_trajectories mismatch")
        if mode != str(expected_mode):
            raise ValueError("checkpoint mode mismatch")
        if closed_boundary != bool(expected_closed_boundary):
            raise ValueError("checkpoint boundary mismatch")
        if not np.isclose(J, float(expected_J), rtol=0.0, atol=1e-15):
            raise ValueError("checkpoint J mismatch")
        if subsystem_sites is None or subsystem_sites != int(expected_subsystem_sites):
            raise ValueError("checkpoint subsystem_sites mismatch")
        if subsystem_start is None or subsystem_start != int(expected_subsystem_start):
            raise ValueError("checkpoint subsystem_start mismatch")

        result = PointResult(
            L=int(data["L"]),
            gamma=float(data["gamma"]),
            dt=float(data["dt"]),
            n_steps=int(data["n_steps"]),
            t_grid=np.asarray(data["t_grid"], dtype=float),
            s_mean=np.asarray(data["s_mean"], dtype=float),
            s_std=np.asarray(data["s_std"], dtype=float),
        )

    if result.t_grid.shape[0] != result.n_steps + 1:
        raise ValueError("checkpoint time-grid length mismatch")
    if result.s_mean.shape != result.t_grid.shape or result.s_std.shape != result.t_grid.shape:
        raise ValueError("checkpoint entropy array shape mismatch")
    if not (np.all(np.isfinite(result.s_mean)) and np.all(np.isfinite(result.s_std))):
        raise ValueError("checkpoint contains non-finite entropy values")
    return result


def build_run_fingerprint(args: argparse.Namespace, L_values: list[int], gamma_values: list[float]) -> str:
    payload = {
        "L_values": [int(x) for x in L_values],
        "gamma_values": [float(x) for x in gamma_values],
        "n_trajectories": int(args.n_trajectories),
        "J": float(args.J),
        "closed_boundary": bool(args.closed_boundary),
        "mode": str(args.mode),
        "workers": int(args.workers),
        "seed": int(args.seed),
        "t_multiplier": float(args.t_multiplier),
        "t_min": float(args.t_min),
        "dt_ratio": float(args.dt_ratio),
        "dt_max": float(args.dt_max),
        "profile": str(args.profile),
        "subsystem_sites": None if args.subsystem_sites is None else int(args.subsystem_sites),
        "subsystem_fraction": float(args.subsystem_fraction),
        "subsystem_start": int(args.subsystem_start),
        "partial_checkpoint_every_chunks": int(args.partial_checkpoint_every_chunks),
        "tag": str(args.tag),
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def run_point(
    *,
    L: int,
    gamma: float,
    J: float,
    n_trajectories: int,
    closed_boundary: bool,
    t_multiplier: float,
    t_min: float,
    dt_ratio: float,
    dt_max: float,
    subsystem_sites: int,
    subsystem_start: int,
    mode: str,
    workers: int,
    master_seed: int,
    checkpoint_dir: Path | None = None,
    partial_checkpoint_every_chunks: int = 1,
) -> PointResult:
    _, dt, n_steps = get_time_params(gamma, t_multiplier, t_min, dt_ratio, dt_max)
    config = TrajectoryConfig(
        L=int(L),
        J=float(J),
        gamma=float(gamma),
        dt=float(dt),
        n_steps=int(n_steps),
        closed_boundary=bool(closed_boundary),
        subsystem_sites=int(subsystem_sites),
        subsystem_start=int(subsystem_start),
    )

    if mode == "serial":
        results = run_serial_trajectories(n_trajectories, config, master_seed)
    else:
        running_count = 0
        running_sum: np.ndarray | None = None
        running_sum_sq: np.ndarray | None = None
        chunk_counter = 0

        def _progress_callback(chunk_out: list[tuple[int, np.ndarray]], completed: int, total: int) -> None:
            nonlocal running_count, running_sum, running_sum_sq, chunk_counter
            chunk_counter += 1
            for _, arr in chunk_out:
                if running_sum is None:
                    running_sum = np.zeros_like(arr, dtype=np.float64)
                    running_sum_sq = np.zeros_like(arr, dtype=np.float64)
                arr64 = np.asarray(arr, dtype=np.float64)
                running_sum += arr64
                running_sum_sq += arr64 * arr64
                running_count += 1

            if checkpoint_dir is None:
                return
            if partial_checkpoint_every_chunks < 1:
                return
            if running_count <= 0 or running_count >= n_trajectories:
                return
            if (chunk_counter % partial_checkpoint_every_chunks) != 0:
                return
            if running_sum is None or running_sum_sq is None:
                return

            mean = running_sum / float(running_count)
            var = np.maximum((running_sum_sq / float(running_count)) - (mean * mean), 0.0)
            std = np.sqrt(var)
            t_grid_partial = np.arange(n_steps + 1, dtype=float) * dt
            save_point_partial_checkpoint(
                checkpoint_dir,
                L=L,
                gamma=gamma,
                dt=dt,
                n_steps=n_steps,
                t_grid=t_grid_partial,
                s_mean_partial=mean,
                s_std_partial=std,
                n_completed=completed,
                n_total=total,
            )

        backend = MultiCpuBackend(
            MultiCpuBackendConfig(
                max_workers=max(1, int(workers)),
                master_seed=int(master_seed),
                nan_mode="fail_on_nan",
            )
        )
        results, _ = backend.run_trajectories(
            total_trajectories=n_trajectories,
            config=config,
            progress_callback=_progress_callback,
        )

    s_stack = np.stack([arr for _, arr in results], axis=0)
    s_mean = np.mean(s_stack, axis=0)
    s_std = np.std(s_stack, axis=0)
    t_grid = np.arange(n_steps + 1, dtype=float) * dt
    return PointResult(L=L, gamma=gamma, dt=dt, n_steps=n_steps, t_grid=t_grid, s_mean=s_mean, s_std=s_std)


def write_long_csv(out_csv: Path, point_results: list[PointResult], n_trajectories: int, mode: str, closed_boundary: bool, seed: int) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp",
                "L",
                "gamma",
                "time",
                "step",
                "S_mean",
                "S_std",
                "dt",
                "n_steps",
                "n_trajectories",
                "mode",
                "closed_boundary",
                "seed",
            ]
        )
        now = datetime.now().isoformat()
        for res in point_results:
            for step, t in enumerate(res.t_grid):
                writer.writerow(
                    [
                        now,
                        res.L,
                        res.gamma,
                        float(t),
                        int(step),
                        float(res.s_mean[step]),
                        float(res.s_std[step]),
                        float(res.dt),
                        int(res.n_steps),
                        int(n_trajectories),
                        mode,
                        bool(closed_boundary),
                        int(seed),
                    ]
                )


def write_npz(out_npz: Path, point_results: list[PointResult], L_values: list[int], gamma_values: list[float]) -> None:
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "L_values": np.asarray(L_values, dtype=int),
        "gamma_values": np.asarray(gamma_values, dtype=float),
    }
    for res in point_results:
        key = f"L{res.L}_g{res.gamma:g}"
        payload[f"t_{key}"] = res.t_grid
        payload[f"smean_{key}"] = res.s_mean
        payload[f"sstd_{key}"] = res.s_std
    np.savez_compressed(out_npz, **payload)


def plot_results(out_png: Path, point_results: list[PointResult], L_values: list[int], gamma_values: list[float]) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    n_gamma = len(gamma_values)
    n_cols = min(3, n_gamma)
    n_rows = int(np.ceil(n_gamma / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 3.6 * n_rows), squeeze=False)
    color_map = {L: c for L, c in zip(L_values, plt.cm.viridis(np.linspace(0.15, 0.9, len(L_values))))}

    for idx, gamma in enumerate(gamma_values):
        ax = axes[idx // n_cols][idx % n_cols]
        panel_data = [r for r in point_results if np.isclose(r.gamma, gamma)]
        panel_data.sort(key=lambda r: r.L)
        for res in panel_data:
            color = color_map[res.L]
            ax.plot(res.t_grid, res.s_mean, color=color, linewidth=1.6, label=f"L={res.L}")
            ax.fill_between(res.t_grid, res.s_mean - res.s_std, res.s_mean + res.s_std, color=color, alpha=0.15)

        ax.set_xscale("log")
        ax.set_title(f"gamma={gamma:g}")
        ax.set_xlabel("t")
        ax.set_ylabel("S_A(t)")
        ax.grid(True, alpha=0.3)

    for idx in range(n_gamma, n_rows * n_cols):
        ax = axes[idx // n_cols][idx % n_cols]
        ax.axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 6), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproduce Figure-2-style QSD entanglement growth curves.")
    parser.add_argument("--L-values", default="32,64,128", help="Comma-separated L values.")
    parser.add_argument("--gamma-values", default="1,2,3,4,6,8", help="Comma-separated gamma values.")
    parser.add_argument("--n-trajectories", type=int, default=200)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--closed-boundary", action="store_true", help="Enable periodic boundary conditions.")
    parser.add_argument("--mode", choices=["serial", "multi-cpu"], default="multi-cpu")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--t-multiplier", type=float, default=60.0)
    parser.add_argument("--t-min", type=float, default=100.0)
    parser.add_argument("--dt-ratio", type=float, default=5e-3)
    parser.add_argument("--dt-max", type=float, default=1e-3)
    parser.add_argument("--profile", choices=["default", "diagnostic-fast"], default="default")

    parser.add_argument(
        "--subsystem-sites",
        type=int,
        default=None,
        help="Entropy subsystem size L_A in particle-sector sites. If omitted, uses subsystem-fraction * L.",
    )
    parser.add_argument(
        "--subsystem-fraction",
        type=float,
        default=0.25,
        help="Default entropy subsystem fraction when --subsystem-sites is not provided.",
    )
    parser.add_argument("--subsystem-start", type=int, default=0)
    parser.add_argument(
        "--partial-checkpoint-every-chunks",
        type=int,
        default=1,
        help="Write partial per-point checkpoints every N completed chunks in multi-cpu mode (0 disables).",
    )

    parser.add_argument("--output-dir", default="results/fig2_reproduction")
    parser.add_argument("--tag", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--resume", action="store_true", help="Resume from existing per-point checkpoints if available.")
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Optional checkpoint directory. Default: <output-dir>/checkpoints_<tag>",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    args.t_multiplier, args.t_min, args.dt_ratio, args.dt_max = apply_profile_overrides(
        args.profile,
        t_multiplier=args.t_multiplier,
        t_min=args.t_min,
        dt_ratio=args.dt_ratio,
        dt_max=args.dt_max,
    )

    L_values = parse_int_list(args.L_values)
    gamma_values = parse_float_list(args.gamma_values)
    if not L_values or not gamma_values:
        raise ValueError("L-values and gamma-values must be non-empty.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / f"fig2_data_{args.tag}.csv"
    out_npz = output_dir / f"fig2_data_{args.tag}.npz"
    out_png = output_dir / f"fig2_plot_{args.tag}.png"
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else (output_dir / f"checkpoints_{args.tag}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    runmeta = {
        "created_at": datetime.now().isoformat(),
        "tag": str(args.tag),
        "output_dir": str(output_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "resume": bool(args.resume),
        "fingerprint": build_run_fingerprint(args, L_values, gamma_values),
        "L_values": [int(x) for x in L_values],
        "gamma_values": [float(x) for x in gamma_values],
        "n_trajectories": int(args.n_trajectories),
        "mode": str(args.mode),
        "closed_boundary": bool(args.closed_boundary),
        "J": float(args.J),
        "profile": str(args.profile),
        "subsystem_sites": None if args.subsystem_sites is None else int(args.subsystem_sites),
        "subsystem_fraction": float(args.subsystem_fraction),
        "subsystem_start": int(args.subsystem_start),
        "partial_checkpoint_every_chunks": int(args.partial_checkpoint_every_chunks),
    }
    runmeta_path = output_dir / f"fig2_runmeta_{args.tag}.json"
    runmeta_path.write_text(json.dumps(runmeta, indent=2, sort_keys=True), encoding="utf-8")

    point_results: list[PointResult] = []
    total = len(L_values) * len(gamma_values)
    done = 0
    resumed = 0
    computed = 0

    for gamma in gamma_values:
        for L in L_values:
            done += 1
            ckpt_file = checkpoint_file_for(checkpoint_dir, L=L, gamma=gamma)
            resolved_subsystem_sites = resolve_subsystem_sites(L, args.subsystem_sites, args.subsystem_fraction)
            if args.resume and ckpt_file.exists():
                try:
                    res = load_point_checkpoint(
                        ckpt_file,
                        expected_n_trajectories=args.n_trajectories,
                        expected_mode=args.mode,
                        expected_closed_boundary=args.closed_boundary,
                        expected_J=args.J,
                        expected_subsystem_sites=resolved_subsystem_sites,
                        expected_subsystem_start=args.subsystem_start,
                    )
                    resumed += 1
                    print(f"[{done}/{total}] Resumed L={L}, gamma={gamma:g} from {ckpt_file.name}")
                except Exception as exc:
                    print(f"[{done}/{total}] Checkpoint invalid for L={L}, gamma={gamma:g} ({exc}); recomputing")
                    res = run_point(
                        L=L,
                        gamma=gamma,
                        J=args.J,
                        n_trajectories=args.n_trajectories,
                        closed_boundary=args.closed_boundary,
                        t_multiplier=args.t_multiplier,
                        t_min=args.t_min,
                        dt_ratio=args.dt_ratio,
                        dt_max=args.dt_max,
                        subsystem_sites=resolved_subsystem_sites,
                        subsystem_start=args.subsystem_start,
                        mode=args.mode,
                        workers=args.workers,
                        master_seed=args.seed + 1000 * done,
                        checkpoint_dir=checkpoint_dir,
                        partial_checkpoint_every_chunks=args.partial_checkpoint_every_chunks,
                    )
                    save_point_checkpoint(
                        checkpoint_dir,
                        res,
                        n_trajectories=args.n_trajectories,
                        mode=args.mode,
                        closed_boundary=args.closed_boundary,
                        J=args.J,
                        subsystem_sites=resolved_subsystem_sites,
                        subsystem_start=args.subsystem_start,
                    )
                    computed += 1
            else:
                print(f"[{done}/{total}] Running L={L}, gamma={gamma:g} ...")
                res = run_point(
                    L=L,
                    gamma=gamma,
                    J=args.J,
                    n_trajectories=args.n_trajectories,
                    closed_boundary=args.closed_boundary,
                    t_multiplier=args.t_multiplier,
                    t_min=args.t_min,
                    dt_ratio=args.dt_ratio,
                    dt_max=args.dt_max,
                    subsystem_sites=resolved_subsystem_sites,
                    subsystem_start=args.subsystem_start,
                    mode=args.mode,
                    workers=args.workers,
                    master_seed=args.seed + 1000 * done,
                    checkpoint_dir=checkpoint_dir,
                    partial_checkpoint_every_chunks=args.partial_checkpoint_every_chunks,
                )
                save_point_checkpoint(
                    checkpoint_dir,
                    res,
                    n_trajectories=args.n_trajectories,
                    mode=args.mode,
                    closed_boundary=args.closed_boundary,
                    J=args.J,
                    subsystem_sites=resolved_subsystem_sites,
                    subsystem_start=args.subsystem_start,
                )
                computed += 1
            point_results.append(res)

    point_results.sort(key=lambda r: (float(r.gamma), int(r.L)))

    write_long_csv(
        out_csv,
        point_results,
        n_trajectories=args.n_trajectories,
        mode=args.mode,
        closed_boundary=args.closed_boundary,
        seed=args.seed,
    )
    write_npz(out_npz, point_results, L_values, gamma_values)
    plot_results(out_png, point_results, L_values, gamma_values)

    print("Done")
    print(f"Computed points: {computed}, resumed points: {resumed}")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"Run meta: {runmeta_path}")
    print(f"CSV: {out_csv}")
    print(f"NPZ: {out_npz}")
    print(f"PNG: {out_png}")


if __name__ == "__main__":
    main()
