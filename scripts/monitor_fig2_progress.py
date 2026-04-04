#!/usr/bin/env python3
from __future__ import annotations

"""Live monitor for Figure-2 validation/reproduction campaigns.

Monitors:
- run metadata (runmeta json)
- per-point checkpoint files
- process liveness (best effort, by tag)

Outputs:
- terminal status summary
- rolling progress plot for completed points
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RESULTS_DIR = Path(__file__).parent.parent / "results" / "fig2_reproduction"
DEFAULT_INTERVAL = 30
DEFAULT_STALE_SECONDS = 15 * 60


def find_runmeta_file(output_dir: Path, tag: str | None) -> Path:
    if tag:
        path = output_dir / f"fig2_runmeta_{tag}.json"
        if not path.exists():
            raise FileNotFoundError(f"Run meta file not found: {path}")
        return path

    candidates = sorted(output_dir.glob("fig2_runmeta_*.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No run meta files found in {output_dir}")
    return candidates[-1]


def load_runmeta(runmeta_file: Path) -> dict[str, Any]:
    return json.loads(runmeta_file.read_text(encoding="utf-8"))


def expected_total_points(runmeta: dict[str, Any]) -> int:
    return len(runmeta.get("L_values", [])) * len(runmeta.get("gamma_values", []))


def list_checkpoint_files(checkpoint_dir: Path) -> list[Path]:
    if not checkpoint_dir.exists():
        return []
    files = sorted(checkpoint_dir.glob("L*_g*.npz"), key=lambda p: p.name)
    return [p for p in files if not p.name.endswith(".partial.npz")]


def list_partial_checkpoint_files(checkpoint_dir: Path) -> list[Path]:
    if not checkpoint_dir.exists():
        return []
    return sorted(checkpoint_dir.glob("L*_g*.partial.npz"), key=lambda p: p.name)


def is_run_process_alive(tag: str) -> bool:
    pattern = f"reproduce_turkeshi_fig2.py"  # broad match, filtered by tag token
    try:
        out = subprocess.run(["pgrep", "-af", pattern], check=False, capture_output=True, text=True)
    except Exception:
        return False
    if out.returncode not in (0, 1):
        return False
    lines = [ln for ln in out.stdout.splitlines() if ln.strip()]
    tag_token = f"--tag {tag}"
    return any(tag_token in ln for ln in lines)


def find_run_pid(tag: str) -> int | None:
    pattern = "reproduce_turkeshi_fig2.py"
    try:
        out = subprocess.run(["pgrep", "-af", pattern], check=False, capture_output=True, text=True)
    except Exception:
        return None
    if out.returncode not in (0, 1):
        return None

    tag_token = f"--tag {tag}"
    for line in out.stdout.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) < 2:
            continue
        if tag_token in parts[1]:
            try:
                return int(parts[0])
            except ValueError:
                continue
    return None


def get_run_cpu_snapshot(run_pid: int | None) -> dict[str, Any]:
    if run_pid is None:
        return {"run_pid": None, "descendant_count": 0, "cpu_total": 0.0, "busy_descendants": 0}

    try:
        out = subprocess.run(["ps", "-eo", "pid,ppid,pcpu,cmd"], check=True, capture_output=True, text=True)
    except Exception:
        return {"run_pid": run_pid, "descendant_count": 0, "cpu_total": 0.0, "busy_descendants": 0}

    rows: list[tuple[int, int, float]] = []
    for line in out.stdout.splitlines()[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=3)
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
            pcpu = float(parts[2])
        except ValueError:
            continue
        rows.append((pid, ppid, pcpu))

    children: dict[int, list[int]] = {}
    cpu_by_pid: dict[int, float] = {}
    for pid, ppid, pcpu in rows:
        cpu_by_pid[pid] = pcpu
        children.setdefault(ppid, []).append(pid)

    descendants: list[int] = []
    stack = [run_pid]
    seen: set[int] = set()
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        for ch in children.get(current, []):
            descendants.append(ch)
            stack.append(ch)

    cpu_total = cpu_by_pid.get(run_pid, 0.0) + sum(cpu_by_pid.get(pid, 0.0) for pid in descendants)
    busy_desc = sum(1 for pid in descendants if cpu_by_pid.get(pid, 0.0) >= 1.0)
    return {
        "run_pid": run_pid,
        "descendant_count": len(descendants),
        "cpu_total": float(cpu_total),
        "busy_descendants": int(busy_desc),
    }


def load_checkpoint_point(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        gamma = float(data["gamma"])
        L = int(data["L"])
        t = np.asarray(data["t_grid"], dtype=float)
        s_mean = np.asarray(data["s_mean"], dtype=float)
        s_std = np.asarray(data["s_std"], dtype=float)
    return {
        "path": path,
        "gamma": gamma,
        "L": L,
        "t": t,
        "s_mean": s_mean,
        "s_std": s_std,
        "mtime": path.stat().st_mtime,
    }


def load_partial_checkpoint_progress(path: Path) -> dict[str, Any] | None:
    try:
        with np.load(path, allow_pickle=False) as data:
            n_completed = int(data["n_completed"])
            n_total = int(data["n_total"])
            gamma = float(data["gamma"])
            L = int(data["L"])
    except Exception:
        return None

    ratio = 0.0
    if n_total > 0:
        ratio = max(0.0, min(1.0, n_completed / n_total))
    return {
        "path": path,
        "L": L,
        "gamma": gamma,
        "n_completed": n_completed,
        "n_total": n_total,
        "ratio": ratio,
        "mtime": path.stat().st_mtime,
    }


def build_progress_snapshot(
    runmeta: dict[str, Any],
    checkpoint_files: list[Path],
    partial_checkpoint_files: list[Path] | None = None,
    *,
    stale_seconds: float,
    process_alive: bool,
    now_ts: float | None = None,
) -> dict[str, Any]:
    now = now_ts if now_ts is not None else time.time()
    total = expected_total_points(runmeta)
    completed = len(checkpoint_files)
    partial_files = partial_checkpoint_files if partial_checkpoint_files is not None else []

    started_ts = None
    try:
        started_ts = datetime.fromisoformat(runmeta.get("created_at", "")).timestamp()
    except Exception:
        started_ts = None

    latest_checkpoint_age = None
    mtimes = [p.stat().st_mtime for p in checkpoint_files]
    mtimes.extend(p.stat().st_mtime for p in partial_files)
    if mtimes:
        latest_mtime = max(mtimes)
        latest_checkpoint_age = max(0.0, now - latest_mtime)

    partial_progress = 0.0
    if partial_files:
        ratios = []
        for p in partial_files:
            info = load_partial_checkpoint_progress(p)
            if info is not None:
                ratios.append(float(info["ratio"]))
        if ratios:
            partial_progress = max(ratios)

    elapsed = max(0.0, now - started_ts) if started_ts is not None else None

    if completed >= total and total > 0:
        state = "completed"
    elif completed == 0 and process_alive:
        if partial_progress > 0.0:
            state = "intra_point_progress"
        elif elapsed is not None and elapsed > stale_seconds:
            state = "waiting_first_checkpoint_late"
        else:
            state = "waiting_first_checkpoint"
    elif process_alive and latest_checkpoint_age is not None and latest_checkpoint_age > stale_seconds:
        state = "stalled"
    elif process_alive:
        state = "running"
    else:
        state = "stopped"

    return {
        "state": state,
        "completed": completed,
        "total": total,
        "progress": (completed / total) if total > 0 else 0.0,
        "partial_progress": float(partial_progress),
        "elapsed_sec": elapsed,
        "latest_checkpoint_age_sec": latest_checkpoint_age,
        "process_alive": process_alive,
    }


def format_seconds(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value < 60:
        return f"{value:.0f}s"
    if value < 3600:
        return f"{value/60:.1f}m"
    return f"{value/3600:.1f}h"


def plot_progress(
    output_plot: Path,
    runmeta: dict[str, Any],
    points: list[dict[str, Any]],
    snapshot: dict[str, Any],
) -> None:
    output_plot.parent.mkdir(parents=True, exist_ok=True)

    gamma_values = [float(x) for x in runmeta.get("gamma_values", [])]
    L_values = [int(x) for x in runmeta.get("L_values", [])]

    if not gamma_values:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        ax.text(0.5, 0.5, "No gamma values in runmeta", ha="center", va="center")
        fig.savefig(output_plot, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    n_gamma = len(gamma_values)
    n_cols = min(3, n_gamma)
    n_rows = int(np.ceil(n_gamma / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 3.4 * n_rows), squeeze=False)

    color_map = {L: c for L, c in zip(L_values, plt.cm.viridis(np.linspace(0.15, 0.9, max(1, len(L_values))))) }

    for idx, gamma in enumerate(gamma_values):
        ax = axes[idx // n_cols][idx % n_cols]
        gamma_points = [p for p in points if np.isclose(p["gamma"], gamma)]
        gamma_points.sort(key=lambda p: p["L"])

        for p in gamma_points:
            color = color_map.get(p["L"], "tab:blue")
            ax.plot(p["t"], p["s_mean"], color=color, linewidth=1.5, label=f"L={p['L']}")
            ax.fill_between(p["t"], p["s_mean"] - p["s_std"], p["s_mean"] + p["s_std"], color=color, alpha=0.16)

        ax.set_xscale("log")
        ax.set_title(f"gamma={gamma:g}")
        ax.set_xlabel("t")
        ax.set_ylabel("S_A(t)")
        ax.grid(True, alpha=0.3)

    for idx in range(n_gamma, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(6, len(labels)), frameon=False)

    fig.suptitle(
        f"Figure-2 Progress | state={snapshot['state']} | "
        f"completed={snapshot['completed']}/{snapshot['total']} | "
        f"updated={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_plot, dpi=170)
    plt.close(fig)


def print_snapshot_line(snapshot: dict[str, Any], runmeta_file: Path, checkpoint_dir: Path) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    partial_ratio = float(snapshot.get("partial_progress", 0.0))
    print(
        f"[{ts}] state={snapshot['state']} progress={snapshot['completed']}/{snapshot['total']} "
        f"({snapshot['progress']*100:.1f}%) partial={partial_ratio*100:.1f}% "
        f"age={format_seconds(snapshot['latest_checkpoint_age_sec'])} "
        f"elapsed={format_seconds(snapshot['elapsed_sec'])} "
        f"alive={snapshot['process_alive']} "
        f"cpu={snapshot.get('cpu_total', 0.0):.1f}% "
        f"workers={snapshot.get('busy_descendants', 0)}/{snapshot.get('descendant_count', 0)}"
    )
    if snapshot["state"] in {"waiting_first_checkpoint_late", "stalled"}:
        print("  ALERT: checkpoint freshness SLA breached.")
        print(f"  runmeta: {runmeta_file}")
        print(f"  checkpoints: {checkpoint_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monitor Figure-2 validation progress from checkpoints.")
    parser.add_argument("--output-dir", default=str(DEFAULT_RESULTS_DIR), help="Results directory containing runmeta/checkpoints.")
    parser.add_argument("--tag", default=None, help="Run tag. If omitted, latest runmeta is used.")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL, help="Refresh interval seconds.")
    parser.add_argument("--stale-seconds", type=int, default=DEFAULT_STALE_SECONDS, help="Freshness SLA seconds.")
    parser.add_argument("--plot-file", default=None, help="Override progress plot path.")
    parser.add_argument("--once", action="store_true", help="Run one monitoring cycle and exit.")
    return parser


def monitor_loop(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    runmeta_file = find_runmeta_file(output_dir, args.tag)
    runmeta = load_runmeta(runmeta_file)

    tag = str(runmeta.get("tag") or args.tag or "unknown")
    checkpoint_dir = Path(runmeta.get("checkpoint_dir", output_dir / f"checkpoints_{tag}"))
    plot_file = Path(args.plot_file) if args.plot_file else (output_dir / f"fig2_progress_{tag}.png")

    print("=" * 80)
    print("FIG2 PROGRESS MONITOR")
    print("=" * 80)
    print(f"runmeta: {runmeta_file}")
    print(f"checkpoints: {checkpoint_dir}")
    print(f"plot: {plot_file}")
    print(f"interval: {args.interval}s, stale SLA: {args.stale_seconds}s")
    print("=" * 80)

    while True:
        checkpoints = list_checkpoint_files(checkpoint_dir)
        partial_checkpoints = list_partial_checkpoint_files(checkpoint_dir)
        points = []
        for path in checkpoints:
            try:
                points.append(load_checkpoint_point(path))
            except Exception:
                continue

        process_alive = is_run_process_alive(tag)
        run_pid = find_run_pid(tag)
        cpu_snapshot = get_run_cpu_snapshot(run_pid)
        snapshot = build_progress_snapshot(
            runmeta,
            checkpoints,
            partial_checkpoints,
            stale_seconds=float(args.stale_seconds),
            process_alive=process_alive,
        )
        snapshot.update(cpu_snapshot)

        plot_progress(plot_file, runmeta, points, snapshot)
        print_snapshot_line(snapshot, runmeta_file, checkpoint_dir)

        if args.once:
            break
        time.sleep(max(1, int(args.interval)))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    monitor_loop(args)


if __name__ == "__main__":
    main()
