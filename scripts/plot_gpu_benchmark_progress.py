#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_state(path: Path) -> dict[str, str]:
    state: dict[str, str] = {}
    if not path.exists():
        return state
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if "=" not in line:
                continue
            k, v = line.strip().split("=", 1)
            state[k] = v
    return state


def compute_eta(df: pd.DataFrame, expected_rows: int | None) -> tuple[float | None, float | None, int | None]:
    rows = len(df)
    if rows == 0 or "campaign_runtime_sec" not in df.columns:
        return None, None, expected_rows

    runtime_sec = float(df["campaign_runtime_sec"].iloc[-1])
    if runtime_sec <= 0:
        return runtime_sec, None, expected_rows

    avg_row_sec = runtime_sec / rows
    if expected_rows is None or expected_rows <= 0:
        return runtime_sec, None, None

    remaining = max(expected_rows - rows, 0)
    return runtime_sec, (remaining * avg_row_sec), expected_rows


def build_plot(df: pd.DataFrame, output: Path, expected_rows: int | None) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    ax_scatter = axes[0, 0]
    ax_runtime = axes[0, 1]
    ax_count = axes[1, 0]
    ax_status = axes[1, 1]

    rows = len(df)
    if rows == 0:
        ax_status.axis("off")
        ax_status.text(0.05, 0.8, "No benchmark rows yet", fontsize=12)
        for ax in [ax_scatter, ax_runtime, ax_count]:
            ax.axis("off")
        fig.suptitle("GPU Benchmark Live Progress")
        fig.tight_layout()
        fig.savefig(output, dpi=120)
        plt.close(fig)
        return

    # Throughput by batch size with device and L visibility.
    for (device, L), group in df.groupby(["device", "L"], dropna=False):
        group = group.sort_values("batch_size")
        label = f"{device}/L={int(L)}"
        ax_scatter.plot(
            group["batch_size"],
            group["throughput_traj_per_sec"],
            marker="o",
            linestyle="-",
            alpha=0.8,
            label=label,
        )
    ax_scatter.set_title("Throughput vs Batch Size")
    ax_scatter.set_xlabel("batch_size")
    ax_scatter.set_ylabel("traj/sec")
    ax_scatter.grid(True, alpha=0.2)
    ax_scatter.legend(fontsize=8)

    # Runtime trend across completed rows.
    runtime = df["campaign_runtime_sec"].astype(float)
    row_idx = list(range(1, rows + 1))
    ax_runtime.plot(row_idx, runtime, color="#1f77b4", linewidth=2)
    ax_runtime.set_title("Campaign Runtime Trend")
    ax_runtime.set_xlabel("completed rows")
    ax_runtime.set_ylabel("runtime_sec")
    ax_runtime.grid(True, alpha=0.2)

    # Device completion counts.
    counts = df["device"].value_counts().sort_index()
    ax_count.bar(counts.index.astype(str), counts.values, color=["#2ca02c", "#ff7f0e", "#1f77b4"][: len(counts)])
    ax_count.set_title("Completed Rows by Device")
    ax_count.set_xlabel("device")
    ax_count.set_ylabel("rows")
    ax_count.grid(True, axis="y", alpha=0.2)

    # Status panel.
    runtime_sec, eta_sec, expected = compute_eta(df, expected_rows)
    last = df.iloc[-1]
    ax_status.axis("off")
    lines = [
        "Live Status",
        f"rows: {rows}",
        f"last timestamp: {last.get('timestamp', 'n/a')}",
        "latest tuple: "
        f"{last.get('device', 'n/a')} / L={last.get('L', 'n/a')} / "
        f"steps={last.get('n_steps', 'n/a')} / batch={last.get('batch_size', 'n/a')}",
    ]

    if runtime_sec is not None and math.isfinite(runtime_sec):
        lines.append(f"elapsed_hours: {runtime_sec / 3600.0:.2f}")
    if expected is not None:
        lines.append(f"expected_rows: {expected}")
    if eta_sec is not None and math.isfinite(eta_sec):
        lines.append(f"eta_hours: {eta_sec / 3600.0:.2f}")
    else:
        lines.append("eta_hours: n/a")

    ax_status.text(0.02, 0.98, "\n".join(lines), va="top", fontsize=10)

    fig.suptitle("GPU Benchmark Live Progress", fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=120)
    plt.close(fig)


def resolve_csv(args: argparse.Namespace) -> tuple[Path | None, int | None]:
    if args.csv is not None:
        return args.csv, args.expected_rows

    state = parse_state(args.state)
    csv_path = state.get("CSV_PATH")
    expected_raw = state.get("EXPECTED_ROWS", "").strip()
    expected_rows = args.expected_rows
    if expected_rows is None and expected_raw.isdigit():
        expected_rows = int(expected_raw)

    if csv_path is None:
        return None, expected_rows
    return Path(csv_path), expected_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render live PNG progress for long GPU benchmark")
    parser.add_argument("--csv", type=Path, default=None, help="Benchmark CSV path. Defaults to state CSV")
    parser.add_argument("--state", type=Path, default=Path("logs/long_gpu_benchmark.state"))
    parser.add_argument("--output", type=Path, default=Path("results/test_scan/gpu_benchmark_live_progress.png"))
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--expected-rows", type=int, default=None)
    parser.add_argument("--once", action="store_true", help="Render once and exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    while True:
        csv_path, expected_rows = resolve_csv(args)
        if csv_path is None or (not csv_path.exists()) or csv_path.stat().st_size == 0:
            build_plot(pd.DataFrame(), args.output, expected_rows)
        else:
            try:
                df = pd.read_csv(csv_path)
                build_plot(df, args.output, expected_rows)
            except Exception as exc:  # pragma: no cover - runtime robustness
                # Keep monitor alive even when a partial CSV write occurs.
                print(f"plot refresh skipped: {exc}")

        if args.once:
            return
        time.sleep(max(args.interval, 1.0))


if __name__ == "__main__":
    main()
