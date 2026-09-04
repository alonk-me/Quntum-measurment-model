#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantum_measurement.experiments import ExperimentStore, list_recent_runs, load_run_points


def cmd_list_runs(db_path: Path, limit: int) -> None:
    rows = list_recent_runs(db_path, limit=limit)
    if not rows:
        print("No runs found.")
        return
    print("id | type      | status      | started_at                  | cores(req/act) | csv")
    print("-" * 110)
    for r in rows:
        req = r.get("requested_cores")
        act = r.get("actual_cores")
        print(
            f"{r['id']:>2} | {r['experiment_type']:<9} | {r['status']:<11} | {r['started_at']:<27} | "
            f"{str(req):>3}/{str(act):<3}      | {r.get('csv_path') or '-'}"
        )


def cmd_export_points(db_path: Path, run_id: int, output_csv: Path) -> None:
    rows = load_run_points(db_path, run_id=run_id)
    if not rows:
        raise SystemExit(f"No run points found for run_id={run_id}")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported {len(rows)} points to {output_csv}")


def cmd_plot_run(db_path: Path, run_id: int, out_dir: Path) -> None:
    conn = None
    try:
        import sqlite3

        conn = sqlite3.connect(db_path)
        kind = conn.execute("SELECT experiment_type FROM runs WHERE id=?", (int(run_id),)).fetchone()
        if kind is None:
            raise SystemExit(f"Unknown run_id={run_id}")
        exp_type = str(kind[0])
    finally:
        if conn is not None:
            conn.close()

    rows = load_run_points(db_path, run_id=run_id)
    if not rows:
        raise SystemExit(f"No run points found for run_id={run_id}")

    out_dir.mkdir(parents=True, exist_ok=True)

    if exp_type == "z2_scan":
        from scripts.run_z2_scan import generate_verification_plots

        csv_path = out_dir / f"run_{run_id}_z2_points.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        generate_verification_plots(csv_path)
        print(f"Generated z2 plots from run_id={run_id} using {csv_path}")
        return

    if exp_type == "ninf_scan":
        from scripts.run_ninf_scan import generate_verification_plots

        csv_path = out_dir / f"run_{run_id}_ninf_points.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        generate_verification_plots(csv_path)
        print(f"Generated ninf plots from run_id={run_id} using {csv_path}")
        return

    raise SystemExit(f"Plotting not configured for experiment_type={exp_type}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment DB utilities")
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Experiment DB path (default resolved by ExperimentStore).",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list-runs", help="List recent runs")
    p_list.add_argument("--limit", type=int, default=20)

    p_export = sub.add_parser("export-points", help="Export run points to CSV")
    p_export.add_argument("--run-id", type=int, required=True)
    p_export.add_argument("--output-csv", type=str, required=True)

    p_plot = sub.add_parser("plot-run", help="Generate plots for a run")
    p_plot.add_argument("--run-id", type=int, required=True)
    p_plot.add_argument("--out-dir", type=str, default="results/db_exports")

    args = parser.parse_args()

    db_path = ExperimentStore.resolve_db_path(args.db)

    if args.command == "list-runs":
        cmd_list_runs(db_path, limit=args.limit)
    elif args.command == "export-points":
        cmd_export_points(db_path, run_id=args.run_id, output_csv=Path(args.output_csv).resolve())
    elif args.command == "plot-run":
        cmd_plot_run(db_path, run_id=args.run_id, out_dir=Path(args.out_dir).resolve())


if __name__ == "__main__":
    main()
