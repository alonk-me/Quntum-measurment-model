from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np


SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "monitor_fig2_progress.py"
SPEC = importlib.util.spec_from_file_location("monitor_fig2_progress", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

build_progress_snapshot = MODULE.build_progress_snapshot
expected_total_points = MODULE.expected_total_points
find_runmeta_file = MODULE.find_runmeta_file
list_checkpoint_files = MODULE.list_checkpoint_files
list_partial_checkpoint_files = MODULE.list_partial_checkpoint_files


def _write_runmeta(path: Path, *, tag: str = "t1") -> None:
    payload = {
        "created_at": "2026-04-04T00:00:00",
        "tag": tag,
        "L_values": [32, 64],
        "gamma_values": [2.0, 4.0, 8.0],
        "checkpoint_dir": str(path.parent / f"checkpoints_{tag}"),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_ckpt(path: Path) -> None:
    t = np.array([0.0, 0.1])
    s = np.array([0.0, 0.2])
    np.savez_compressed(
        path,
        L=np.asarray(32),
        gamma=np.asarray(2.0),
        dt=np.asarray(0.1),
        n_steps=np.asarray(1),
        t_grid=t,
        s_mean=s,
        s_std=np.array([0.0, 0.01]),
        n_trajectories=np.asarray(2),
        mode=np.asarray("serial"),
        closed_boundary=np.asarray(0),
        J=np.asarray(1.0),
    )


def _write_partial_ckpt(path: Path) -> None:
    t = np.array([0.0, 0.1])
    s = np.array([0.0, 0.2])
    np.savez_compressed(
        path,
        L=np.asarray(32),
        gamma=np.asarray(2.0),
        dt=np.asarray(0.1),
        n_steps=np.asarray(1),
        t_grid=t,
        s_mean_partial=s,
        s_std_partial=np.array([0.0, 0.01]),
        n_completed=np.asarray(1),
        n_total=np.asarray(4),
    )


def test_find_runmeta_by_tag(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    rm1 = out / "fig2_runmeta_a.json"
    rm2 = out / "fig2_runmeta_b.json"
    _write_runmeta(rm1, tag="a")
    time.sleep(0.01)
    _write_runmeta(rm2, tag="b")

    found_a = find_runmeta_file(out, "a")
    found_latest = find_runmeta_file(out, None)
    assert found_a.name == "fig2_runmeta_a.json"
    assert found_latest.name == "fig2_runmeta_b.json"


def test_snapshot_waiting_first_checkpoint_late(tmp_path: Path) -> None:
    now = 1_000.0
    runmeta = {
        "created_at": "1970-01-01T00:00:00",
        "L_values": [32, 64],
        "gamma_values": [2.0, 4.0, 8.0],
    }
    snap = build_progress_snapshot(
        runmeta,
        checkpoint_files=[],
        stale_seconds=10,
        process_alive=True,
        now_ts=now,
    )
    assert snap["state"] == "waiting_first_checkpoint_late"
    assert snap["completed"] == 0
    assert snap["total"] == 6


def test_snapshot_completed(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    runmeta = {
        "created_at": "2026-04-04T00:00:00",
        "L_values": [32],
        "gamma_values": [2.0],
    }
    ckpt = ckpt_dir / "L32_g2.npz"
    _write_ckpt(ckpt)
    checkpoints = list_checkpoint_files(ckpt_dir)
    assert len(checkpoints) == 1
    assert expected_total_points(runmeta) == 1

    snap = build_progress_snapshot(
        runmeta,
        checkpoint_files=checkpoints,
        stale_seconds=60,
        process_alive=False,
    )
    assert snap["state"] == "completed"
    assert np.isclose(snap["progress"], 1.0)


def test_list_checkpoint_filters_partial(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    full = ckpt_dir / "L32_g2.npz"
    partial = ckpt_dir / "L32_g2.partial.npz"
    _write_ckpt(full)
    _write_partial_ckpt(partial)

    full_files = list_checkpoint_files(ckpt_dir)
    partial_files = list_partial_checkpoint_files(ckpt_dir)
    assert [p.name for p in full_files] == ["L32_g2.npz"]
    assert [p.name for p in partial_files] == ["L32_g2.partial.npz"]


def test_snapshot_intra_point_progress(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    partial = ckpt_dir / "L32_g2.partial.npz"
    _write_partial_ckpt(partial)

    runmeta = {
        "created_at": "2026-04-04T00:00:00",
        "L_values": [32],
        "gamma_values": [2.0],
    }
    snap = build_progress_snapshot(
        runmeta,
        checkpoint_files=[],
        partial_checkpoint_files=[partial],
        stale_seconds=600,
        process_alive=True,
        now_ts=10_000.0,
    )
    assert snap["state"] == "intra_point_progress"
    assert snap["partial_progress"] > 0.0
