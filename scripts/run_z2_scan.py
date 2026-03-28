#!/usr/bin/env python3
"""
Production Script: 1+<z^2> vs log(gamma/4) parameter sweep

This script runs a long parameter sweep for the L-qubit correlation simulator
and records the mean of z^2 over time and sites. Results are written
incrementally to CSV for safe resume and live monitoring.

Usage:
    python scripts/run_z2_scan.py

    # Resume an existing run
    python scripts/run_z2_scan.py --csv results/z2_scan/z2_data_20260214_123000.csv
"""

import argparse
import csv
import functools
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure local imports work when run as script
sys.path.insert(0, str(Path(__file__).parent.parent))
from quantum_measurement.jw_expansion.l_qubit_correlation_simulator import (
    LQubitCorrelationSimulator,
)
from quantum_measurement.parallel import ParameterSweepExecutor
from quantum_measurement.backends import MultiCpuBackend, MultiCpuBackendConfig

# ============================================================================
# Configuration Parameters
# ============================================================================

J_GLOBAL = 1.0

# Gamma grid
GAMMA_GLOBAL_MIN = 1e-1
GAMMA_GLOBAL_MAX = 1e2
N_GAMMA_GLOBAL = 80

G_CRITICAL_CENTER = 1.0
G_CRITICAL_WIDTH = 0.4
N_G_CRITICAL = 120

# Time-step construction
T_MULTIPLIER = 60.0
T_MIN = 100.0  # Increased from 10 for better convergence
DT_RATIO = 5e-3
DT_MAX = 1e-3
N_STEPS_WARNING = 1e8
DT_ABSOLUTE_MAX = 1e-2
DT_MIN = 1e-6
N_STEPS_MIN = 5_000
N_STEPS_MAX = 500_000

PHASE_CRITICAL_G_MIN = 0.9
PHASE_CRITICAL_G_MAX = 1.1
PHASE_STEPS_MULT_BEFORE = 0.75
PHASE_STEPS_MULT_CRITICAL = 1.30
PHASE_STEPS_MULT_AFTER = 0.60
PHASE_TIME_MULT_BEFORE = 1.00
PHASE_TIME_MULT_CRITICAL = 1.10
PHASE_TIME_MULT_AFTER = 0.85
NONCRITICAL_MAX_STEP_RATIO = 1.00
CRITICAL_MAX_STEP_RATIO = 1.15
ADAPTIVE_PROFILE_LEGACY = "legacy"
ADAPTIVE_PROFILE_NEXT_PASS_V1 = "next_pass_v1"
ADAPTIVE_PROFILE_OVERRIDES = {
    ADAPTIVE_PROFILE_NEXT_PASS_V1: {
        "t_multiplier": 0.95,
        "t_min": 4.0,
        "n_steps_min": 3_500,
        "phase_steps_mult_before": 0.70,
        "phase_steps_mult_critical": 1.12,
        "phase_steps_mult_after": 0.60,
        "phase_time_mult_before": 0.82,
        "phase_time_mult_critical": 1.00,
        "phase_time_mult_after": 0.82,
        "noncritical_max_step_ratio": 0.90,
        "critical_max_step_ratio": 1.08,
    },
}
RUN_META_VERSION = 1
PHASE_MAP_VERSION = 1
PHASE_MAP_SLOPE_THRESHOLD = 0.35
PHASE_MAP_MIN_POINTS = 9

# L values (existing list capped at 128, no fallback)
L_VALUES_BASE = [9, 17, 33, 65, 129, 256]
L_MAX = 129

# Output paths
RESULTS_DIR = Path(__file__).parent.parent / "results" / "z2_scan"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
DEFAULT_CSV = RESULTS_DIR / f"z2_data_{TIMESTAMP}.csv"

CSV_HEADER = [
    "timestamp",
    "L",
    "gamma",
    "g",
    "z2_mean",
    "z2_plus_one",
    "n_trajectories",
    "batch_size",
    "z2_std",
    "z2_stderr",
    "tau",
    "T",
    "dt",
    "N_steps",
    "epsilon",
    "runtime_sec",
    "nan_detected",
    "range_violation",
    "point_status",
]

NAN_MODES = ("fail_on_nan", "finish_full_sweep")


# ============================================================================
# Grid and parameter helpers
# ============================================================================

def construct_gamma_grid():
    """Build combined gamma grid: global log-spaced + critical region."""
    gamma_global = np.logspace(
        np.log10(GAMMA_GLOBAL_MIN), np.log10(GAMMA_GLOBAL_MAX), N_GAMMA_GLOBAL
    )
    g_critical = np.linspace(
        G_CRITICAL_CENTER - G_CRITICAL_WIDTH,
        G_CRITICAL_CENTER + G_CRITICAL_WIDTH,
        N_G_CRITICAL,
    )
    gamma_critical = g_critical * 4 * J_GLOBAL

    gamma_combined = np.unique(np.concatenate([gamma_global, gamma_critical]))
    gamma_combined = gamma_combined[gamma_combined > 0]
    return gamma_combined


def get_time_params(gamma):
    """Return (T, dt, N_steps, tau) based on gamma."""
    return get_time_params_with_config(
        gamma,
        t_multiplier=T_MULTIPLIER,
        t_min=T_MIN,
        dt_ratio=DT_RATIO,
        dt_max=DT_MAX,
    )


def classify_phase_region(gamma, critical_g_min=PHASE_CRITICAL_G_MIN, critical_g_max=PHASE_CRITICAL_G_MAX):
    """Classify point location relative to transition using g=gamma/(4J)."""
    g = float(gamma) / (4.0 * J_GLOBAL)
    if g < float(critical_g_min):
        return "before"
    if g > float(critical_g_max):
        return "after"
    return "critical"


def _phase_map_fingerprint(phase_map: Optional[dict]) -> Optional[str]:
    if phase_map is None:
        return None
    payload = json.dumps(phase_map, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_phase_map_artifact(phase_map_file: Union[Path, str]) -> dict:
    """Load and validate static phase-map artifact used for two-pass scheduling."""
    path = Path(phase_map_file)
    with open(path, "r", encoding="utf-8") as f:
        phase_map = json.load(f)

    required = {"phase_map_version", "critical_g_min", "critical_g_max", "method"}
    missing = sorted(required - set(phase_map.keys()))
    if missing:
        raise ValueError(f"Invalid phase-map artifact {path}: missing fields {missing}")

    cmin = float(phase_map["critical_g_min"])
    cmax = float(phase_map["critical_g_max"])
    if not np.isfinite(cmin) or not np.isfinite(cmax) or cmin <= 0.0 or cmax <= cmin:
        raise ValueError(
            f"Invalid phase-map critical bounds in {path}: critical_g_min={cmin}, critical_g_max={cmax}"
        )

    return phase_map


def _derive_phase_bounds_from_df(df: pd.DataFrame, slope_threshold_rel: float, min_points: int) -> Tuple[float, float, dict]:
    """Derive critical window from calibration data using |d(1+<z^2>)/dlog10(g)|."""
    if "g" not in df.columns or "z2_plus_one" not in df.columns:
        raise ValueError("Phase-map derivation requires columns: g, z2_plus_one")

    work = df[["g", "z2_plus_one"]].copy()
    work = work[np.isfinite(work["g"]) & np.isfinite(work["z2_plus_one"]) & (work["g"] > 0)]
    if len(work) < int(min_points):
        raise ValueError(
            f"Insufficient finite rows for phase-map derivation: {len(work)} < {int(min_points)}"
        )

    agg = work.groupby("g", as_index=False)["z2_plus_one"].median().sort_values("g")
    if len(agg) < int(min_points):
        raise ValueError(
            f"Insufficient unique g points for phase-map derivation: {len(agg)} < {int(min_points)}"
        )

    log_g = np.log10(agg["g"].to_numpy(dtype=float))
    z = agg["z2_plus_one"].to_numpy(dtype=float)
    slope = np.gradient(z, log_g)
    slope_abs = np.abs(slope)
    peak = float(np.max(slope_abs))
    if not np.isfinite(peak) or peak <= 0.0:
        raise ValueError("Unable to derive critical window: non-finite or zero slope peak")

    threshold = float(slope_threshold_rel) * peak
    mask = slope_abs >= threshold
    idx = np.where(mask)[0]
    if idx.size == 0:
        raise ValueError("Unable to derive critical window: empty slope mask")

    lo = max(0, int(idx[0]) - 1)
    hi = min(len(agg) - 1, int(idx[-1]) + 1)
    g_vals = agg["g"].to_numpy(dtype=float)
    critical_min = float(g_vals[lo])
    critical_max = float(g_vals[hi])
    if critical_max <= critical_min:
        center = float(g_vals[int(idx[0])])
        critical_min = 0.9 * center
        critical_max = 1.1 * center

    diagnostics = {
        "n_rows_used": int(len(work)),
        "n_unique_g": int(len(agg)),
        "slope_peak_abs": float(peak),
        "slope_threshold_abs": float(threshold),
    }
    return critical_min, critical_max, diagnostics


def build_phase_map_artifact(
    calibration_csv: Union[Path, str],
    output_path: Union[Path, str],
    *,
    slope_threshold_rel: float = PHASE_MAP_SLOPE_THRESHOLD,
    min_points: int = PHASE_MAP_MIN_POINTS,
) -> Path:
    """Build and persist phase-map artifact from calibration CSV (pass 1)."""
    source = Path(calibration_csv)
    out = Path(output_path)
    if not source.exists():
        raise FileNotFoundError(f"Calibration CSV not found: {source}")
    if not (0.0 < float(slope_threshold_rel) <= 1.0):
        raise ValueError("slope_threshold_rel must be in (0, 1]")
    if int(min_points) < 5:
        raise ValueError("min_points must be >= 5")

    df = pd.read_csv(source)
    critical_g_min, critical_g_max, diagnostics = _derive_phase_bounds_from_df(
        df,
        slope_threshold_rel=float(slope_threshold_rel),
        min_points=int(min_points),
    )

    artifact = {
        "phase_map_version": PHASE_MAP_VERSION,
        "created_at": datetime.now().isoformat(),
        "method": "median-z2-slope-on-log-g",
        "source_csv": str(source),
        "critical_g_min": float(critical_g_min),
        "critical_g_max": float(critical_g_max),
        "derivation": {
            "slope_threshold_rel": float(slope_threshold_rel),
            "min_points": int(min_points),
        },
        "diagnostics": diagnostics,
    }
    artifact["fingerprint"] = _phase_map_fingerprint(artifact)

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    return out


def _baseline_time_schedule(gamma, t_multiplier, t_min, dt_ratio, dt_max):
    tau = 1.0 / float(gamma)
    T = max(tau * float(t_multiplier), float(t_min))
    dt = min(tau * float(dt_ratio), float(dt_max))
    N_steps = int(np.ceil(T / dt))
    return T, dt, N_steps, tau


def build_time_schedule(
    gamma,
    t_multiplier,
    t_min,
    dt_ratio,
    dt_max,
    *,
    enable_phase_adaptation=False,
    phase_critical_g_min=PHASE_CRITICAL_G_MIN,
    phase_critical_g_max=PHASE_CRITICAL_G_MAX,
    phase_steps_mult_before=PHASE_STEPS_MULT_BEFORE,
    phase_steps_mult_critical=PHASE_STEPS_MULT_CRITICAL,
    phase_steps_mult_after=PHASE_STEPS_MULT_AFTER,
    phase_time_mult_before=PHASE_TIME_MULT_BEFORE,
    phase_time_mult_critical=PHASE_TIME_MULT_CRITICAL,
    phase_time_mult_after=PHASE_TIME_MULT_AFTER,
    noncritical_max_step_ratio=NONCRITICAL_MAX_STEP_RATIO,
    critical_max_step_ratio=CRITICAL_MAX_STEP_RATIO,
    n_steps_min=N_STEPS_MIN,
    n_steps_max=N_STEPS_MAX,
    dt_min=DT_MIN,
    phase_map=None,
):
    """Build deterministic time schedule with optional phase-aware bounded step budget."""
    if float(dt_max) > DT_ABSOLUTE_MAX:
        raise ValueError(f"dt_max={dt_max} exceeds absolute bound {DT_ABSOLUTE_MAX}")
    if int(n_steps_min) <= 0 or int(n_steps_max) <= 0 or int(n_steps_min) > int(n_steps_max):
        raise ValueError("Require 0 < n_steps_min <= n_steps_max")

    T_base, dt_base, N_base, tau = _baseline_time_schedule(
        gamma,
        t_multiplier=t_multiplier,
        t_min=t_min,
        dt_ratio=dt_ratio,
        dt_max=dt_max,
    )
    if phase_map is not None:
        phase_critical_g_min = float(phase_map["critical_g_min"])
        phase_critical_g_max = float(phase_map["critical_g_max"])

    phase_label = classify_phase_region(
        gamma,
        critical_g_min=phase_critical_g_min,
        critical_g_max=phase_critical_g_max,
    )

    if not enable_phase_adaptation:
        return {
            "T": float(T_base),
            "dt": float(dt_base),
            "N_steps": int(N_base),
            "tau": float(tau),
            "phase_label": phase_label,
            "adaptive_schedule": False,
            "phase_source": "phase_map" if phase_map is not None else "inline",
        }

    steps_mult = {
        "before": float(phase_steps_mult_before),
        "critical": float(phase_steps_mult_critical),
        "after": float(phase_steps_mult_after),
    }[phase_label]
    time_mult = {
        "before": float(phase_time_mult_before),
        "critical": float(phase_time_mult_critical),
        "after": float(phase_time_mult_after),
    }[phase_label]

    # Keep epsilon <= 0.5 by limiting dt based on gamma.
    epsilon_dt_cap = ((0.5 - 1e-12) ** 2) / float(gamma)
    dt_upper = min(float(dt_max), float(epsilon_dt_cap), DT_ABSOLUTE_MAX)
    if not np.isfinite(dt_upper) or dt_upper <= 0.0:
        raise ValueError(f"No feasible dt upper bound for gamma={gamma}: dt_upper={dt_upper}")

    T_target = max(float(T_base) * time_mult, float(dt_min) * int(n_steps_min))
    # Hard cap horizon so dt_max and n_steps_max are always jointly feasible.
    T_target = min(T_target, dt_upper * int(n_steps_max))

    n_target = int(np.ceil(float(N_base) * steps_mult))
    if phase_label == "critical":
        phase_step_cap = int(np.ceil(float(N_base) * float(critical_max_step_ratio)))
    else:
        phase_step_cap = int(np.ceil(float(N_base) * float(noncritical_max_step_ratio)))
    phase_step_cap = int(np.clip(phase_step_cap, int(n_steps_min), int(n_steps_max)))
    n_target = min(n_target, phase_step_cap)
    n_target = int(np.clip(n_target, int(n_steps_min), int(n_steps_max)))

    dt = float(T_target / n_target)
    dt = float(np.clip(dt, float(dt_min), dt_upper))
    N_steps = int(np.ceil(T_target / dt))

    if N_steps > phase_step_cap:
        N_steps = int(phase_step_cap)
        dt = float(np.clip(T_target / N_steps, float(dt_min), dt_upper))
        T_target = float(dt * N_steps)

    if N_steps > int(n_steps_max):
        N_steps = int(n_steps_max)
        dt = float(T_target / N_steps)

    if dt > dt_upper:
        dt = float(dt_upper)
        T_target = float(dt * N_steps)

    if N_steps < int(n_steps_min):
        N_steps = int(n_steps_min)
        dt = float(np.clip(T_target / N_steps, float(dt_min), dt_upper))
        T_target = float(dt * N_steps)

    return {
        "T": float(T_target),
        "dt": float(dt),
        "N_steps": int(N_steps),
        "tau": float(tau),
        "phase_label": phase_label,
        "adaptive_schedule": True,
        "phase_source": "phase_map" if phase_map is not None else "inline",
    }


def get_time_params_with_config(
    gamma,
    t_multiplier,
    t_min,
    dt_ratio,
    dt_max,
    **schedule_kwargs,
):
    """Return (T, dt, N_steps, tau) using explicit runtime parameters."""
    sched = build_time_schedule(
        gamma,
        t_multiplier=t_multiplier,
        t_min=t_min,
        dt_ratio=dt_ratio,
        dt_max=dt_max,
        **schedule_kwargs,
    )
    return sched["T"], sched["dt"], sched["N_steps"], sched["tau"]


def validate_time_grid(gamma_grid, t_multiplier, t_min, dt_ratio, dt_max, **schedule_kwargs):
    """Validate effective time-grid values before launching workers."""
    dt_vals = []
    epsilon_vals = []
    n_steps_vals = []
    invalid_reasons = []

    for gamma in gamma_grid:
        sched = build_time_schedule(
            gamma,
            t_multiplier=t_multiplier,
            t_min=t_min,
            dt_ratio=dt_ratio,
            dt_max=dt_max,
            **schedule_kwargs,
        )
        T = float(sched["T"])
        dt = float(sched["dt"])
        n_steps = int(sched["N_steps"])
        epsilon = float(np.sqrt(gamma * dt))
        dt_vals.append(float(dt))
        epsilon_vals.append(float(epsilon))
        n_steps_vals.append(int(n_steps))

        if not np.isfinite(dt) or dt <= 0.0 or dt > min(float(dt_max), DT_ABSOLUTE_MAX):
            invalid_reasons.append(f"gamma={gamma}: invalid dt={dt}")
        if not np.isfinite(epsilon) or epsilon <= 0.0 or epsilon > 0.5:
            invalid_reasons.append(f"gamma={gamma}: invalid epsilon={epsilon}")
        if not np.isfinite(T) or T <= 0.0 or n_steps <= 0:
            invalid_reasons.append(f"gamma={gamma}: invalid T/N_steps (T={T}, N_steps={n_steps})")
        if schedule_kwargs.get("enable_phase_adaptation", False) and n_steps > int(schedule_kwargs.get("n_steps_max", N_STEPS_MAX)):
            invalid_reasons.append(f"gamma={gamma}: N_steps exceeds cap ({n_steps})")

    print(
        "Time-grid preflight: "
        f"dt in [{min(dt_vals):.3e}, {max(dt_vals):.3e}], "
        f"epsilon in [{min(epsilon_vals):.3e}, {max(epsilon_vals):.3e}], "
        f"N_steps in [{min(n_steps_vals)}, {max(n_steps_vals)}]"
    )

    if invalid_reasons:
        preview = "; ".join(invalid_reasons[:6])
        if len(invalid_reasons) > 6:
            preview += f"; ... and {len(invalid_reasons) - 6} more"
        raise ValueError(
            "Time-grid preflight failed. Adjust --t-multiplier/--t-min/--dt-ratio/--dt-max. "
            + preview
        )


def resolve_L_values():
    """Filter L list to requested cap."""
    return [L for L in L_VALUES_BASE if L <= L_MAX]


def _metadata_path(csv_file: Path) -> Path:
    return csv_file.with_name(f"{csv_file.stem}_runmeta.json")


def _build_run_metadata(*, use_stable_integrator, enable_phase_adaptation, nan_mode, schedule_config):
    payload = {
        "run_meta_version": RUN_META_VERSION,
        "use_stable_integrator": bool(use_stable_integrator),
        "enable_phase_adaptation": bool(enable_phase_adaptation),
        "nan_mode": str(nan_mode),
        "schedule": dict(schedule_config),
    }
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["fingerprint"] = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
    return payload


def _ensure_resume_metadata(csv_file: Path, resume: bool, metadata: dict):
    meta_path = _metadata_path(csv_file)
    if csv_file.exists() and resume:
        if not meta_path.exists():
            raise ValueError(
                f"Resume blocked: metadata file missing for {csv_file}. "
                f"Expected {meta_path}. Use a fresh CSV or regenerate metadata intentionally."
            )
        with open(meta_path, "r", encoding="utf-8") as f:
            current = json.load(f)
        if current.get("fingerprint") != metadata.get("fingerprint"):
            raise ValueError(
                "Resume blocked: run configuration mismatch detected by metadata fingerprint. "
                f"existing={current.get('fingerprint')} new={metadata.get('fingerprint')}"
            )
        return

    if not csv_file.exists() or not resume:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)


def _apply_adaptive_profile(args):
    """Apply curated adaptive profile overrides to scheduling arguments."""
    profile = str(getattr(args, "adaptive_profile", ADAPTIVE_PROFILE_LEGACY))
    if profile == ADAPTIVE_PROFILE_LEGACY:
        return

    if profile not in ADAPTIVE_PROFILE_OVERRIDES:
        raise ValueError(f"Unknown adaptive profile: {profile}")

    if not bool(getattr(args, "enable_phase_adaptation", False)):
        print(
            f"Note: --adaptive-profile={profile} is ignored unless --enable-phase-adaptation is set."
        )
        return

    for key, value in ADAPTIVE_PROFILE_OVERRIDES[profile].items():
        setattr(args, key, value)

    print(f"Applied adaptive profile: {profile}")


# ============================================================================
# CSV management
# ============================================================================

def initialize_csv(csv_file):
    """Create CSV file with header."""
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
    print(f"Initialized CSV: {csv_file}")


def append_to_csv(csv_file, data_dict):
    """Append single row to CSV."""
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        row = [data_dict[key] for key in CSV_HEADER]
        writer.writerow(row)


def load_existing_results(csv_file):
    """Load existing CSV to avoid recomputing."""
    if not csv_file.exists():
        return set()

    completed = set()
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            L = int(row["L"])
            gamma = float(row["gamma"])
            completed.add((L, gamma))

    return completed


# ============================================================================
# Main sweep
# ============================================================================

def _run_single_point_with_config(
    L,
    gamma,
    backend_device,
    rng,
    n_trajectories_per_point,
    batch_size_per_point,
    compute_uncertainty,
    use_stable_integrator=False,
    enable_stability_monitor=False,
    t_multiplier=T_MULTIPLIER,
    t_min=T_MIN,
    dt_ratio=DT_RATIO,
    dt_max=DT_MAX,
    nan_mode="fail_on_nan",
    enable_phase_adaptation=False,
    phase_critical_g_min=PHASE_CRITICAL_G_MIN,
    phase_critical_g_max=PHASE_CRITICAL_G_MAX,
    phase_steps_mult_before=PHASE_STEPS_MULT_BEFORE,
    phase_steps_mult_critical=PHASE_STEPS_MULT_CRITICAL,
    phase_steps_mult_after=PHASE_STEPS_MULT_AFTER,
    phase_time_mult_before=PHASE_TIME_MULT_BEFORE,
    phase_time_mult_critical=PHASE_TIME_MULT_CRITICAL,
    phase_time_mult_after=PHASE_TIME_MULT_AFTER,
    noncritical_max_step_ratio=NONCRITICAL_MAX_STEP_RATIO,
    critical_max_step_ratio=CRITICAL_MAX_STEP_RATIO,
    n_steps_min=N_STEPS_MIN,
    n_steps_max=N_STEPS_MAX,
    dt_min=DT_MIN,
    phase_map=None,
):
    if nan_mode not in NAN_MODES:
        raise ValueError(f"Invalid nan_mode={nan_mode}. Expected one of {NAN_MODES}.")

    if enable_phase_adaptation and not use_stable_integrator:
        raise ValueError("Phase adaptation requires --use-stable-integrator for Stage 1 safety.")
    # Adaptive mode supports both fail-fast and full-sweep NaN policies.
    # fail_on_nan remains the default for production-quality runs, while
    # finish_full_sweep is allowed for stability-proving continuation scans.

    seed = int(rng.integers(0, np.iinfo(np.int32).max))
    g = gamma / (4 * J_GLOBAL)
    schedule = build_time_schedule(
        gamma,
        t_multiplier=t_multiplier,
        t_min=t_min,
        dt_ratio=dt_ratio,
        dt_max=dt_max,
        enable_phase_adaptation=enable_phase_adaptation,
        phase_critical_g_min=phase_critical_g_min,
        phase_critical_g_max=phase_critical_g_max,
        phase_steps_mult_before=phase_steps_mult_before,
        phase_steps_mult_critical=phase_steps_mult_critical,
        phase_steps_mult_after=phase_steps_mult_after,
        phase_time_mult_before=phase_time_mult_before,
        phase_time_mult_critical=phase_time_mult_critical,
        phase_time_mult_after=phase_time_mult_after,
        noncritical_max_step_ratio=noncritical_max_step_ratio,
        critical_max_step_ratio=critical_max_step_ratio,
        n_steps_min=n_steps_min,
        n_steps_max=n_steps_max,
        dt_min=dt_min,
        phase_map=phase_map,
    )
    T = float(schedule["T"])
    dt = float(schedule["dt"])
    N_steps = int(schedule["N_steps"])
    tau = float(schedule["tau"])
    phase_label = str(schedule["phase_label"])
    adaptive_schedule = bool(schedule["adaptive_schedule"])
    phase_source = str(schedule.get("phase_source", "inline"))

    epsilon = float(np.sqrt(gamma * dt))
    if not np.isfinite(dt) or dt <= 0.0 or dt > min(float(dt_max), DT_ABSOLUTE_MAX):
        raise ValueError(f"Invalid dt={dt} for L={L}, gamma={gamma}")
    if not np.isfinite(epsilon) or epsilon <= 0.0 or epsilon > 0.5:
        raise ValueError(f"Invalid epsilon={epsilon} for L={L}, gamma={gamma}")
    if enable_phase_adaptation and (N_steps < int(n_steps_min) or N_steps > int(n_steps_max)):
        raise ValueError(
            f"Invalid N_steps={N_steps} for L={L}, gamma={gamma}; expected [{n_steps_min}, {n_steps_max}]"
        )

    start_time = time.time()
    sim = LQubitCorrelationSimulator(
        L=L,
        J=J_GLOBAL,
        epsilon=epsilon,
        N_steps=N_steps,
        T=T,
        closed_boundary=True,
        device=backend_device,
        use_stable_integrator=bool(use_stable_integrator),
        enable_stability_monitor=bool(enable_stability_monitor),
        rng=np.random.default_rng(seed),
    )

    # Ensure backend RNG is reproducible for batched device-side sampling.
    if hasattr(sim.backend, "seed"):
        sim.backend.seed(seed)

    batch_size_used = int(batch_size_per_point) if batch_size_per_point is not None else None
    result = sim.simulate_z2_mean_ensemble(
        n_trajectories=int(n_trajectories_per_point),
        batch_size=batch_size_used,
        return_std_err=bool(compute_uncertainty),
    )

    if compute_uncertainty:
        z2_mean, z2_std, z2_stderr = result
    else:
        z2_mean = float(result)
        z2_std = float("nan")
        z2_stderr = float("nan")

    z2_plus_one = 1.0 + z2_mean
    nan_detected = not np.isfinite(z2_mean)
    range_violation = bool(
        np.isfinite(z2_mean) and ((z2_mean < -1e-9) or (z2_mean > (1.0 + 1e-6)))
    )
    if nan_detected:
        point_status = "nan"
    elif range_violation:
        point_status = "range_violation"
    else:
        point_status = "ok"
    if nan_detected and nan_mode == "fail_on_nan":
        raise ValueError(
            f"NaN detected for point L={L}, gamma={gamma}, seed={seed}, mode={nan_mode}."
        )

    runtime = time.time() - start_time

    return {
        "timestamp": datetime.now().isoformat(),
        "L": int(L),
        "gamma": float(gamma),
        "g": float(g),
        "z2_mean": float(z2_mean),
        "z2_plus_one": float(z2_plus_one),
        "n_trajectories": int(n_trajectories_per_point),
        "batch_size": batch_size_used,
        "z2_std": float(z2_std),
        "z2_stderr": float(z2_stderr),
        "tau": float(tau),
        "T": float(T),
        "dt": float(dt),
        "N_steps": int(N_steps),
        "epsilon": epsilon,
        "runtime_sec": float(runtime),
        "nan_detected": bool(nan_detected),
        "range_violation": bool(range_violation),
        "point_status": point_status,
        "phase_label": phase_label,
        "adaptive_schedule": adaptive_schedule,
        "phase_source": phase_source,
    }


def _build_point_runner(
    n_trajectories_per_point,
    batch_size_per_point,
    compute_uncertainty,
    use_stable_integrator=False,
    enable_stability_monitor=False,
    t_multiplier=T_MULTIPLIER,
    t_min=T_MIN,
    dt_ratio=DT_RATIO,
    dt_max=DT_MAX,
    nan_mode="fail_on_nan",
    enable_phase_adaptation=False,
    phase_critical_g_min=PHASE_CRITICAL_G_MIN,
    phase_critical_g_max=PHASE_CRITICAL_G_MAX,
    phase_steps_mult_before=PHASE_STEPS_MULT_BEFORE,
    phase_steps_mult_critical=PHASE_STEPS_MULT_CRITICAL,
    phase_steps_mult_after=PHASE_STEPS_MULT_AFTER,
    phase_time_mult_before=PHASE_TIME_MULT_BEFORE,
    phase_time_mult_critical=PHASE_TIME_MULT_CRITICAL,
    phase_time_mult_after=PHASE_TIME_MULT_AFTER,
    noncritical_max_step_ratio=NONCRITICAL_MAX_STEP_RATIO,
    critical_max_step_ratio=CRITICAL_MAX_STEP_RATIO,
    n_steps_min=N_STEPS_MIN,
    n_steps_max=N_STEPS_MAX,
    dt_min=DT_MIN,
    phase_map=None,
):
    return functools.partial(
        _run_single_point_with_config,
        n_trajectories_per_point=n_trajectories_per_point,
        batch_size_per_point=batch_size_per_point,
        compute_uncertainty=compute_uncertainty,
        use_stable_integrator=use_stable_integrator,
        enable_stability_monitor=enable_stability_monitor,
        t_multiplier=t_multiplier,
        t_min=t_min,
        dt_ratio=dt_ratio,
        dt_max=dt_max,
        nan_mode=nan_mode,
        enable_phase_adaptation=enable_phase_adaptation,
        phase_critical_g_min=phase_critical_g_min,
        phase_critical_g_max=phase_critical_g_max,
        phase_steps_mult_before=phase_steps_mult_before,
        phase_steps_mult_critical=phase_steps_mult_critical,
        phase_steps_mult_after=phase_steps_mult_after,
        phase_time_mult_before=phase_time_mult_before,
        phase_time_mult_critical=phase_time_mult_critical,
        phase_time_mult_after=phase_time_mult_after,
        noncritical_max_step_ratio=noncritical_max_step_ratio,
        critical_max_step_ratio=critical_max_step_ratio,
        n_steps_min=n_steps_min,
        n_steps_max=n_steps_max,
        dt_min=dt_min,
        phase_map=phase_map,
    )


def run_parameter_sweep(
    csv_file,
    backend_device="cpu",
    parallel_backend="sequential",
    executor_kind="parameter_sweep",
    n_workers=None,
    base_seed=42,
    resume=True,
    l_values_override=None,
    gamma_grid_override=None,
    n_trajectories_per_point=1,
    batch_size_per_point=None,
    compute_uncertainty=False,
    use_stable_integrator=False,
    enable_stability_monitor=False,
    t_multiplier=T_MULTIPLIER,
    t_min=T_MIN,
    dt_ratio=DT_RATIO,
    dt_max=DT_MAX,
    nan_mode="fail_on_nan",
    event_log_path=None,
    enable_phase_adaptation=False,
    phase_critical_g_min=PHASE_CRITICAL_G_MIN,
    phase_critical_g_max=PHASE_CRITICAL_G_MAX,
    phase_steps_mult_before=PHASE_STEPS_MULT_BEFORE,
    phase_steps_mult_critical=PHASE_STEPS_MULT_CRITICAL,
    phase_steps_mult_after=PHASE_STEPS_MULT_AFTER,
    phase_time_mult_before=PHASE_TIME_MULT_BEFORE,
    phase_time_mult_critical=PHASE_TIME_MULT_CRITICAL,
    phase_time_mult_after=PHASE_TIME_MULT_AFTER,
    noncritical_max_step_ratio=NONCRITICAL_MAX_STEP_RATIO,
    critical_max_step_ratio=CRITICAL_MAX_STEP_RATIO,
    n_steps_min=N_STEPS_MIN,
    n_steps_max=N_STEPS_MAX,
    dt_min=DT_MIN,
    phase_map_file=None,
):
    if nan_mode not in NAN_MODES:
        raise ValueError(f"Invalid nan_mode={nan_mode}. Expected one of {NAN_MODES}.")

    print("=" * 80)
    print("1+<z^2> PARAMETER SWEEP")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"CSV file: {csv_file}")
    print()

    L_values = sorted({int(v) for v in l_values_override}) if l_values_override is not None else resolve_L_values()
    gamma_grid = (
        np.array(sorted({float(v) for v in gamma_grid_override}), dtype=float)
        if gamma_grid_override is not None
        else construct_gamma_grid()
    )
    g_grid = gamma_grid / (4 * J_GLOBAL)

    print("Parameter grid:")
    print(f"  L values: {L_values}")
    print(f"  gamma range: [{gamma_grid.min():.3e}, {gamma_grid.max():.3e}]")
    print(f"  g range: [{g_grid.min():.3e}, {g_grid.max():.3e}]")
    print(f"  N_gamma points: {len(gamma_grid)}")
    print(f"  Total runs: {len(L_values) * len(gamma_grid)}")
    print(f"  trajectories per point: {int(n_trajectories_per_point)}")
    print(f"  batch size per point: {batch_size_per_point if batch_size_per_point is not None else 'auto'}")
    print(f"  compute uncertainty: {bool(compute_uncertainty)}")
    print(f"  use stable integrator: {bool(use_stable_integrator)}")
    print(f"  enable stability monitor: {bool(enable_stability_monitor)}")
    print(f"  nan mode: {nan_mode}")
    print(f"  enable phase adaptation: {bool(enable_phase_adaptation)}")

    phase_map = None
    if phase_map_file is not None:
        phase_map = load_phase_map_artifact(phase_map_file)
        phase_critical_g_min = float(phase_map["critical_g_min"])
        phase_critical_g_max = float(phase_map["critical_g_max"])
        print(f"  phase-map file: {Path(phase_map_file)}")
        print(
            "  phase-map critical window: "
            f"[{phase_critical_g_min:.6g}, {phase_critical_g_max:.6g}] "
            f"(method={phase_map.get('method')}, fingerprint={phase_map.get('fingerprint')})"
        )
    print(
        "  time-grid: "
        f"t_multiplier={float(t_multiplier):.6g}, "
        f"t_min={float(t_min):.6g}, "
        f"dt_ratio={float(dt_ratio):.6g}, "
        f"dt_max={float(dt_max):.6g}"
    )
    if enable_phase_adaptation:
        print(
            "  phase profile: "
            f"critical_g=[{float(phase_critical_g_min):.3g}, {float(phase_critical_g_max):.3g}], "
            f"steps_mult(before/critical/after)=({phase_steps_mult_before:.3g}/{phase_steps_mult_critical:.3g}/{phase_steps_mult_after:.3g}), "
            f"time_mult(before/critical/after)=({phase_time_mult_before:.3g}/{phase_time_mult_critical:.3g}/{phase_time_mult_after:.3g}), "
            f"max_step_ratio(noncritical/critical)=({noncritical_max_step_ratio:.3g}/{critical_max_step_ratio:.3g}), "
            f"N_steps=[{int(n_steps_min)}, {int(n_steps_max)}], dt_min={float(dt_min):.3g}"
        )

    if enable_phase_adaptation and not use_stable_integrator:
        raise ValueError("Phase adaptation requires --use-stable-integrator.")
    # Adaptive mode can run with either NaN policy. Operators must choose
    # explicitly and record mode in launch metadata.

    validate_time_grid(
        gamma_grid,
        t_multiplier=t_multiplier,
        t_min=t_min,
        dt_ratio=dt_ratio,
        dt_max=dt_max,
        enable_phase_adaptation=enable_phase_adaptation,
        phase_critical_g_min=phase_critical_g_min,
        phase_critical_g_max=phase_critical_g_max,
        phase_steps_mult_before=phase_steps_mult_before,
        phase_steps_mult_critical=phase_steps_mult_critical,
        phase_steps_mult_after=phase_steps_mult_after,
        phase_time_mult_before=phase_time_mult_before,
        phase_time_mult_critical=phase_time_mult_critical,
        phase_time_mult_after=phase_time_mult_after,
        noncritical_max_step_ratio=noncritical_max_step_ratio,
        critical_max_step_ratio=critical_max_step_ratio,
        n_steps_min=n_steps_min,
        n_steps_max=n_steps_max,
        dt_min=dt_min,
        phase_map=phase_map,
    )
    schedule_config = {
        "t_multiplier": float(t_multiplier),
        "t_min": float(t_min),
        "dt_ratio": float(dt_ratio),
        "dt_max": float(dt_max),
        "dt_min": float(dt_min),
        "enable_phase_adaptation": bool(enable_phase_adaptation),
        "phase_critical_g_min": float(phase_critical_g_min),
        "phase_critical_g_max": float(phase_critical_g_max),
        "phase_steps_mult_before": float(phase_steps_mult_before),
        "phase_steps_mult_critical": float(phase_steps_mult_critical),
        "phase_steps_mult_after": float(phase_steps_mult_after),
        "phase_time_mult_before": float(phase_time_mult_before),
        "phase_time_mult_critical": float(phase_time_mult_critical),
        "phase_time_mult_after": float(phase_time_mult_after),
        "noncritical_max_step_ratio": float(noncritical_max_step_ratio),
        "critical_max_step_ratio": float(critical_max_step_ratio),
        "n_steps_min": int(n_steps_min),
        "n_steps_max": int(n_steps_max),
        "phase_map_file": str(phase_map_file) if phase_map_file is not None else None,
        "phase_map_fingerprint": _phase_map_fingerprint(phase_map),
    }
    run_meta = _build_run_metadata(
        use_stable_integrator=bool(use_stable_integrator),
        enable_phase_adaptation=bool(enable_phase_adaptation),
        nan_mode=nan_mode,
        schedule_config=schedule_config,
    )
    _ensure_resume_metadata(csv_file=csv_file, resume=resume, metadata=run_meta)
    print()

    if resume and csv_file.exists():
        completed = load_existing_results(csv_file)
        print(f"Resuming: {len(completed)} runs already completed")
        print()

    if executor_kind == "multi_cpu":
        effective_event_log = (
            Path(event_log_path)
            if event_log_path is not None
            else csv_file.with_name(f"{csv_file.stem}_events.jsonl")
        )
        cfg = MultiCpuBackendConfig(
            max_workers=(n_workers if n_workers is not None else 38),
            master_seed=base_seed,
            nan_mode=nan_mode,
            log_path=effective_event_log,
        )
        executor = MultiCpuBackend(config=cfg)
        print(f"Executor: multi_cpu (workers={cfg.max_workers}, reserve_cores={cfg.reserve_cores})")
        print(f"Event log: {effective_event_log}")
    else:
        executor = ParameterSweepExecutor(
            parallel_backend=parallel_backend,
            n_workers=n_workers,
            base_seed=base_seed,
            verbose=True,
            continue_on_error=True,
        )
        print(f"Executor: parameter_sweep ({parallel_backend})")

    point_runner = _build_point_runner(
        n_trajectories_per_point=n_trajectories_per_point,
        batch_size_per_point=batch_size_per_point,
        compute_uncertainty=compute_uncertainty,
        use_stable_integrator=use_stable_integrator,
        enable_stability_monitor=enable_stability_monitor,
        t_multiplier=t_multiplier,
        t_min=t_min,
        dt_ratio=dt_ratio,
        dt_max=dt_max,
        nan_mode=nan_mode,
        enable_phase_adaptation=enable_phase_adaptation,
        phase_critical_g_min=phase_critical_g_min,
        phase_critical_g_max=phase_critical_g_max,
        phase_steps_mult_before=phase_steps_mult_before,
        phase_steps_mult_critical=phase_steps_mult_critical,
        phase_steps_mult_after=phase_steps_mult_after,
        phase_time_mult_before=phase_time_mult_before,
        phase_time_mult_critical=phase_time_mult_critical,
        phase_time_mult_after=phase_time_mult_after,
        noncritical_max_step_ratio=noncritical_max_step_ratio,
        critical_max_step_ratio=critical_max_step_ratio,
        n_steps_min=n_steps_min,
        n_steps_max=n_steps_max,
        dt_min=dt_min,
        phase_map=phase_map,
    )

    try:
        _ = executor.run_sweep(
            L_values=L_values,
            gamma_grid=gamma_grid,
            simulator_factory=point_runner,
            backend_device=backend_device,
            output_csv=csv_file,
            resume=resume,
            csv_header=CSV_HEADER,
        )
    except Exception as exc:
        print("=" * 80)
        print("SWEEP FAILED")
        print("=" * 80)
        print(f"Failure type: {type(exc).__name__}")
        print(f"Failure detail: {exc}")
        print(f"CSV target: {csv_file}")
        if executor_kind == "multi_cpu":
            print(f"Event log: {cfg.log_path}")
            print(f"Event counts: {executor.overflow_logger.counts()}")
        raise

    print("=" * 80)
    print("PARAMETER SWEEP COMPLETE")
    print("=" * 80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to: {csv_file}")
    if executor_kind == "multi_cpu":
        print(f"Event counts: {executor.overflow_logger.counts()}")
    print()


# ============================================================================
# Post-processing: Generate Plots
# ============================================================================

def generate_verification_plots(csv_file):
    """Generate plots from completed z2 scan data."""
    
    print("\nGenerating verification plots...")
    
    if not csv_file.exists():
        print("  ✗ No CSV file found, skipping plots")
        return
    
    df = pd.read_csv(csv_file)
    df = df.dropna(subset=["g", "z2_plus_one"])
    print(f"  ✓ Loaded {len(df)} data points from {csv_file.name}")
    
    if len(df) == 0:
        print("  ✗ No valid data points, skipping plots")
        return
    
    L_unique = sorted(df["L"].unique())
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(L_unique)))
    
    # Plot 1: Log scale on g (gamma/4)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for idx, L in enumerate(L_unique):
        df_L = df[df["L"] == L].sort_values("g")
        ax.plot(df_L["g"], df_L["z2_plus_one"], marker="o", markersize=4, 
                linewidth=1.2, color=colors[idx], label=f"L={L} ({len(df_L)} pts)")
    
    ax.axhline(1.25, color="orange", linestyle="--", linewidth=1.5, 
               label="1.25", alpha=0.7)
    ax.axhline(2.0, color="green", linestyle="--", linewidth=1.5, 
               label="2.0", alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("g = γ/(4J)", fontweight="bold")
    ax.set_ylabel("1+<z²>", fontweight="bold")
    ax.set_title("1+<z²> vs γ/4 (log scale)", fontweight="bold")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=9, ncol=2)
    plt.tight_layout()
    
    plot_file = csv_file.parent / f"{csv_file.stem}_logscale.png"
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved: {plot_file.name}")
    plt.close()
    
    # Plot 2: Linear x-axis with log10(gamma/4)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for idx, L in enumerate(L_unique):
        df_L = df[df["L"] == L].sort_values("g")
        log_g = np.log10(df_L["g"].values)
        ax.plot(log_g, df_L["z2_plus_one"], marker="o", markersize=4, 
                linewidth=1.2, color=colors[idx], label=f"L={L} ({len(df_L)} pts)")
    
    ax.axhline(1.25, color="orange", linestyle="--", linewidth=1.5, 
               label="1.25", alpha=0.7)
    ax.axhline(2.0, color="green", linestyle="--", linewidth=1.5, 
               label="2.0", alpha=0.7)
    ax.set_xlabel("log₁₀(γ/4)", fontweight="bold")
    ax.set_ylabel("1+<z²>", fontweight="bold")
    ax.set_title("1+<z²> vs log₁₀(γ/4)", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9, ncol=2)
    plt.tight_layout()
    
    plot_file = csv_file.parent / f"{csv_file.stem}_log10.png"
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved: {plot_file.name}")
    plt.close()
    
    # Plot 3: Runtime analysis
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for idx, L in enumerate(L_unique):
        df_L = df[df["L"] == L].sort_values("gamma")
        ax.scatter(df_L["gamma"], df_L["runtime_sec"], s=30, color=colors[idx],
                   alpha=0.6, label=f"L={L}")
    
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("γ", fontweight="bold")
    ax.set_ylabel("Runtime (s)", fontweight="bold")
    ax.set_title("Runtime vs γ", fontweight="bold")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    
    plot_file = csv_file.parent / f"{csv_file.stem}_runtime.png"
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved: {plot_file.name}")
    plt.close()
    
    print("  ✓ All plots generated")


# ============================================================================
# Entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Long-run scan for 1+<z^2> vs log(gamma/4)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to CSV file (resume if exists; create if missing)",
    )
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument(
        "--parallel-backend",
        choices=["sequential", "multiprocessing"],
        default="sequential",
    )
    parser.add_argument(
        "--executor",
        choices=["parameter_sweep", "multi_cpu"],
        default="parameter_sweep",
        help="Executor implementation to use. 'multi_cpu' enables strict core-affinity backend.",
    )
    parser.add_argument("--n-workers", type=int, default=None)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--l-values", nargs="+", type=int, default=None)
    parser.add_argument("--gamma-values", nargs="+", type=float, default=None)
    parser.add_argument("--n-trajectories-per-point", type=int, default=1)
    parser.add_argument("--batch-size-per-point", type=int, default=None)
    parser.add_argument("--compute-uncertainty", action="store_true")
    parser.add_argument("--use-stable-integrator", action="store_true")
    parser.add_argument("--enable-stability-monitor", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--t-multiplier", type=float, default=T_MULTIPLIER)
    parser.add_argument("--t-min", type=float, default=T_MIN)
    parser.add_argument("--dt-ratio", type=float, default=DT_RATIO)
    parser.add_argument("--dt-max", type=float, default=DT_MAX)
    parser.add_argument("--dt-min", type=float, default=DT_MIN)
    parser.add_argument("--n-steps-min", type=int, default=N_STEPS_MIN)
    parser.add_argument("--n-steps-max", type=int, default=N_STEPS_MAX)
    parser.add_argument("--enable-phase-adaptation", action="store_true")
    parser.add_argument(
        "--adaptive-profile",
        choices=[ADAPTIVE_PROFILE_LEGACY, ADAPTIVE_PROFILE_NEXT_PASS_V1],
        default=ADAPTIVE_PROFILE_LEGACY,
        help="Curated adaptive schedule profile. Applied only when --enable-phase-adaptation is set.",
    )
    parser.add_argument("--phase-critical-g-min", type=float, default=PHASE_CRITICAL_G_MIN)
    parser.add_argument("--phase-critical-g-max", type=float, default=PHASE_CRITICAL_G_MAX)
    parser.add_argument("--phase-steps-mult-before", type=float, default=PHASE_STEPS_MULT_BEFORE)
    parser.add_argument("--phase-steps-mult-critical", type=float, default=PHASE_STEPS_MULT_CRITICAL)
    parser.add_argument("--phase-steps-mult-after", type=float, default=PHASE_STEPS_MULT_AFTER)
    parser.add_argument("--phase-time-mult-before", type=float, default=PHASE_TIME_MULT_BEFORE)
    parser.add_argument("--phase-time-mult-critical", type=float, default=PHASE_TIME_MULT_CRITICAL)
    parser.add_argument("--phase-time-mult-after", type=float, default=PHASE_TIME_MULT_AFTER)
    parser.add_argument("--noncritical-max-step-ratio", type=float, default=NONCRITICAL_MAX_STEP_RATIO)
    parser.add_argument("--critical-max-step-ratio", type=float, default=CRITICAL_MAX_STEP_RATIO)
    parser.add_argument(
        "--phase-map-file",
        type=str,
        default=None,
        help="Static phase-map artifact JSON from calibration pass (two-pass mode).",
    )
    parser.add_argument(
        "--build-phase-map-from-csv",
        type=str,
        default=None,
        help="Build phase-map artifact from calibration CSV before running.",
    )
    parser.add_argument(
        "--phase-map-output",
        type=str,
        default=None,
        help="Output path for built phase-map JSON (default: <calibration_csv_stem>_phase_map.json).",
    )
    parser.add_argument(
        "--phase-map-slope-threshold",
        type=float,
        default=PHASE_MAP_SLOPE_THRESHOLD,
        help="Relative threshold on |d(1+<z^2>)/dlog10(g)| used to derive critical window.",
    )
    parser.add_argument(
        "--phase-map-min-points",
        type=int,
        default=PHASE_MAP_MIN_POINTS,
        help="Minimum finite points required to derive phase map.",
    )
    parser.add_argument(
        "--build-phase-map-only",
        action="store_true",
        help="Build phase-map artifact and exit without running sweep.",
    )
    parser.add_argument(
        "--event-log",
        type=str,
        default=None,
        help="Path to JSONL event log for multi_cpu executor (default: <csv_stem>_events.jsonl).",
    )
    parser.add_argument(
        "--nan-mode",
        choices=NAN_MODES,
        default="fail_on_nan",
        help="NaN policy: fail on first NaN point or finish full sweep with NaNs flagged.",
    )

    args = parser.parse_args()
    _apply_adaptive_profile(args)

    if args.t_multiplier <= 0 or args.t_min <= 0 or args.dt_ratio <= 0 or args.dt_max <= 0:
        raise ValueError("Time-grid overrides must be positive: --t-multiplier, --t-min, --dt-ratio, --dt-max")
    if args.dt_max > DT_ABSOLUTE_MAX:
        raise ValueError(f"--dt-max must be <= {DT_ABSOLUTE_MAX}")
    if args.dt_min <= 0:
        raise ValueError("--dt-min must be positive")
    if args.n_steps_min <= 0 or args.n_steps_max <= 0 or args.n_steps_min > args.n_steps_max:
        raise ValueError("Require 0 < --n-steps-min <= --n-steps-max")
    if args.phase_critical_g_min <= 0 or args.phase_critical_g_max <= 0:
        raise ValueError("Phase critical-g bounds must be positive")
    if args.phase_critical_g_min >= args.phase_critical_g_max:
        raise ValueError("Require --phase-critical-g-min < --phase-critical-g-max")
    if args.noncritical_max_step_ratio <= 0 or args.critical_max_step_ratio <= 0:
        raise ValueError("Step-ratio limits must be positive")
    if args.noncritical_max_step_ratio > args.critical_max_step_ratio:
        raise ValueError("Expected --noncritical-max-step-ratio <= --critical-max-step-ratio")

    if args.phase_map_file is not None and args.build_phase_map_from_csv is not None:
        raise ValueError("Use either --phase-map-file or --build-phase-map-from-csv, not both.")

    resolved_phase_map_file = args.phase_map_file
    if args.build_phase_map_from_csv is not None:
        source_csv = Path(args.build_phase_map_from_csv)
        if args.phase_map_output is not None:
            output_path = Path(args.phase_map_output)
        else:
            output_path = source_csv.with_name(f"{source_csv.stem}_phase_map.json")
        built = build_phase_map_artifact(
            calibration_csv=source_csv,
            output_path=output_path,
            slope_threshold_rel=args.phase_map_slope_threshold,
            min_points=args.phase_map_min_points,
        )
        print(f"Built phase-map artifact: {built}")
        resolved_phase_map_file = str(built)
        if args.build_phase_map_only:
            return

    csv_file = Path(args.csv) if args.csv else DEFAULT_CSV

    run_parameter_sweep(
        csv_file,
        backend_device=args.device,
        parallel_backend=args.parallel_backend,
        executor_kind=args.executor,
        n_workers=args.n_workers,
        base_seed=args.base_seed,
        resume=(not args.no_resume),
        l_values_override=args.l_values,
        gamma_grid_override=args.gamma_values,
        n_trajectories_per_point=args.n_trajectories_per_point,
        batch_size_per_point=args.batch_size_per_point,
        compute_uncertainty=args.compute_uncertainty,
        use_stable_integrator=args.use_stable_integrator,
        enable_stability_monitor=args.enable_stability_monitor,
        t_multiplier=args.t_multiplier,
        t_min=args.t_min,
        dt_ratio=args.dt_ratio,
        dt_max=args.dt_max,
        dt_min=args.dt_min,
        n_steps_min=args.n_steps_min,
        n_steps_max=args.n_steps_max,
        enable_phase_adaptation=args.enable_phase_adaptation,
        phase_critical_g_min=args.phase_critical_g_min,
        phase_critical_g_max=args.phase_critical_g_max,
        phase_steps_mult_before=args.phase_steps_mult_before,
        phase_steps_mult_critical=args.phase_steps_mult_critical,
        phase_steps_mult_after=args.phase_steps_mult_after,
        phase_time_mult_before=args.phase_time_mult_before,
        phase_time_mult_critical=args.phase_time_mult_critical,
        phase_time_mult_after=args.phase_time_mult_after,
        noncritical_max_step_ratio=args.noncritical_max_step_ratio,
        critical_max_step_ratio=args.critical_max_step_ratio,
        phase_map_file=resolved_phase_map_file,
        nan_mode=args.nan_mode,
        event_log_path=args.event_log,
    )
    if not args.skip_plots:
        generate_verification_plots(csv_file)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C)")
        sys.exit(1)
