"""Shared fixtures for all quantum measurement tests."""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest

# Ensure subpackage directories are on the path for direct imports
_ROOT = Path(__file__).parent.parent / "quantum_measurement"
for _sub in ("jw_expansion", "sse_simulation", "krauss_operators",
             "free_fermion", "jw_sanity_check"):
    _p = str(_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)


@pytest.fixture(autouse=True)
def matplotlib_backend():
    """Use non-interactive matplotlib backend to avoid display errors."""
    matplotlib.use("Agg")


@pytest.fixture
def fixed_rng():
    """Seeded random number generator for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def minimal_params_sse():
    """Minimal SSE simulator parameters for quick tests."""
    return {
        "epsilon": 0.1,
        "N_steps": 100,
        "J": 0.0,
        "initial_state": "bloch_equator",
        "rng": np.random.default_rng(42),
    }


@pytest.fixture
def minimal_params_correlation():
    """Minimal L-qubit correlation simulator parameters for quick tests."""
    return {
        "L": 2,
        "J": 1.0,
        "epsilon": 0.1,
        "N_steps": 100,
        "T": 1.0,
        "closed_boundary": False,
        "rng": np.random.default_rng(42),
    }
