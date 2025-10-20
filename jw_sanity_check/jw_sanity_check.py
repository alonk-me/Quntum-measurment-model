"""Compatibility shim for jw_sanity_check (renamed to jw_H_XY).

This module has been renamed. Please import from jw_H_XY instead.
All public functions are re-exported here for backward compatibility.
"""

from __future__ import annotations

import warnings as _warnings

_warnings.warn(
    "jw_sanity_check has been renamed to jw_H_XY; please update your imports.",
    category=DeprecationWarning,
    stacklevel=2,
)

# Re-export public API from the new module name
# Use absolute import to avoid relying on package-relative imports
from jw_H_XY import (  # type: ignore[F401]
    wavefunction_magnetization,
    corr_magnetization,
    simulate_wavefunction_parallel,
)

