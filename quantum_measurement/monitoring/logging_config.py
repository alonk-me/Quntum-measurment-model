"""Logging configuration and experiment monitoring utilities.

Functions
---------
configure_logging
    Configure the root ``quantum_measurement`` logger with a consistent format
    and optional file handler.

Classes
-------
ExperimentLogger
    Context manager that logs the start, end, and elapsed time of an
    experiment block.

Typical usage::

    from quantum_measurement.monitoring import configure_logging, ExperimentLogger

    configure_logging(level="INFO", log_file="logs/run.log")

    with ExperimentLogger("Kraus experiment") as el:
        result = run_krauss_experiment(cfg)
    print(f"Elapsed: {el.elapsed:.2f}s")
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Optional


_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    logger_name: str = "quantum_measurement",
) -> logging.Logger:
    """Configure the package logger.

    Sets up a :class:`logging.StreamHandler` writing to *stderr* and,
    optionally, a :class:`logging.FileHandler`.  Idempotent: calling this
    function multiple times with the same arguments will not add duplicate
    handlers.

    Parameters
    ----------
    level : str
        Logging level name (e.g. ``'DEBUG'``, ``'INFO'``, ``'WARNING'``).
    log_file : str or None
        Path of a file to write log output to.  The parent directory is
        created if it does not exist.  If *None*, only the stream handler
        is attached.
    logger_name : str
        Name of the logger to configure.  Defaults to ``'quantum_measurement'``
        which is the top-level package name.

    Returns
    -------
    logging.Logger
        The configured logger instance.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    pkg_logger = logging.getLogger(logger_name)
    pkg_logger.setLevel(log_level)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # Stream handler (avoid duplicates)
    if not any(isinstance(h, logging.StreamHandler) for h in pkg_logger.handlers):
        sh = logging.StreamHandler(sys.stderr)
        sh.setLevel(log_level)
        sh.setFormatter(formatter)
        pkg_logger.addHandler(sh)

    # File handler
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if not any(
            isinstance(h, logging.FileHandler)
            and getattr(h, "baseFilename", None) == str(log_path.resolve())
            for h in pkg_logger.handlers
        ):
            fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
            fh.setLevel(log_level)
            fh.setFormatter(formatter)
            pkg_logger.addHandler(fh)

    return pkg_logger


class ExperimentLogger:
    """Context manager that logs experiment start, end, and elapsed time.

    Parameters
    ----------
    name : str
        Human-readable name for the experiment block that will appear in log
        messages.
    logger : logging.Logger or None
        Logger to use.  If *None*, the ``quantum_measurement`` package logger
        is used.

    Attributes
    ----------
    elapsed : float
        Wall-clock time in seconds between ``__enter__`` and ``__exit__``.
        Only valid after the context has exited.

    Example
    -------
    ::

        with ExperimentLogger("my experiment") as el:
            do_work()
        print(el.elapsed)
    """

    def __init__(
        self,
        name: str,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.name = name
        self._logger = logger or logging.getLogger("quantum_measurement")
        self._start: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> "ExperimentLogger":
        self._start = time.monotonic()
        self._logger.info("=== %s — started ===", self.name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.elapsed = time.monotonic() - self._start
        if exc_type is None:
            self._logger.info(
                "=== %s — finished in %.2fs ===", self.name, self.elapsed
            )
        else:
            self._logger.error(
                "=== %s — failed after %.2fs: %s ===",
                self.name,
                self.elapsed,
                exc_val,
            )
        return False  # do not suppress exceptions
