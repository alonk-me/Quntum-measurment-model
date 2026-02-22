try:
    from .rust_simulator import FastLQubitSimulator
except ImportError:
    FastLQubitSimulator = None  # type: ignore[assignment,misc]
