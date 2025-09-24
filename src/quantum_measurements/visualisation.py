"""Visualisation helpers for entropy production data."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np

from .theory import dressel_eq14_pdf


def plot_histogram_with_theory(
    Q_values: Iterable[float],
    epsilon: float,
    steps: int,
    filename: Optional[str | Path] = None,
    bins: int = 60,
) -> None:
    """Plot a histogram of entropy production with the theoretical curve."""

    Q_values = np.asarray(Q_values, dtype=float)
    theta = 2.0 * steps * epsilon

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.hist(Q_values, bins=bins, density=True, alpha=0.6, color="#4682b4", edgecolor="black")

    xs = np.linspace(Q_values.min(), Q_values.max(), 400)
    ax.plot(xs, dressel_eq14_pdf(xs, theta), label=f"Eq. 14 (θ={theta:.2f})", linewidth=2.0)

    ax.set_xlabel("Entropy production Q")
    ax.set_ylabel("Probability density")
    ax.set_title("Entropy production distribution")
    ax.legend()
    fig.tight_layout()

    if filename is not None:
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300)
    plt.close(fig)

