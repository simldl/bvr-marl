"""SQI (shot quality index) vs slant range for several closure rates."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_style import paper_figure, save_paper_figure  # noqa: E402
from sqi_model import compute_sqi, get_amraam_params  # noqa: E402


def create_sqi_range_closure_plot():
    """Plot SQI vs slant range for a range of closure rates."""
    base_range_km, _ = get_amraam_params()
    range_km = np.linspace(0, base_range_km, 100)
    closure_rates_mps = [-400, -200, 0, 100, 200, 400, 600]

    fig, ax = paper_figure()
    distance_normalized = 1.0 - np.clip(range_km / base_range_km, 0, 1)
    for closure_rate in closure_rates_mps:
        closure_norm = np.clip(closure_rate / 400, -1, 1)
        sqi = [compute_sqi(d, closure_norm, 0.0) for d in distance_normalized]
        if closure_rate == 0:
            label = "Stationary"
        elif closure_rate < 0:
            label = f"Diverging {abs(closure_rate)} m/s"
        else:
            label = f"Closure {closure_rate} m/s"
        ax.plot(range_km, sqi, label=label, marker="o", markevery=10)

    ax.axhline(
        0.55, color="red", linestyle="--", linewidth=1.2, alpha=0.8, label="Threshold (0.55)"
    )
    ax.fill_between(range_km, 0.55, 1.0, alpha=0.1, color="green")
    ax.fill_between(range_km, 0, 0.55, alpha=0.1, color="red")
    ax.set_xlabel("Slant range [km]")
    ax.set_ylabel("SQI [0-1]")
    ax.set_xlim(0, base_range_km)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="best", ncol=2)
    ax.grid(True, linestyle=":")
    return fig


if __name__ == "__main__":
    save_paper_figure(create_sqi_range_closure_plot(), "sqi_range_closure")
