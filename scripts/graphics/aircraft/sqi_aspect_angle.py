"""SQI (shot quality index) vs target aspect angle."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_style import paper_figure, save_paper_figure  # noqa: E402
from sqi_model import compute_sqi, get_amraam_params  # noqa: E402


def create_sqi_aspect_angle_plot():
    """Plot SQI vs aspect angle for several normalised ranges."""
    base_range_km, _ = get_amraam_params()

    aspect_angles = np.linspace(0, 180, 100)
    aspect_cos = np.cos(np.radians(aspect_angles))
    ranges_km = [base_range_km * f for f in (0.067, 0.2, 0.333, 0.467, 0.6)]
    closure_norm = np.clip(400 / 400, -1, 1)

    fig, ax = paper_figure()
    for rng in ranges_km:
        distance_norm = 1.0 - np.clip(rng / base_range_km, 0, 1)
        sqi = [compute_sqi(distance_norm, closure_norm, c) for c in aspect_cos]
        ax.plot(aspect_angles, sqi, label=f"{rng:.0f} km", marker="s", markevery=10)

    ax.axhline(0.55, color="red", linestyle="--", linewidth=1.2, alpha=0.8, label="Shot threshold")
    ax.axvline(90, color="blue", linestyle=":", alpha=0.5, linewidth=1.0)
    ax.set_xlabel("Target aspect angle [deg]")
    ax.set_ylabel("SQI [0-1]")
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 1.0)
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_xticklabels(["0\n(head-on)", "45", "90\n(beam)", "135", "180\n(tail)"])
    ax.legend(loc="best", ncol=2)
    ax.grid(True, linestyle=":")
    return fig


if __name__ == "__main__":
    save_paper_figure(create_sqi_aspect_angle_plot(), "sqi_aspect_angle")
