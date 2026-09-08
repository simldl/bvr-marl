"""SQI (shot quality index) vs target aspect angle."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_style import paper_figure, save_paper_figure
from sqi_model import compute_sqi


def create_sqi_aspect_angle_plot():
    """Plot SQI vs target heading for several slant ranges."""
    headings = np.linspace(0.0, 180.0, 73)
    ranges_km = [20.0, 40.0, 60.0, 90.0, 120.0]

    fig, ax = paper_figure()
    for rng in ranges_km:
        sqi = [compute_sqi(rng, tgt_speed_mps=250.0, tgt_yaw_deg=float(h)) for h in headings]
        ax.plot(headings, sqi, linewidth=1.4, label=f"{rng:.0f} km", marker="s", markevery=12)

    ax.axvline(90, color="blue", linestyle=":", alpha=0.5, linewidth=1.0)
    ax.set_xlabel("Target heading [deg]")
    ax.set_ylabel("Shot quality index")
    ax.set_title("SQI vs aspect (target 250 m/s, co-altitude)")
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 1.0)
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_xticklabels(["0\n(running)", "45", "90\n(beam)", "135", "180\n(head-on)"])
    ax.legend(loc="lower right", ncol=2, fontsize=7, framealpha=0.9)
    ax.grid(True, linestyle=":")
    return fig


if __name__ == "__main__":
    save_paper_figure(create_sqi_aspect_angle_plot(), "sqi_aspect_angle")
