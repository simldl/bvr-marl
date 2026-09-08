"""SQI (shot quality index) vs slant range for several target behaviours."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_style import paper_figure, save_paper_figure  # noqa: E402
from sqi_model import compute_sqi, dlz_at  # noqa: E402

# The zone model reads the target's velocity component along the line of sight,
# so a "closure rate" is expressed here as what the target is doing.
CASES = [
    ("Head-on, 250 m/s", 250.0, 180.0),
    ("Head-on, 400 m/s", 400.0, 180.0),
    ("Beaming", 250.0, 90.0),
    ("Running, 250 m/s", 250.0, 0.0),
    ("Running, 400 m/s", 400.0, 0.0),
]


def create_sqi_range_closure_plot():
    """Plot SQI vs slant range for a range of target aspects and speeds."""
    range_km = np.linspace(2.0, 170.0, 120)

    fig, ax = paper_figure()
    for label, speed, yaw in CASES:
        sqi = [compute_sqi(r, tgt_speed_mps=speed, tgt_yaw_deg=yaw) for r in range_km]
        ax.plot(range_km, sqi, linewidth=1.4, label=label)

    # Mark the zone edges for the head-on reference case the book quotes.
    ref = dlz_at(tgt_speed_mps=250.0, tgt_yaw_deg=180.0)
    for value, name in (
        (ref.r_nez_out_m / 1000.0, "$R_{NEZ}$"),
        (ref.r_pi_m / 1000.0, "$R_{PI}$"),
        (ref.r_aero_m / 1000.0, "$R_{Aero}$"),
    ):
        ax.axvline(value, color="0.55", linestyle=":", linewidth=0.9)
        ax.text(
            value - 1.5,
            0.60,
            name,
            ha="right",
            va="bottom",
            fontsize=7,
            color="0.35",
            rotation=90,
        )

    ax.set_xlabel("Slant range [km]")
    ax.set_ylabel("Shot quality index")
    ax.set_title("SQI vs range (shooter 10 km, 300 m/s)")
    ax.set_xlim(0, 170)
    ax.set_ylim(0, 1.0)
    ax.grid(True, linestyle=":")
    ax.legend(loc="upper right", fontsize=7)
    return fig


if __name__ == "__main__":
    save_paper_figure(create_sqi_range_closure_plot(), "sqi_range_closure")
