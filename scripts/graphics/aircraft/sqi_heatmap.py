"""SQI (shot quality index) heatmap over slant range and closure rate."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_style import paper_figure, save_paper_figure
from sqi_model import compute_sqi


def create_sqi_heatmap():
    """SQI over slant range and the target's line-of-sight velocity.

    The vertical axis is the TARGET's velocity component along the line of
    sight -- positive when it is running -- because that, not the total closing
    speed, is what the zone model reads.
    """
    range_vals = np.linspace(2.0, 170.0, 90)
    escape_vals = np.linspace(-400.0, 400.0, 90)

    sqi = np.empty((escape_vals.size, range_vals.size))
    for i, escape in enumerate(escape_vals):
        # A target running at |escape| heads away (yaw 0); closing heads in (180).
        yaw = 0.0 if escape >= 0 else 180.0
        speed = abs(float(escape))
        for j, rng in enumerate(range_vals):
            sqi[i, j] = compute_sqi(float(rng), tgt_speed_mps=speed, tgt_yaw_deg=yaw)

    grid_x, grid_y = np.meshgrid(range_vals, escape_vals)

    fig, ax = paper_figure(row_height_in=2.8)
    im = ax.contourf(grid_x, grid_y, sqi, levels=20, cmap="RdYlGn")
    contours = ax.contour(
        grid_x,
        grid_y,
        sqi,
        levels=[0.3, 0.45, 0.6, 0.8],
        colors="black",
        alpha=0.4,
        linewidths=0.8,
    )
    ax.clabel(contours, inline=True, fontsize=8, fmt="%.2f")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("SQI value", fontsize=8)
    cbar.ax.tick_params(labelsize=8)

    ax.axhline(0.0, color="white", linestyle=":", alpha=0.8, linewidth=1.2)
    ax.set_xlabel("Slant range [km]")
    ax.set_ylabel("Target LOS velocity [m/s]   (+ = running)")
    ax.set_title("SQI (shooter 10 km, 300 m/s, co-altitude)")
    ax.set_xlim(range_vals[0], range_vals[-1])
    ax.set_ylim(-400, 400)
    return fig


if __name__ == "__main__":
    save_paper_figure(create_sqi_heatmap(), "sqi_heatmap")
