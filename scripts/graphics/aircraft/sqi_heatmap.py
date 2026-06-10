"""SQI (shot quality index) heatmap over slant range and closure rate."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_style import paper_figure, save_paper_figure  # noqa: E402
from sqi_model import compute_sqi, get_amraam_params  # noqa: E402


def create_sqi_heatmap():
    """Plot an SQI heatmap as a function of range and closure rate."""
    base_range_km, _ = get_amraam_params()
    range_vals = np.linspace(0, base_range_km, 80)
    closure_vals = np.linspace(-200, 600, 80)
    grid_x, grid_y = np.meshgrid(range_vals, closure_vals)

    distance_norm = 1.0 - np.clip(grid_x / base_range_km, 0, 1)
    closure_norm = np.clip(grid_y / 400, -1, 1)
    sqi = compute_sqi(distance_norm, closure_norm, 0.0)

    fig, ax = paper_figure(row_height_in=2.8)
    im = ax.contourf(grid_x, grid_y, sqi, levels=20, cmap="RdYlGn")
    contours = ax.contour(
        grid_x,
        grid_y,
        sqi,
        levels=[0.3, 0.55, 0.7, 0.9],
        colors="black",
        alpha=0.4,
        linewidths=0.8,
    )
    ax.clabel(contours, inline=True, fontsize=8, fmt="%.2f")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("SQI value", fontsize=8)
    cbar.ax.tick_params(labelsize=8)

    ax.axhline(400, color="white", linestyle=":", alpha=0.8, linewidth=1.2)
    ax.text(
        base_range_km * 0.05,
        415,
        "Optimal closure",
        fontsize=8,
        color="white",
        bbox=dict(boxstyle="round", facecolor="black", alpha=0.5),
    )
    ax.set_xlabel("Slant range [km]")
    ax.set_ylabel("Closure rate [m/s]")
    ax.set_xlim(0, base_range_km)
    ax.set_ylim(-200, 600)
    return fig


if __name__ == "__main__":
    save_paper_figure(create_sqi_heatmap(), "sqi_heatmap")
