"""Detection-probability heatmap for an Eurofighter radar vs an Eurofighter."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_style import paper_figure, save_paper_figure

from bvr_marl_core.aircraft.types.eurofighter import Eurofighter
from bvr_marl_core.radar.core.lut import DetectionLUT
from bvr_marl_core.simulator.core.helpers import Position

_MAP_LIMITS = {"north": 100000, "south": -100000, "east": 100000, "west": -100000}


def plot_detection_probability_heatmap():
    """Plot P(detect) over distance x RCS for an Eurofighter radar."""
    ref_pos = Position(lat=45.0, lon=10.0, alt=5000.0)
    eurofighter = Eurofighter(
        position=ref_pos,
        yaw_deg=0.0,
        speed_mps=250.0,
        group="blue",
        map_limits=_MAP_LIMITS,
        min_alt_m=0.0,
        max_alt_m=20000.0,
    )
    cfg = eurofighter.config
    lut = DetectionLUT(
        freq_hz=cfg.get("radar_frequency_hz", 10e9),
        tx_power_w=cfg.get("radar_tx_power_w", 18e3),
        gain=10 ** (cfg.get("radar_antenna_gain_db", 36.0) / 10),
        max_range_m=200_000.0,
        snr_threshold_db=cfg.get("radar_snr_threshold_db", 9.0),
        # Coherent-integration (processing) gain — the same value the real
        # aircraft radar uses. Without it the LUT reproduces the pre-refactor
        # (gainless) detection and a fighter cannot see a fighter at range.
        processing_gain_db=cfg.get("radar_processing_gain_db", 30.0),
        max_rcs=3.5,
        rcs_bins=256,
        dist_bins=256,
    )

    eurofighter_rcs_nominal = 3.0
    distances_km = np.linspace(0, 200, 256)
    distances_m = distances_km * 1000
    rcs_values = np.linspace(0, 3.5, 256)
    dist_mesh, rcs_mesh = np.meshgrid(distances_m, rcs_values, indexing="ij")
    prob = np.vectorize(lut.get_probability)(dist_mesh, rcs_mesh)

    fig, ax = paper_figure(row_height_in=2.8)
    im = ax.contourf(distances_km, rcs_values, prob.T, levels=20, cmap="RdYlGn")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Detection probability", fontsize=8)
    cbar.ax.tick_params(labelsize=8)
    ax.axhline(
        y=eurofighter_rcs_nominal,
        color="blue",
        linestyle="--",
        linewidth=1.2,
        label=f"Eurofighter RCS ({eurofighter_rcs_nominal:.1f} m$^2$)",
    )
    ax.set_xlabel("Distance [km]")
    ax.set_ylabel("Radar cross section [m$^2$]")
    ax.legend(loc="upper right")
    ax.grid(True)
    return fig


if __name__ == "__main__":
    save_paper_figure(plot_detection_probability_heatmap(), "detection_probability_heatmap")
