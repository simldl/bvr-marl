"""Detection-probability heatmaps for an F-22 radar vs F-22 and Eurofighter.

Each figure stacks the full RCS range (top) and a zoomed range (bottom) so both
panels keep the full single-column width.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_style import paper_figure, save_paper_figure

from bvr_marl_core.aircraft.types.f22 import F22
from bvr_marl_core.radar.core.lut import DetectionLUT
from bvr_marl_core.simulator.core.helpers import Position

_MAP_LIMITS = {"north": 100000, "south": -100000, "east": 100000, "west": -100000}


def _f22_radar_lut(max_rcs):
    ref_pos = Position(lat=45.0, lon=10.0, alt=5000.0)
    f22 = F22(
        position=ref_pos,
        yaw_deg=0.0,
        speed_mps=250.0,
        group="blue",
        map_limits=_MAP_LIMITS,
        min_alt_m=0.0,
        max_alt_m=20000.0,
    )
    cfg = f22.config
    return DetectionLUT(
        freq_hz=cfg.get("radar_frequency_hz", 10e9),
        tx_power_w=cfg.get("radar_tx_power_w", 5e3),
        gain=10 ** (cfg.get("radar_antenna_gain_db", 34.0) / 10),
        max_range_m=200_000.0,
        snr_threshold_db=cfg.get("radar_snr_threshold_db", 10.0),
        # Coherent-integration (processing) gain — match the real aircraft radar.
        processing_gain_db=cfg.get("radar_processing_gain_db", 30.0),
        max_rcs=max_rcs,
        rcs_bins=256,
        dist_bins=256,
    )


def _panel(ax, fig, lut, distances_km, rcs_values, rcs_marker, marker_label, marker_color):
    distances_m = distances_km * 1000
    dist_mesh, rcs_mesh = np.meshgrid(distances_m, rcs_values, indexing="ij")
    prob = np.vectorize(lut.get_probability)(dist_mesh, rcs_mesh)
    im = ax.contourf(distances_km, rcs_values, prob.T, levels=20, cmap="RdYlGn")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("P(detect)", fontsize=8)
    cbar.ax.tick_params(labelsize=8)
    contours = ax.contour(
        distances_km,
        rcs_values,
        prob.T,
        levels=[0.5, 0.9],
        colors="black",
        linewidths=0.8,
        alpha=0.6,
    )
    ax.clabel(contours, inline=True, fontsize=8, fmt="P=%.1f")
    ax.axhline(y=rcs_marker, color=marker_color, linestyle="--", linewidth=1.2, label=marker_label)
    ax.set_xlabel("Distance [km]")
    ax.set_ylabel("RCS [m$^2$]")
    ax.legend(loc="upper right")
    ax.grid(True)


def plot_f22_vs_f22():
    """F-22 radar vs F-22 target: full and zoomed RCS ranges."""
    lut = _f22_radar_lut(max_rcs=1.5)
    distances_km = np.linspace(0, 200, 256)
    nominal = 0.0069
    fig, (ax_full, ax_zoom) = paper_figure(nrows=2, row_height_in=2.6)
    _panel(
        ax_full,
        fig,
        lut,
        distances_km,
        np.linspace(0, 1.5, 256),
        nominal,
        f"F-22 RCS ({nominal:.4f} m$^2$)",
        "blue",
    )
    ax_full.set_title("F-22 radar vs F-22 (full range)")
    _panel(
        ax_zoom,
        fig,
        lut,
        distances_km,
        np.linspace(0.003, nominal * 100, 256),
        nominal,
        f"F-22 RCS ({nominal:.4f} m$^2$)",
        "blue",
    )
    ax_zoom.set_title("F-22 radar vs F-22 (zoom)")
    return fig


def plot_f22_vs_eurofighter():
    """F-22 radar vs Eurofighter target: full and zoomed RCS ranges."""
    lut = _f22_radar_lut(max_rcs=3.5)
    distances_km = np.linspace(0, 200, 256)
    nominal = 3.0
    fig, (ax_full, ax_zoom) = paper_figure(nrows=2, row_height_in=2.6)
    _panel(
        ax_full,
        fig,
        lut,
        distances_km,
        np.linspace(0, 3.5, 256),
        nominal,
        f"Eurofighter RCS ({nominal:.1f} m$^2$)",
        "red",
    )
    ax_full.set_title("F-22 radar vs Eurofighter (full range)")
    _panel(
        ax_zoom,
        fig,
        lut,
        distances_km,
        np.linspace(0.10, nominal * 1.2, 256),
        nominal,
        f"Eurofighter RCS ({nominal:.1f} m$^2$)",
        "red",
    )
    ax_zoom.set_title("F-22 radar vs Eurofighter (zoom)")
    return fig


if __name__ == "__main__":
    save_paper_figure(plot_f22_vs_f22(), "detection_probability_f22_vs_f22")
    save_paper_figure(plot_f22_vs_eurofighter(), "detection_probability_f22_vs_eurofighter")
