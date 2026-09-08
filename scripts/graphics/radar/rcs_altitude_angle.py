"""Radar cross section (RCS) vs altitude (elevation) angle for several types."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_style import paper_figure, save_paper_figure

from bvr_marl_core.aircraft.types.eurofighter import Eurofighter
from bvr_marl_core.aircraft.types.f22 import F22
from bvr_marl_core.aircraft.types.f35 import F35
from bvr_marl_core.aircraft.types.su57 import Su57
from bvr_marl_core.radar.core.utils import _effective_rcs
from bvr_marl_core.simulator.core.helpers import Position

_MAP_LIMITS = {"north": 100000, "south": -100000, "east": 100000, "west": -100000}


def plot_rcs_altitude_angle():
    """Plot effective RCS vs elevation angle (negative = viewed from below)."""
    ref_pos = Position(lat=45.0, lon=10.0, alt=5000.0)
    common = dict(
        position=ref_pos,
        yaw_deg=0.0,
        speed_mps=250.0,
        group="blue",
        map_limits=_MAP_LIMITS,
        min_alt_m=0.0,
        max_alt_m=20000.0,
    )
    f22 = F22(**common)
    f35 = F35(**common)
    eurofighter = Eurofighter(**common)
    su57 = Su57(
        position=ref_pos,
        yaw_deg=0.0,
        speed_mps=250.0,
        group="red",
        map_limits=_MAP_LIMITS,
        min_alt_m=0.0,
        max_alt_m=20000.0,
    )

    el_deg = np.linspace(-90, 90, 181)
    rcs = {ac: np.zeros_like(el_deg) for ac in ("f22", "f35", "su57", "ef")}
    horiz_dist = 100000.0
    for i, el in enumerate(el_deg):
        vert_dist = horiz_dist * np.tan(np.radians(el))
        radar_pos = Position(
            lat=ref_pos.lat + (horiz_dist / 111000.0), lon=ref_pos.lon, alt=ref_pos.alt + vert_dist
        )
        rcs["f22"][i] = _effective_rcs(f22, radar_pos)
        rcs["f35"][i] = _effective_rcs(f35, radar_pos)
        rcs["su57"][i] = _effective_rcs(su57, radar_pos)
        rcs["ef"][i] = _effective_rcs(eurofighter, radar_pos)

    fig, ax = paper_figure(row_height_in=2.6)
    ax.plot(
        el_deg,
        rcs["f22"],
        color="blue",
        linewidth=1.6,
        label=f"F-22 ($\\sigma_0$={f22.rcs:.4f} m$^2$)",
    )
    ax.plot(
        el_deg,
        rcs["f35"],
        color="cyan",
        linestyle="--",
        label=f"F-35 ($\\sigma_0$={f35.rcs:.4f} m$^2$)",
    )
    ax.plot(
        el_deg,
        rcs["su57"],
        color="red",
        linestyle="-.",
        label=f"Su-57 ($\\sigma_0$={su57.rcs:.4f} m$^2$)",
    )
    ax.plot(
        el_deg,
        rcs["ef"],
        color="green",
        linestyle=":",
        label=f"Eurofighter ($\\sigma_0$={eurofighter.rcs:.4f} m$^2$)",
    )
    ax.axvline(x=0.0, color="gray", linestyle="--", alpha=0.4, linewidth=0.6)

    ax.set_xlabel("Elevation angle [deg]")
    ax.set_ylabel("Radar cross section [m$^2$]")
    ax.set_xlim([-90, 90])
    ax.set_yscale("log")
    ax.grid(True, which="both")
    ax.legend(loc="upper left")
    return fig


if __name__ == "__main__":
    save_paper_figure(plot_rcs_altitude_angle(), "rcs_altitude_angle")
