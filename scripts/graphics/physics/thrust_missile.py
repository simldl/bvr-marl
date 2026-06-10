"""Air-to-air missile available thrust vs flight time and phase graphic."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_style import paper_figure, save_paper_figure  # noqa: E402

from bvr_marl_core.missiles.core.phases import MissilePhaseManager  # noqa: E402


def plot_missile_thrust():
    """Plot missile thrust vs flight time for several motor profiles."""
    generic_phases = {
        "boost": {"duration_s": 20.0, "thrust_kN": 50.0},
        "middle": {"duration_s": 30.0, "thrust_kN": 45.0},
        "terminal": {"duration_s": 50.0, "thrust_kN": 0.0},
    }
    longrange_phases = {
        "boost": {"duration_s": 20.0, "thrust_kN": 55.0},
        "middle": {"duration_s": 60.0, "thrust_kN": 40.0},
        "terminal": {"duration_s": 50.0, "thrust_kN": 0.0},
    }
    shortrange_phases = {
        "boost": {"duration_s": 20.0, "thrust_kN": 65.0},
        "middle": {"duration_s": 6.0, "thrust_kN": 50.0},
        "terminal": {"duration_s": 30.0, "thrust_kN": 0.0},
    }

    time_s = np.linspace(0, 150, 1000)
    generic_mgr = MissilePhaseManager(generic_phases, life_time_s=160.0, motor_burn_s=50.0)
    longrange_mgr = MissilePhaseManager(longrange_phases, life_time_s=160.0, motor_burn_s=80.0)
    shortrange_mgr = MissilePhaseManager(shortrange_phases, life_time_s=80.0, motor_burn_s=26.0)

    thrust_generic = np.zeros_like(time_s)
    thrust_longrange = np.zeros_like(time_s)
    thrust_shortrange = np.zeros_like(time_s)
    for i, t in enumerate(time_s):
        generic_mgr.update(t)
        longrange_mgr.update(t)
        shortrange_mgr.update(t)
        thrust_generic[i] = generic_mgr.get_thrust_kN()
        thrust_longrange[i] = longrange_mgr.get_thrust_kN()
        thrust_shortrange[i] = shortrange_mgr.get_thrust_kN()

    fig, ax = paper_figure()
    ax.plot(time_s, thrust_generic, color="blue", linewidth=1.6, label="Medium-range")
    ax.plot(time_s, thrust_longrange, color="red", linestyle="--", label="Long-range")
    ax.plot(time_s, thrust_shortrange, color="green", linestyle=":", label="Short-range")

    ax.axvline(x=20, color="gray", alpha=0.4, linewidth=0.8)
    ax.text(21, 66, "Boost→Middle", fontsize=8, rotation=90, va="top")
    ax.axvline(x=50, color="gray", alpha=0.4, linewidth=0.8)
    ax.text(51, 66, "Motor burnout", fontsize=8, rotation=90, va="top")

    ax.set_xlabel("Flight time [s]")
    ax.set_ylabel("Available thrust [kN]")
    ax.set_ylim([0, 70])
    ax.grid(True, which="both")
    ax.legend(loc="upper right")
    return fig


if __name__ == "__main__":
    save_paper_figure(plot_missile_thrust(), "thrust_missile")
