"""Fighter-jet available thrust vs Mach number graphic (with afterburner)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_style import paper_figure, save_paper_figure

from bvr_marl_core.physics.aircraft import AircraftPhysics
from bvr_marl_core.physics.physics import get_speed_of_sound


def plot_aircraft_thrust():
    """Plot military (dry) and afterburner thrust vs Mach number at 5000 m."""
    aircraft = AircraftPhysics(AircraftPhysics.Params(mass_kg=19_700.0, reference_area_m2=78.0))
    alt_m = 5_000.0
    mach = np.linspace(0.05, 2.1, 400)
    v_mps = get_speed_of_sound(alt_m) * mach

    thrust_mil = np.array([aircraft.get_engine_force(v, alt_m, throttle=0.5) for v in v_mps])
    thrust_ab = np.array([aircraft.get_engine_force(v, alt_m, throttle=1.0) for v in v_mps])

    fig, ax = paper_figure()
    ax.plot(mach, thrust_mil / 1e3, color="red", linestyle="--", label="Military thrust (50 %)")
    ax.plot(mach, thrust_ab / 1e3, color="blue", linewidth=1.6, label="Afterburner thrust (100 %)")
    ax.set_xlabel("Mach number $M$")
    ax.set_ylabel("Available thrust [kN]")
    ax.grid(True, which="both")
    ax.legend(loc="upper left")
    return fig


if __name__ == "__main__":
    save_paper_figure(plot_aircraft_thrust(), "thrust_aircraft")
