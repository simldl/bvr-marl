"""Fighter-jet parasite drag coefficient (CD0) vs Mach number graphic."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_style import paper_figure, save_paper_figure  # noqa: E402

from bvr_marl_core.physics.aircraft import AircraftPhysics  # noqa: E402
from bvr_marl_core.physics.physics import get_speed_of_sound  # noqa: E402


def plot_aircraft_cd0():
    """Plot fighter-jet parasite drag coefficient vs Mach number at 5000 m."""
    aircraft = AircraftPhysics(AircraftPhysics.Params(mass_kg=19_700.0, reference_area_m2=78.0))
    alt_m = 5_000.0
    mach = np.linspace(0.05, 2.1, 400)
    v_mps = get_speed_of_sound(alt_m) * mach
    cd0 = np.array([aircraft.get_base_drag_cd(v, alt_m) for v in v_mps])

    fig, ax = paper_figure()
    ax.plot(mach, cd0, color="blue", linewidth=1.6)
    ax.set_xlabel("Mach number $M$")
    ax.set_ylabel(r"Parasite drag coefficient $C_{D0}$")
    ax.set_title("Fighter jet at 5000 m")
    ax.grid(True, which="both")
    return fig


if __name__ == "__main__":
    save_paper_figure(plot_aircraft_cd0(), "drag_coefficient_aircraft")
