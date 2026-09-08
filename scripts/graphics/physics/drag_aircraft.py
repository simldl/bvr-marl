"""Fighter-jet drag vs Mach number graphic."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_style import paper_figure, save_paper_figure

from bvr_marl_core.physics.aircraft import AircraftPhysics
from bvr_marl_core.physics.physics import get_speed_of_sound


def plot_aircraft_drag():
    """Plot fighter-jet drag components vs Mach number at 5000 m altitude."""
    aircraft = AircraftPhysics(AircraftPhysics.Params(mass_kg=19_700.0, reference_area_m2=78.0))
    alt_m = 5_000.0
    mach = np.linspace(0.05, 2.1, 400)
    v_mps = get_speed_of_sound(alt_m) * mach
    rho = aircraft.air.get_density(alt_m)

    d_parasite = np.zeros_like(mach)
    d_induced = np.zeros_like(mach)
    weight_n = aircraft.mass_kg * aircraft.g
    for i, v in enumerate(v_mps):
        q = 0.5 * rho * v**2
        d_parasite[i] = q * aircraft.A_m2 * aircraft.get_base_drag_cd(v, alt_m)
        cl = weight_n / (0.5 * rho * v**2 * aircraft.A_m2) if v > 1e-3 else 0.0
        d_induced[i] = q * aircraft.A_m2 * aircraft.K_ind * cl**2
    d_total = d_parasite + d_induced

    fig, ax = paper_figure()
    ax.plot(mach, d_parasite / 1e3, color="gray", label=r"$D_{\mathrm{parasite/wave}}$")
    ax.plot(mach, d_induced / 1e3, color="red", label=r"$D_{\mathrm{induced}}$")
    ax.plot(mach, d_total / 1e3, color="blue", linewidth=1.6, label=r"$D_{\mathrm{total}}$")
    ax.set_xlabel("Mach number $M$")
    ax.set_ylabel("Drag $D$ [kN]")
    ax.set_ylim([0, 500])
    ax.grid(True, which="both")
    ax.legend(loc="upper left")
    return fig


if __name__ == "__main__":
    save_paper_figure(plot_aircraft_drag(), "drag_aircraft")
