"""Fighter-jet turn rate vs Mach number graphic."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_style import paper_figure, save_paper_figure

from bvr_marl_core.physics.aircraft import AircraftPhysics
from bvr_marl_core.physics.physics import get_speed_of_sound


def plot_aircraft_turn_rate():
    """Plot fighter-jet instantaneous turn rate and its limits vs Mach number."""
    aircraft = AircraftPhysics(AircraftPhysics.Params(mass_kg=19_700.0, reference_area_m2=78.0))
    alt_m = 5_000.0
    mach = np.linspace(0.15, 1.8, 400)
    v_mps = get_speed_of_sound(alt_m) * mach

    turn_rate_cl = np.zeros_like(mach)
    turn_rate_n = np.zeros_like(mach)
    turn_rate_inst = np.zeros_like(mach)
    weight_n = aircraft.mass_kg * aircraft.g
    g = aircraft.g
    rho = aircraft.air.get_density(alt_m)
    for i, v in enumerate(v_mps):
        if v < 1e-3:
            continue
        q = 0.5 * rho * v**2
        n_cl = (q * aircraft.A_m2 * aircraft.get_cl_max(v, alt_m)) / weight_n
        if n_cl >= 1.0:
            turn_rate_cl[i] = math.degrees((g / v) * math.sqrt(max(0.0, n_cl**2 - 1.0)))
        turn_rate_n[i] = math.degrees((g / v) * math.sqrt(max(0.0, aircraft.n_max**2 - 1.0)))
        turn_rate_inst[i] = aircraft.compute_instantaneous_turn_rate(v, alt_m)

    fig, ax = paper_figure()
    ax.plot(mach, turn_rate_cl, color="green", linestyle="--", label=r"$\dot\chi(C_L^{\max})$")
    ax.plot(mach, turn_rate_n, color="red", linestyle="--", label=r"$\dot\chi(n_{\max}=9)$")
    ax.plot(mach, turn_rate_inst, color="blue", linewidth=1.6, label="Instantaneous")
    ax.set_xlabel("Mach number $M$")
    ax.set_ylabel(r"Turn rate $\dot{\chi}$ [deg/s]")
    ax.set_ylim([0, 30])
    ax.grid(True, which="both")
    ax.legend(loc="upper right")
    return fig


if __name__ == "__main__":
    save_paper_figure(plot_aircraft_turn_rate(), "turn_rate_aircraft")
