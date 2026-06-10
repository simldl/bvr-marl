"""Air-to-air missile parasite drag coefficient (CD0) vs Mach number graphic."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_style import paper_figure, save_paper_figure  # noqa: E402

from bvr_marl_core.physics.missiles import MissilePhysics  # noqa: E402
from bvr_marl_core.physics.physics import get_speed_of_sound  # noqa: E402


def plot_missile_cd0():
    """Plot missile parasite drag coefficient vs Mach number at 5000 m."""
    missile = MissilePhysics(MissilePhysics.Params(mass_kg=300.0, reference_area_m2=0.08))
    alt_m = 5_000.0
    mach = np.linspace(0.05, 5.0, 500)
    v_mps = get_speed_of_sound(alt_m) * mach
    cd0 = np.array([missile.get_base_drag_cd(v, alt_m) for v in v_mps])

    fig, ax = paper_figure()
    ax.plot(mach, cd0, color="blue", linewidth=1.6)
    ax.set_xlabel("Mach number $M$")
    ax.set_ylabel(r"Parasite drag coefficient $C_{D0}$")
    ax.set_title("Air-to-air missile at 5000 m")
    ax.grid(True, which="both")
    return fig


if __name__ == "__main__":
    save_paper_figure(plot_missile_cd0(), "drag_coefficient_missile")
