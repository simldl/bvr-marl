"""Air density vs altitude graphic based on the ISA model."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_style import paper_figure, save_paper_figure  # noqa: E402

from bvr_marl_core.physics.physics import AirLayer  # noqa: E402


def plot_air_density():
    """Plot air density vs altitude using the actual ``AirLayer`` ISA model."""
    air = AirLayer()
    altitude_m = np.linspace(0, 25_000, 300)
    density_kg_m3 = np.array([air.get_density(alt) for alt in altitude_m])

    fig, ax = paper_figure()
    ax.plot(altitude_m, density_kg_m3, color="blue", linewidth=1.6)
    ax.set_xlabel("Altitude [m]")
    ax.set_ylabel("Density [kg/m$^3$]")
    ax.set_title("Air density vs altitude")
    ax.grid(True, which="both")
    return fig


if __name__ == "__main__":
    save_paper_figure(plot_air_density(), "air_density")
