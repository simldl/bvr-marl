"""NEZ (no-escape zone) and DLZ (dynamic launch zone) visualisation.

Top panel: DLZ band comparison (R1-R4) for AMRAAM and Meteor.
Bottom panel: AMRAAM engagement ranges vs target aspect angle.
Panels are stacked so each keeps the full single-column width.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_style import paper_figure, save_paper_figure  # noqa: E402
from sqi_model import get_amraam_params, get_meteor_base_range_km  # noqa: E402


def _dlz_zones(min_range_km, base_range_km):
    """Return (r_tr, r_pi, r_aero, r_max) using the nez.py:compute_dlz() formula."""
    r_tr = min_range_km + 0.60 * (base_range_km - min_range_km)
    r_pi = min_range_km + 0.88 * (base_range_km - min_range_km)
    r_aero = min_range_km + 1.04 * (base_range_km - min_range_km)
    r_max = base_range_km * 1.3
    return r_tr, r_pi, r_aero, r_max


def _draw_dlz_bands(ax, amraam, meteor):
    """Draw stacked R1-R4 range bands for the two missiles on ``ax``."""
    y_height = 0.4
    missiles = [
        ("AIM-120D AMRAAM", *amraam, ["#95E1D3", "#6DDF9B", "#38E19B", "#0BC965"]),
        ("Meteor", *meteor, ["#FFE66D", "#FFC700", "#FF9800", "#FF6B35"]),
    ]
    for y_pos, (name, r_min, r_tr, r_pi, r_max, colors) in enumerate(missiles):
        segments = [
            (0, r_min, colors[0]),
            (r_min, r_tr - r_min, colors[1]),
            (r_tr, r_pi - r_tr, colors[2]),
            (r_pi, r_max - r_pi, colors[3]),
        ]
        for left, width, color in segments:
            ax.barh(
                y_pos,
                width,
                left=left,
                height=y_height,
                color=color,
                alpha=0.8,
                edgecolor="black",
                linewidth=0.8,
            )
        labels = [
            (r_min / 2, "R1"),
            ((r_min + r_tr) / 2, "R2\nNEZ"),
            ((r_tr + r_pi) / 2, "R3\nopt."),
            ((r_pi + r_max) / 2, "R4\next."),
        ]
        for x, text in labels:
            ax.text(x, y_pos, text, ha="center", va="center", fontsize=8, fontweight="bold")
        ax.text(
            0, y_pos + y_height + 0.12, name, ha="left", va="bottom", fontsize=8, fontweight="bold"
        )
        ax.text(r_max + 2, y_pos, f"{r_max:.0f} km", ha="left", va="center", fontsize=8)


def create_nez_dlz_plot():
    """Build the stacked NEZ/DLZ figure."""
    amraam_base_km, amraam_min_km = get_amraam_params()
    meteor_base_km = get_meteor_base_range_km()
    meteor_min_km = 2.0

    a_tr, a_pi, _, a_max = _dlz_zones(amraam_min_km, amraam_base_km)
    m_tr, m_pi, _, m_max = _dlz_zones(meteor_min_km, meteor_base_km)

    fig, (ax_top, ax_bot) = paper_figure(nrows=2, row_height_in=2.6)

    _draw_dlz_bands(
        ax_top,
        (amraam_min_km, a_tr, a_pi, a_max),
        (meteor_min_km, m_tr, m_pi, m_max),
    )
    ax_top.set_xlim(-15, 280)
    ax_top.set_ylim(-0.5, 2.0)
    ax_top.set_xlabel("Slant range [km]")
    ax_top.set_title("Fox-3 missile DLZ comparison")
    ax_top.set_yticks([])
    ax_top.grid(axis="x", linestyle=":")
    legend_elements = [
        Patch(facecolor="#95E1D3", edgecolor="black", label="R1: too close"),
        Patch(facecolor="#6DDF9B", edgecolor="black", label="R2: NEZ (best)"),
        Patch(facecolor="#38E19B", edgecolor="black", label="R3: optimal"),
        Patch(facecolor="#0BC965", edgecolor="black", label="R4: extended"),
    ]
    ax_top.legend(handles=legend_elements, loc="lower right", ncol=2)

    aspect = np.linspace(0, 180, 100)
    aspect_mod = np.cos(2 * np.radians(aspect - 90))
    nez = a_tr + 8 * aspect_mod
    r3 = a_pi + 12 * aspect_mod
    r_max_aspect = a_max + 15 * aspect_mod
    ax_bot.fill_between(aspect, 0, nez, alpha=0.3, color="green")
    ax_bot.fill_between(aspect, nez, r3, alpha=0.2, color="yellow")
    ax_bot.fill_between(aspect, r3, r_max_aspect, alpha=0.1, color="orange")
    ax_bot.plot(aspect, nez, color="green", linewidth=1.4, label="NEZ boundary")
    ax_bot.plot(aspect, r3, color="darkorange", linewidth=1.4, linestyle="--", label="R3 boundary")
    ax_bot.plot(
        aspect, r_max_aspect, color="darkred", linewidth=1.4, linestyle=":", label="Max range"
    )
    for x, color in ((0, "blue"), (90, "darkgreen"), (180, "blue")):
        ax_bot.axvline(x, color=color, linestyle=":", alpha=0.5, linewidth=1.0)
    ax_bot.set_xlabel("Target aspect angle [deg]")
    ax_bot.set_ylabel("Effective range [km]")
    ax_bot.set_title(f"AMRAAM NEZ vs aspect ({amraam_base_km:.0f} km base)")
    ax_bot.set_xlim(0, 180)
    ax_bot.set_ylim(0, m_max * 1.05)
    ax_bot.set_xticks([0, 45, 90, 135, 180])
    ax_bot.grid(True, linestyle=":")
    ax_bot.legend(loc="upper right")
    return fig


if __name__ == "__main__":
    save_paper_figure(create_nez_dlz_plot(), "nez_dlz_zones")
