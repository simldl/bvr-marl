"""NEZ (no-escape zone) and DLZ (dynamic launch zone) visualisation.

Top panel: R1-R4 zone bands for AMRAAM and Meteor at a stated reference
geometry. Bottom panel: how the R_PI edge moves with target aspect and with
altitude difference.

Every edge is read from ``NoEscapeZoneCalculator.compute_dlz`` via
``nez_probe``. The previous version of this figure computed the edges from its
own fractions and drew the aspect dependence as a cosine, neither of which
matched the model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nez_probe import amraam_params, dlz_at, meteor_params  # noqa: E402
from paper_style import paper_figure, save_paper_figure  # noqa: E402

# Reference geometry for the band comparison: co-altitude at 10 km, shooter at
# 300 m/s, target closing head-on at 250 m/s.
REF = dict(own_alt_m=10_000.0, own_speed_mps=300.0, tgt_alt_m=10_000.0, tgt_speed_mps=250.0)


def _edges_km(params):
    d = dlz_at(params=params, tgt_yaw_deg=180.0, **REF)
    return (
        d.r_min_m / 1000.0,
        d.r_nez_out_m / 1000.0,
        d.r_tr_m / 1000.0,
        d.r_pi_m / 1000.0,
        d.r_aero_m / 1000.0,
    )


def _draw_dlz_bands(ax, amraam, meteor):
    """Draw stacked R1-R4 range bands for the two missiles on ``ax``."""
    y_height = 0.4
    missiles = [
        ("AIM-120D AMRAAM", *amraam, ["#95E1D3", "#6DDF9B", "#38E19B", "#0BC965"]),
        ("Meteor", *meteor, ["#FFE66D", "#FFC700", "#FF9800", "#FF6B35"]),
    ]
    for y_pos, (name, r_min, r_tr, r_pi, r_aero, colors) in enumerate(missiles):
        segments = [
            (0, r_min, colors[0]),
            (r_min, r_tr - r_min, colors[1]),
            (r_tr, r_pi - r_tr, colors[2]),
            (r_pi, r_aero - r_pi, colors[3]),
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
        # R1 is ~1.5 km against a ~150 km axis; a label there would collide with
        # the axis, so only segments wide enough to hold text are annotated.
        labels = [
            (r_min, r_tr, "R2"),
            (r_tr, r_pi, "R3"),
            (r_pi, r_aero, "R4"),
        ]
        for left, right, text in labels:
            if (right - left) > 0.05 * r_aero:
                ax.text(
                    (left + right) / 2,
                    y_pos,
                    text,
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                )
        ax.text(
            0,
            y_pos + y_height / 2 + 0.06,
            name,
            ha="left",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )
        ax.text(
            r_aero + 3,
            y_pos,
            f"{r_aero:.0f} km",
            ha="left",
            va="center",
            fontsize=8,
            fontweight="bold",
        )


def create_nez_dlz_plot():
    """Build the stacked NEZ/DLZ figure from the production zone model."""
    amraam, meteor = amraam_params(), meteor_params()
    a_min, _, a_tr, a_pi, a_aero = _edges_km(amraam)
    m_min, _, m_tr, m_pi, m_aero = _edges_km(meteor)

    fig, (ax_top, ax_bot) = paper_figure(nrows=2, row_height_in=2.6)

    _draw_dlz_bands(ax_top, (a_min, a_tr, a_pi, a_aero), (m_min, m_tr, m_pi, m_aero))
    ax_top.set_xlim(-8, max(a_aero, m_aero) * 1.16)
    # Headroom above the bars for the legend, so it never covers a range band.
    ax_top.set_ylim(-0.45, 2.45)
    ax_top.set_xlabel("Slant range [km]")
    ax_top.set_title("Fox-3 DLZ, head-on at 10 km / 300 m/s")
    ax_top.set_yticks([])
    ax_top.grid(axis="x", linestyle=":")
    ax_top.legend(
        handles=[
            Patch(facecolor="#6DDF9B", edgecolor="black", label="R2: to $R_{TR}$"),
            Patch(facecolor="#38E19B", edgecolor="black", label="R3: to $R_{PI}$"),
            Patch(facecolor="#0BC965", edgecolor="black", label="R4: to $R_{Aero}$"),
        ],
        loc="upper center",
        ncol=3,
        fontsize=7,
        handlelength=1.4,
        columnspacing=1.0,
        framealpha=0.9,
    )

    # Aspect sweep: target heading 180 deg closes head-on, 0 deg runs away.
    aspect = np.linspace(0.0, 180.0, 61)
    for delta_h_m, style, label in (
        (3000.0, "-", r"$\Delta h = +3$ km"),
        (0.0, "--", r"$\Delta h = 0$"),
        (-3000.0, ":", r"$\Delta h = -3$ km"),
    ):
        r_pi = [
            dlz_at(
                params=amraam,
                own_alt_m=10_000.0 + delta_h_m,
                own_speed_mps=300.0,
                tgt_alt_m=10_000.0,
                tgt_speed_mps=250.0,
                tgt_yaw_deg=float(a),
            ).r_pi_m
            / 1000.0
            for a in aspect
        ]
        ax_bot.plot(aspect, r_pi, style, linewidth=1.4, label=label)

    ax_bot.set_xlabel("Target heading [deg]  (0 = running, 180 = head-on)")
    ax_bot.set_ylabel("$R_{PI}$ [km]")
    ax_bot.set_title("AMRAAM $R_{PI}$ vs aspect and altitude difference")
    ax_bot.set_xlim(0, 180)
    ax_bot.set_ylim(0, None)
    ax_bot.set_xticks([0, 45, 90, 135, 180])
    ax_bot.grid(True, linestyle=":")
    ax_bot.legend(loc="upper left")
    return fig


if __name__ == "__main__":
    save_paper_figure(create_nez_dlz_plot(), "nez_dlz_zones")
