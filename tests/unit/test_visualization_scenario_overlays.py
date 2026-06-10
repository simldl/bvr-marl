"""Tests for visualization scenario overlay helpers."""

from __future__ import annotations

from types import SimpleNamespace

from bvr_marl_core.services.visualization import build_visualization_cmd
from bvr_marl_core.simulator import MapLimits
from bvr_marl_core.visualization.scenario_overlays import (
    build_scenario_overlay_drawables,
    build_scenario_status_lines,
    get_visualization_scenario,
    normalize_visualization_scenario,
    select_display_limits,
)
from bvr_marl_core.visualization.scenplotter.scenario_plotter import (
    compute_map_figure_size_inches,
)


def test_build_visualization_cmd_includes_scenario_override():
    cmd = build_visualization_cmd(mode="standard", scenario="oca_v_dca_scenario")

    assert "--scenario" in cmd
    assert "oca_v_dca_scenario" in cmd


def test_build_visualization_cmd_includes_manual_overlay_options():
    cmd = build_visualization_cmd(
        mode="standard",
        show_line_overlay=True,
        line_east_km=75.0,
        map_extents_mode="full",
    )

    assert "--show-line" in cmd
    assert "--line-east-km" in cmd
    assert "75.0" in cmd
    assert "--map-extents" in cmd
    assert "full" in cmd


def test_build_visualization_cmd_can_hide_map_text():
    cmd = build_visualization_cmd(mode="standard", show_text=False)

    assert "--no-show-text" in cmd


def test_oca_v_dca_scenario_overlay_builds_line_and_status_text():
    map_limits = SimpleNamespace(
        left_lon=-2.0,
        right_lon=2.0,
        bottom_lat=-1.0,
        top_lat=1.0,
    )

    drawables = build_scenario_overlay_drawables(map_limits, "oca_v_dca_scenario")
    status_lines = build_scenario_status_lines("oca_v_dca_scenario")
    scenario = get_visualization_scenario("OCA v DCA Scenario (Line of Engagement)")

    assert normalize_visualization_scenario(scenario.label) == "oca_v_dca_scenario"
    assert scenario.line_of_engagement_east_m == 0.0
    assert len(drawables) == 1
    assert drawables[0].points[0][0] == -1.0
    assert drawables[0].points[1][0] == 1.0
    assert drawables[0].points[0][1] == 0.0
    assert "Scenario: OCA v DCA Scenario" in status_lines[0]
    assert status_lines[1] == "LOE: E +0 km"


def test_manual_line_overlay_can_be_enabled_for_default_scenario():
    map_limits = SimpleNamespace(
        left_lon=-2.0,
        right_lon=2.0,
        bottom_lat=-1.0,
        top_lat=1.0,
    )

    drawables = build_scenario_overlay_drawables(
        map_limits,
        "default",
        force_line_overlay=True,
        line_of_engagement_east_m=50_000.0,
    )
    status_lines = build_scenario_status_lines(
        "default",
        force_line_overlay=True,
        line_of_engagement_east_m=50_000.0,
    )

    assert len(drawables) == 1
    assert drawables[0].points[0][1] == 50.0 / 111.0
    assert status_lines == ["LOE: E +50 km"]


def test_oca_v_dca_scenario_prefers_combat_map_limits_for_display():
    env = SimpleNamespace(
        map_limits="combat",
        full_map_limits="full",
    )

    assert select_display_limits(env, "oca_v_dca_scenario") == "combat"
    assert select_display_limits(env, "default") == "full"
    assert select_display_limits(env, "oca_v_dca_scenario", map_extents_mode="full") == "full"
    assert select_display_limits(env, "default", map_extents_mode="combat") == "combat"


def test_plotter_figure_size_tracks_rectangular_map_aspect_ratio():
    map_limits = MapLimits(
        left_lon=-200.0 / 111.0,
        bottom_lat=-100.0 / 111.0,
        right_lon=200.0 / 111.0,
        top_lat=100.0 / 111.0,
    )

    width_in, height_in = compute_map_figure_size_inches(map_limits, base_height_in=6.0)

    assert width_in == 12.0
    assert height_in == 6.0
