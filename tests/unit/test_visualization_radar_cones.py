"""Regression tests for radar coverage shown by the 2D visualizer."""

from types import SimpleNamespace

from bvr_marl_core.visualization.scenplotter.scenario_plotter import ScenarioPlotter
from bvr_marl_core.visualization.scenplotter.video_generation import plot_radar_cone


def _unit(*, support: bool, yaw_deg: float = 45.0):
    radar = SimpleNamespace(max_range_m=300_000.0, h_fov_deg=120.0)
    return SimpleNamespace(
        radar=radar,
        is_support_asset=support,
        yaw_deg=yaw_deg,
        position=SimpleNamespace(lat=1.0, lon=2.0),
    )


def test_awacs_radar_cone_is_full_circle():
    cone = plot_radar_cone(_unit(support=True), "blue")[0]

    assert cone.angle1 == 0.0
    assert cone.angle2 == 360.0
    assert cone.angle2 - cone.angle1 == 360.0
    assert cone.radius == 250.0


def test_fighter_radar_cone_remains_directional():
    cone = plot_radar_cone(_unit(support=False), "blue")[0]

    assert cone.angle1 == 345.0
    assert cone.angle2 == 105.0


def test_fighter_cone_radius_shows_instrumented_range_even_against_stealth():
    unit = _unit(support=False)
    unit.radar.lut = SimpleNamespace(get_probability=lambda _range, _rcs: 0.01)
    unit.radar.max_range_m = 400_000.0

    cone = plot_radar_cone(unit, "red", enemy_rcs=0.0001)[0]

    assert cone.radius == 400.0


def test_full_circle_radar_drawing_has_no_radial_seam():
    class RecordingContext:
        def __init__(self):
            self.calls = []

        def __getattr__(self, name):
            return lambda *args: self.calls.append((name, args))

    plotter = SimpleNamespace(
        _get_image_xya=lambda *_args: (100.0, 100.0, 0.0),
        _get_image_distance=lambda radius: radius,
    )
    cone = plot_radar_cone(_unit(support=True), "blue")[0]
    ctx = RecordingContext()

    ScenarioPlotter._draw_radar_cone(plotter, ctx, cone)

    names = [name for name, _args in ctx.calls]
    assert "arc" in names
    assert "arc_negative" not in names
    assert "move_to" not in names
    assert "line_to" not in names
