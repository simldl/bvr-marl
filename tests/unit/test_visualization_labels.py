from types import SimpleNamespace

from bvr_marl_core.visualization.scenplotter import MapLabel
from bvr_marl_core.visualization.scenplotter.label_manager import FighterLabelManager
from bvr_marl_core.visualization.scenplotter.video_generation import (
    _aircraft_label_text,
    _deterministic_focus,
)


class _Plotter:
    img_width = 400
    img_height = 300
    cfg = SimpleNamespace(sprites_info_font_size=10, sprites_info_spacing=26)

    @staticmethod
    def _get_image_xya(lat, lon, _yaw):
        return lon, lat, 0.0


def _unit(unit_id="fighter-1", *, lat=100.0, lon=100.0):
    return SimpleNamespace(
        id=unit_id,
        unit_kind="aircraft",
        position=SimpleNamespace(lat=lat, lon=lon, alt=9000.0),
        speed=343.0,
        remaining_missiles=4,
        sensor=SimpleNamespace(sensor_tracks=[]),
    )


def test_compact_label_has_only_identity_and_kinematic_line():
    text = _aircraft_label_text(_unit(), {"agent_name": "BLUE-1"}, detail="compact")

    assert text.splitlines() == ["BLUE-1 · SimpleNamespace", "9.0 km · M1.00"]
    assert "MSL" not in text


def test_selected_label_includes_weapon_and_track_details():
    unit = _unit()
    unit.sensor.sensor_tracks = [
        SimpleNamespace(engageable=True, suspect_deception=False),
        SimpleNamespace(engageable=False, suspect_deception=False),
    ]

    text = _aircraft_label_text(unit, {"agent_name": "BLUE-1"}, detail="full")

    assert len(text.splitlines()) == 3
    assert "MSL 4" in text
    assert "tracks 1" in text


def test_threat_label_is_single_line():
    text = _aircraft_label_text(_unit(), {"agent_name": "RED-1"}, detail="short")

    assert text == "RED-1 · SimpleNamespace"


def test_label_manager_offsets_overlapping_labels_and_draws_leaders():
    labels = [
        MapLabel(100, 100, f"F-{index}\n9 km · M1.0", unit_id=index, priority=index)
        for index in range(3)
    ]

    visible = FighterLabelManager().layout(labels, _Plotter())

    assert len(visible) == 3
    assert len({(label.offset_x, label.offset_y) for label in visible}) == 3
    assert any(label.draw_leader for label in visible)


def test_label_manager_collapses_dense_low_priority_group():
    labels = [
        MapLabel(
            100 + index,
            100 + index,
            f"BLUE-{index}",
            unit_id=index,
            priority=0,
            affiliation="friendly",
            cluster_name="F-22A",
        )
        for index in range(3)
    ]

    visible = FighterLabelManager().layout(labels, _Plotter())

    assert len(visible) == 1
    assert visible[0].text == "3× F-22A"


def test_video_nearest_threat_focus_is_deterministic():
    near_target = _unit("near", lat=10, lon=10)
    far_target = _unit("far", lat=50, lon=50)
    missiles = [
        SimpleNamespace(
            id="m2",
            unit_kind="missile",
            position=SimpleNamespace(lat=30, lon=30),
            target=far_target,
        ),
        SimpleNamespace(
            id="m1",
            unit_kind="missile",
            position=SimpleNamespace(lat=11, lon=11),
            target=near_target,
        ),
    ]

    assert _deterministic_focus([far_target, *missiles, near_target], "nearest_threat") == "near"
    assert _deterministic_focus([far_target, near_target], "none") is None
