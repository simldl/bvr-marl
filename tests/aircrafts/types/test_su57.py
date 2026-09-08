from bvr_marl_core.aircraft.types.su57 import Su57


def test_su57_radar_uses_reported_search_range_ceiling():
    assert Su57.Config().radar_max_range_m == 400_000.0
