import pytest

from bvr_marl_core.radar.scan_scheduler import ScanScheduler


def test_scheduler_covers_search_volume_with_explicit_revisit():
    scheduler = ScanScheduler(120.0, 40.0, sectors=4, dwell_duration_s=0.5)
    dwells = [scheduler.next_dwell(0.5) for _ in range(4)]

    assert [d.center_azimuth_offset_deg for d in dwells] == [-45.0, -15.0, 15.0, 45.0]
    assert all(d.horizontal_width_deg == 30.0 for d in dwells)
    assert all(d.revisit_interval_s == 2.0 for d in dwells)
    assert scheduler.next_dwell(0.5).center_azimuth_offset_deg == -45.0


def test_partial_macro_steps_do_not_accelerate_physical_scan():
    scheduler = ScanScheduler(120.0, 40.0, sectors=4, dwell_duration_s=1.0)

    dwells = [scheduler.next_dwell(0.25) for _ in range(8)]

    assert [d.sequence for d in dwells] == [0, 0, 0, 0, 1, 1, 1, 1]
    assert [d.duration_s for d in dwells] == [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    assert [d.center_azimuth_offset_deg for d in dwells[3::4]] == [-45.0, -15.0]
    assert all(d.revisit_interval_s == 4.0 for d in dwells)


def test_search_sector_admission_uses_independent_azimuth_and_elevation():
    scheduler = ScanScheduler(120.0, 40.0, sectors=4)
    dwell = scheduler.next_dwell(1.0)

    assert scheduler.admits(-45.0, 0.0, dwell)
    assert not scheduler.admits(0.0, 0.0, dwell)
    assert not scheduler.admits(-45.0, 30.0, dwell)
