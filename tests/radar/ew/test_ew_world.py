"""
Tests for EW (Electronic Warfare) World coordination functionality.
"""

import math

import numpy as np
import pytest

from bvr_marl_core.radar.ew.ecm_emitter import ECMEmitter
from bvr_marl_core.radar.ew.ew_world import EWWorld


class MockOwner:
    """Mock aircraft/unit owner."""

    def __init__(self, unit_id, group, x=0.0, y=0.0, z=5000.0):
        self.id = unit_id
        self.group = group
        # Convert x/y (meters) to approximate lat/lon (degrees)
        # At equator: 1 degree ≈ 111km
        # Use reference point at (45°N, 0°E) for testing
        ref_lat = 45.0
        ref_lon = 0.0
        lat = ref_lat + (y / 111000.0)  # North offset in degrees
        lon = ref_lon + (x / (111000.0 * math.cos(math.radians(ref_lat))))  # East offset

        self.position = type(
            "Pos", (), {"x": x, "y": y, "z": z, "lat": lat, "lon": lon, "alt": z}
        )()


class MockRadar:
    """Mock radar for testing."""

    def __init__(self, owner, jam_susceptible=True):
        self.owner = owner
        self.freq_hz = 10e9
        self.antenna_gain_db = 30.0
        self.jam_susceptible = jam_susceptible
        self.h_fov_deg = 120.0
        self.v_fov_deg = 60.0
        self.snr_threshold_db = 10.0


class MockSimulator:
    """Mock simulator for testing EWWorld."""

    def __init__(self):
        self.active_units = {}

    def add_unit(self, unit_id, unit):
        self.active_units[unit_id] = unit


class MockUnit:
    """Mock unit with ECM capability."""

    def __init__(self, unit_id, group, position, ecm=None):
        self.id = unit_id
        self.group = group
        self.position = position
        self.ecm = ecm


@pytest.fixture
def mock_sim():
    """Create a mock simulator."""
    return MockSimulator()


@pytest.fixture
def ew_world(mock_sim):
    """Create an EW world with mock simulator."""
    return EWWorld(mock_sim)


class TestEWWorldInitialization:
    """Test EW World initialization."""

    def test_basic_initialization(self, mock_sim):
        """Test that EW world initializes with simulator."""
        ew = EWWorld(mock_sim)
        assert ew.sim == mock_sim

    def test_empty_simulation(self, ew_world):
        """Test collect_incoming with no units."""
        victim_owner = MockOwner("victim1", "BLUE", x=0.0, y=0.0, z=5000.0)
        victim_radar = MockRadar(victim_owner)

        jammer_info, ghosts = ew_world.collect_incoming(victim_radar, t=0.0)

        # No jammers = empty results
        assert jammer_info == []
        assert ghosts == []


class TestMissileImmunity:
    """Test that missile seekers are immune to jamming."""

    def test_missile_seeker_immunity(self, ew_world, mock_sim):
        """Test that missile seekers return no jamming effects."""
        # Add enemy jammer
        jammer_owner = MockOwner("red1", "RED", x=5000.0, y=0.0, z=5000.0)
        jammer = ECMEmitter(
            owner=jammer_owner, erp_w=2000.0, bw_hz=2e6, techniques={"drfm_multi_false"}
        )
        unit_with_ecm = MockUnit("red1", "RED", jammer_owner.position, ecm=jammer)
        mock_sim.add_unit("red1", unit_with_ecm)

        # Create missile seeker (not susceptible to jamming)
        missile_owner = MockOwner("missile1", "BLUE", x=3000.0, y=0.0, z=5000.0)
        missile_radar = MockRadar(missile_owner, jam_susceptible=False)

        jammer_info, ghosts = ew_world.collect_incoming(missile_radar, t=0.0)

        # Missile seekers should be immune
        assert jammer_info == []
        assert ghosts == []


class TestSingleJammerScenarios:
    """Test scenarios with a single jammer."""

    def test_single_enemy_jammer(self, ew_world, mock_sim):
        """Test jamming from a single enemy jammer."""
        # Add RED jammer
        jammer_owner = MockOwner("red1", "RED", x=5000.0, y=0.0, z=5000.0)
        jammer = ECMEmitter(
            owner=jammer_owner,
            erp_w=2000.0,
            bw_hz=2e6,
            techniques={"drfm_multi_false"},
            np_rng=np.random.default_rng(42),
        )
        unit_with_ecm = MockUnit("red1", "RED", jammer_owner.position, ecm=jammer)
        mock_sim.add_unit("red1", unit_with_ecm)

        # BLUE victim radar
        victim_owner = MockOwner("blue1", "BLUE", x=0.0, y=0.0, z=5000.0)
        victim_radar = MockRadar(victim_owner)

        jammer_info, ghosts = ew_world.collect_incoming(victim_radar, t=0.0)

        # Should have jammer info
        assert len(jammer_info) == 1
        jammer_az, jammer_el, J_power = jammer_info[0]

        # Check jammer geometry (should be roughly east of victim)
        assert isinstance(jammer_az, float)
        assert isinstance(jammer_el, float)
        assert J_power > 0.0, "Jammer power should be positive"

        # Should have ghosts (DRFM generates 3-5 ghosts)
        assert isinstance(ghosts, list)
        # Ghosts may or may not be generated depending on FOV and randomization
        for ghost in ghosts:
            assert "az" in ghost
            assert "el" in ghost
            assert "d" in ghost
            assert "dop" in ghost
            assert "is_deception" in ghost
            assert ghost["is_deception"] is True
            assert "engagement_id" in ghost
            assert "jammer_id" in ghost
            assert ghost["T"] is None

    def test_friendly_jammer_no_effect(self, ew_world, mock_sim):
        """Test that friendly jammers don't affect same-group radars."""
        # Add BLUE jammer
        jammer_owner = MockOwner("blue_ecm", "BLUE", x=5000.0, y=0.0, z=5000.0)
        jammer = ECMEmitter(
            owner=jammer_owner, erp_w=2000.0, bw_hz=2e6, techniques={"drfm_multi_false"}
        )
        unit_with_ecm = MockUnit("blue_ecm", "BLUE", jammer_owner.position, ecm=jammer)
        mock_sim.add_unit("blue_ecm", unit_with_ecm)

        # BLUE victim radar (same group)
        victim_owner = MockOwner("blue1", "BLUE", x=0.0, y=0.0, z=5000.0)
        victim_radar = MockRadar(victim_owner)

        jammer_info, ghosts = ew_world.collect_incoming(victim_radar, t=0.0)

        # Friendly jammer should not affect same-group radar
        assert jammer_info == []
        assert ghosts == []


class TestMultipleJammerScenarios:
    """Test scenarios with multiple jammers."""

    def test_two_enemy_jammers(self, ew_world, mock_sim):
        """Test jamming from two enemy jammers."""
        # Add RED jammer 1
        jammer1_owner = MockOwner("red1", "RED", x=5000.0, y=0.0, z=5000.0)
        jammer1 = ECMEmitter(owner=jammer1_owner, erp_w=1500.0, bw_hz=1.5e6, techniques={"noise"})
        unit1 = MockUnit("red1", "RED", jammer1_owner.position, ecm=jammer1)
        mock_sim.add_unit("red1", unit1)

        # Add RED jammer 2
        jammer2_owner = MockOwner("red2", "RED", x=-5000.0, y=0.0, z=5000.0)
        jammer2 = ECMEmitter(
            owner=jammer2_owner,
            erp_w=1500.0,
            bw_hz=1.5e6,
            techniques={"drfm_multi_false"},
            np_rng=np.random.default_rng(123),
        )
        unit2 = MockUnit("red2", "RED", jammer2_owner.position, ecm=jammer2)
        mock_sim.add_unit("red2", unit2)

        # BLUE victim radar
        victim_owner = MockOwner("blue1", "BLUE", x=0.0, y=0.0, z=5000.0)
        victim_radar = MockRadar(victim_owner)

        jammer_info, ghosts = ew_world.collect_incoming(victim_radar, t=0.0)

        # Should have info from both jammers
        assert len(jammer_info) == 2

        # Check that both jammers contribute power
        for jammer_az, jammer_el, J_power in jammer_info:
            assert J_power > 0.0

        # Ghosts should only come from jammer2 (has DRFM)
        # jammer1 has only "noise" technique
        assert isinstance(ghosts, list)

    def test_mixed_friendly_enemy_jammers(self, ew_world, mock_sim):
        """Test that only enemy jammers affect the radar."""
        # Add BLUE friendly jammer
        blue_owner = MockOwner("blue_ecm", "BLUE", x=2000.0, y=0.0, z=5000.0)
        blue_jammer = ECMEmitter(owner=blue_owner, erp_w=3000.0, bw_hz=3e6)
        blue_unit = MockUnit("blue_ecm", "BLUE", blue_owner.position, ecm=blue_jammer)
        mock_sim.add_unit("blue_ecm", blue_unit)

        # Add RED enemy jammer
        red_owner = MockOwner("red1", "RED", x=-2000.0, y=0.0, z=5000.0)
        red_jammer = ECMEmitter(owner=red_owner, erp_w=1500.0, bw_hz=1.5e6)
        red_unit = MockUnit("red1", "RED", red_owner.position, ecm=red_jammer)
        mock_sim.add_unit("red1", red_unit)

        # BLUE victim radar
        victim_owner = MockOwner("blue1", "BLUE", x=0.0, y=0.0, z=5000.0)
        victim_radar = MockRadar(victim_owner)

        jammer_info, ghosts = ew_world.collect_incoming(victim_radar, t=0.0)

        # Only RED jammer should affect BLUE radar
        assert len(jammer_info) == 1


class TestJammerGeometry:
    """Test jammer angle calculations."""

    def test_jammer_directly_east(self, ew_world, mock_sim):
        """Test jammer positioned directly east."""
        # Jammer 5km east
        jammer_owner = MockOwner("red1", "RED", x=5000.0, y=0.0, z=5000.0)
        jammer = ECMEmitter(owner=jammer_owner, erp_w=1000.0, bw_hz=1e6)
        unit = MockUnit("red1", "RED", jammer_owner.position, ecm=jammer)
        mock_sim.add_unit("red1", unit)

        # Victim at origin
        victim_owner = MockOwner("blue1", "BLUE", x=0.0, y=0.0, z=5000.0)
        victim_radar = MockRadar(victim_owner)

        jammer_info, _ = ew_world.collect_incoming(victim_radar, t=0.0)

        jammer_az, jammer_el, _ = jammer_info[0]
        # Mathematical azimuth: 0° = East, CCW positive (atan2(N, E))
        # Jammer is directly east, so azimuth should be ~0°
        assert abs(jammer_az) < 5.0
        # Same altitude, so elevation ~0
        assert abs(jammer_el) < 5.0

    def test_jammer_above_victim(self, ew_world, mock_sim):
        """Test jammer positioned above victim."""
        # Jammer 3km above, 5km away
        jammer_owner = MockOwner("red1", "RED", x=5000.0, y=0.0, z=8000.0)
        jammer = ECMEmitter(owner=jammer_owner, erp_w=1000.0, bw_hz=1e6)
        unit = MockUnit("red1", "RED", jammer_owner.position, ecm=jammer)
        mock_sim.add_unit("red1", unit)

        # Victim at lower altitude
        victim_owner = MockOwner("blue1", "BLUE", x=0.0, y=0.0, z=5000.0)
        victim_radar = MockRadar(victim_owner)

        jammer_info, _ = ew_world.collect_incoming(victim_radar, t=0.0)

        jammer_az, jammer_el, _ = jammer_info[0]
        # Jammer is above, so positive elevation
        assert jammer_el > 0.0


class TestJammerPowerCalculation:
    """Test jammer power calculations."""

    def test_jammer_power_decreases_with_range(self, ew_world, mock_sim):
        """Test that jammer power follows inverse square law."""
        # Create victim
        victim_owner = MockOwner("blue1", "BLUE", x=0.0, y=0.0, z=5000.0)
        victim_radar = MockRadar(victim_owner)

        # Test at 5km
        jammer1_owner = MockOwner("red1", "RED", x=5000.0, y=0.0, z=5000.0)
        jammer1 = ECMEmitter(owner=jammer1_owner, erp_w=2000.0, bw_hz=2e6)
        unit1 = MockUnit("red1", "RED", jammer1_owner.position, ecm=jammer1)
        mock_sim.add_unit("red1", unit1)

        jammer_info_5km, _ = ew_world.collect_incoming(victim_radar, t=0.0)
        J_5km = jammer_info_5km[0][2]

        # Reset and test at 10km
        mock_sim.active_units.clear()
        jammer2_owner = MockOwner("red2", "RED", x=10000.0, y=0.0, z=5000.0)
        jammer2 = ECMEmitter(owner=jammer2_owner, erp_w=2000.0, bw_hz=2e6)
        unit2 = MockUnit("red2", "RED", jammer2_owner.position, ecm=jammer2)
        mock_sim.add_unit("red2", unit2)

        jammer_info_10km, _ = ew_world.collect_incoming(victim_radar, t=0.0)
        J_10km = jammer_info_10km[0][2]

        # Power at 5km should be ~4x power at 10km (inverse square law)
        ratio = J_5km / J_10km
        assert 3.5 < ratio < 4.5, f"Expected ~4x ratio, got {ratio}"


class TestDeceptionGeneration:
    """Test deception (ghost) generation."""

    def test_deception_ghosts_have_correct_fields(self, ew_world, mock_sim):
        """Test that deception ghosts have all required fields."""
        # Add jammer with DRFM
        jammer_owner = MockOwner("red1", "RED", x=5000.0, y=0.0, z=5000.0)
        jammer = ECMEmitter(
            owner=jammer_owner,
            erp_w=2000.0,
            bw_hz=2e6,
            techniques={"drfm_multi_false"},
            np_rng=np.random.default_rng(42),
        )
        unit = MockUnit("red1", "RED", jammer_owner.position, ecm=jammer)
        mock_sim.add_unit("red1", unit)

        # Victim radar
        victim_owner = MockOwner("blue1", "BLUE", x=0.0, y=0.0, z=5000.0)
        victim_radar = MockRadar(victim_owner)

        _, ghosts = ew_world.collect_incoming(victim_radar, t=0.0)

        # Check ghost structure
        for ghost in ghosts:
            assert isinstance(ghost, dict)
            assert "az" in ghost
            assert "el" in ghost
            assert "d" in ghost
            assert "dop" in ghost
            assert "snr_db" in ghost
            assert "is_deception" in ghost
            assert ghost["is_deception"] is True
            assert "engagement_id" in ghost
            assert ghost["engagement_id"] == jammer_owner.id
            assert "jammer_id" in ghost
            assert ghost["jammer_id"] == jammer_owner.id
            assert "T" in ghost
            assert ghost["T"] is None

    def test_no_deception_without_drfm(self, ew_world, mock_sim):
        """Test that jammers without DRFM don't generate ghosts."""
        # Add jammer with only noise technique
        jammer_owner = MockOwner("red1", "RED", x=5000.0, y=0.0, z=5000.0)
        jammer = ECMEmitter(
            owner=jammer_owner,
            erp_w=2000.0,
            bw_hz=2e6,
            techniques={"noise"},  # No DRFM
        )
        unit = MockUnit("red1", "RED", jammer_owner.position, ecm=jammer)
        mock_sim.add_unit("red1", unit)

        # Victim radar
        victim_owner = MockOwner("blue1", "BLUE", x=0.0, y=0.0, z=5000.0)
        victim_radar = MockRadar(victim_owner)

        _, ghosts = ew_world.collect_incoming(victim_radar, t=0.0)

        # Should have no ghosts (no DRFM)
        assert len(ghosts) == 0


class TestAssistScheduling:
    """Test cooperative jamming assist scheduling."""

    def test_schedule_assist_no_crash(self, ew_world, mock_sim):
        """Test that schedule_assist doesn't crash with valid inputs."""
        # Create radar with ECM
        radar_owner = MockOwner("blue1", "BLUE", x=0.0, y=0.0, z=5000.0)
        ecm = ECMEmitter(owner=radar_owner, erp_w=1000.0, bw_hz=1e6)
        radar_owner.ecm = ecm
        radar = MockRadar(radar_owner)

        # Create enemy tracks (mock format)
        enemy_tracks = []

        # Should not crash
        ew_world.schedule_assist(
            own_radar=radar, team_radars=[], enemy_tracks=enemy_tracks, t=0.0, K=2
        )

    def test_schedule_assist_with_tracks(self, ew_world, mock_sim):
        """Test assist scheduling with actual tracks."""
        # Create radar with ECM
        radar_owner = MockOwner("blue1", "BLUE", x=0.0, y=0.0, z=5000.0)
        ecm = ECMEmitter(owner=radar_owner, erp_w=1000.0, bw_hz=1e6)
        radar_owner.ecm = ecm
        radar = MockRadar(radar_owner)

        # Create mock enemy tracks (simplified tuple format)
        tid = 1
        state = np.array([5000.0, 0.0, 5000.0, 0.0, 0.0, 0.0])
        cov = np.eye(6)
        tgt = type("Target", (), {"id": "red1"})()
        utype = "aircraft"
        ref = type("Ref", (), {})()
        confidence = 0.9
        n_obs = 10
        lifetime = 10
        update_count = 10
        is_deception = False
        suspect_deception = False
        engagement_id = None
        jammer_id = None
        engageable = True

        track = (
            tid,
            state,
            cov,
            tgt,
            utype,
            ref,
            confidence,
            n_obs,
            lifetime,
            update_count,
            is_deception,
            suspect_deception,
            engagement_id,
            jammer_id,
            engageable,
        )

        enemy_tracks = [track]

        # Should not crash
        ew_world.schedule_assist(
            own_radar=radar, team_radars=[], enemy_tracks=enemy_tracks, t=0.0, K=2
        )


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_radar_without_owner(self, ew_world):
        """Test handling radar without owner."""
        radar = type("Radar", (), {"jam_susceptible": True})()
        # No owner attribute

        jammer_info, ghosts = ew_world.collect_incoming(radar, t=0.0)

        # Should handle gracefully
        assert jammer_info == []
        assert ghosts == []

    def test_owner_without_position(self, ew_world):
        """Test handling owner without position."""
        # Create owner without position attribute
        owner = type("Owner", (), {"id": "test", "group": "BLUE"})()
        # Create radar manually without using MockRadar to avoid position creation
        radar = type("Radar", (), {"owner": owner, "jam_susceptible": True, "freq_hz": 10e9})()

        jammer_info, ghosts = ew_world.collect_incoming(radar, t=0.0)

        # Should handle gracefully
        assert jammer_info == []
        assert ghosts == []

    def test_unit_without_ecm(self, ew_world, mock_sim):
        """Test that units without ECM are ignored."""
        # Add unit without ECM
        unit_owner = MockOwner("red1", "RED", x=5000.0, y=0.0, z=5000.0)
        unit = MockUnit("red1", "RED", unit_owner.position, ecm=None)
        mock_sim.add_unit("red1", unit)

        # Victim radar
        victim_owner = MockOwner("blue1", "BLUE", x=0.0, y=0.0, z=5000.0)
        victim_radar = MockRadar(victim_owner)

        jammer_info, ghosts = ew_world.collect_incoming(victim_radar, t=0.0)

        # No ECM = no jamming
        assert jammer_info == []
        assert ghosts == []


class TestRealisticScenarios:
    """Test realistic EW engagement scenarios."""

    def test_strike_package_with_escort_jammer(self, ew_world, mock_sim):
        """Test escort jammer protecting strike package."""
        # Escort jammer aircraft
        escort_owner = MockOwner("blue_escort", "BLUE", x=2000.0, y=0.0, z=7000.0)
        escort_jammer = ECMEmitter(
            owner=escort_owner,
            erp_w=5000.0,  # Powerful escort jammer
            bw_hz=10e6,
            techniques={"noise", "drfm_multi_false"},
            np_rng=np.random.default_rng(789),
        )
        escort = MockUnit("blue_escort", "BLUE", escort_owner.position, ecm=escort_jammer)
        mock_sim.add_unit("blue_escort", escort)

        # Enemy SAM radar
        sam_owner = MockOwner("red_sam", "RED", x=10000.0, y=0.0, z=100.0)
        sam_radar = MockRadar(sam_owner)

        jammer_info, ghosts = ew_world.collect_incoming(sam_radar, t=0.0)

        # Escort should jam the SAM radar
        assert len(jammer_info) == 1
        _, _, J_power = jammer_info[0]
        assert J_power > 0.0

        # DRFM should generate ghosts
        assert isinstance(ghosts, list)

    def test_mutual_jamming_scenario(self, ew_world, mock_sim):
        """Test scenario where both sides have jammers."""
        # BLUE jammer
        blue_owner = MockOwner("blue1", "BLUE", x=-3000.0, y=0.0, z=6000.0)
        blue_jammer = ECMEmitter(
            owner=blue_owner, erp_w=2000.0, bw_hz=2e6, techniques={"drfm_multi_false"}
        )
        blue_unit = MockUnit("blue1", "BLUE", blue_owner.position, ecm=blue_jammer)
        mock_sim.add_unit("blue1", blue_unit)

        # RED jammer
        red_owner = MockOwner("red1", "RED", x=3000.0, y=0.0, z=6000.0)
        red_jammer = ECMEmitter(
            owner=red_owner, erp_w=2000.0, bw_hz=2e6, techniques={"drfm_multi_false"}
        )
        red_unit = MockUnit("red1", "RED", red_owner.position, ecm=red_jammer)
        mock_sim.add_unit("red1", red_unit)

        # BLUE radar sees RED jammer
        blue_radar_owner = MockOwner("blue_radar", "BLUE", x=-5000.0, y=0.0, z=5000.0)
        blue_radar = MockRadar(blue_radar_owner)
        blue_jammer_info, blue_ghosts = ew_world.collect_incoming(blue_radar, t=0.0)
        assert len(blue_jammer_info) == 1  # Only RED jammer affects BLUE

        # RED radar sees BLUE jammer
        red_radar_owner = MockOwner("red_radar", "RED", x=5000.0, y=0.0, z=5000.0)
        red_radar = MockRadar(red_radar_owner)
        red_jammer_info, red_ghosts = ew_world.collect_incoming(red_radar, t=0.0)
        assert len(red_jammer_info) == 1  # Only BLUE jammer affects RED
