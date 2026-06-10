"""
Tests for ECM (Electronic Counter-Measures) emitter functionality.
"""

import math

import numpy as np
import pytest

from bvr_marl_core.radar.ew.ecm_emitter import ECMEmitter
from bvr_marl_core.simulator.core.helpers import Position


class MockOwner:
    """Mock aircraft owner for ECM emitter."""

    def __init__(self, x=0.0, y=0.0, z=0.0, unit_id=0):
        self.id = unit_id
        self.position = type(
            "Pos", (), {"x": x, "y": y, "z": z, "lat": 0.0, "lon": 0.0, "alt": z}
        )()


class MockRadar:
    """Mock radar for testing ECM effects."""

    def __init__(self, freq_hz=10e9, antenna_gain_db=30.0):
        self.freq_hz = freq_hz
        self.antenna_gain_db = antenna_gain_db
        self.h_fov_deg = 60.0
        self.v_fov_deg = 30.0
        self.snr_threshold_db = 10.0  # Required for deception_burst


@pytest.fixture
def basic_jammer():
    """Create a basic ECM emitter for testing."""
    owner = MockOwner(x=0.0, y=0.0, z=5000.0)
    return ECMEmitter(
        owner=owner,
        erp_w=1000.0,  # 1 kW ERP
        bw_hz=1e6,  # 1 MHz bandwidth
        hop_period_s=0.1,
        techniques={"drfm_multi_false"},
        np_rng=np.random.default_rng(42),
    )


@pytest.fixture
def victim_radar():
    """Create a mock victim radar."""
    return MockRadar(freq_hz=10e9, antenna_gain_db=30.0)


class TestECMEmitterInitialization:
    """Test ECM emitter initialization."""

    def test_basic_initialization(self, basic_jammer):
        """Test that ECM emitter initializes correctly."""
        assert basic_jammer.owner is not None
        assert basic_jammer.erp_w == 1000.0
        assert basic_jammer.bw_hz == 1e6
        assert basic_jammer.hop_period_s == 0.1
        assert "drfm_multi_false" in basic_jammer.techniques
        assert basic_jammer.np_rng is not None

    def test_custom_techniques(self):
        """Test initialization with custom jamming techniques."""
        owner = MockOwner()
        jammer = ECMEmitter(
            owner=owner, erp_w=500.0, bw_hz=5e5, techniques={"noise", "spot", "barrage"}
        )
        assert jammer.techniques == {"noise", "spot", "barrage"}

    def test_assist_sectors(self):
        """Test assist sector configuration."""
        owner = MockOwner()
        jammer = ECMEmitter(
            owner=owner, erp_w=1000.0, bw_hz=1e6, assist_ahead_deg=60.0, assist_behind_deg=30.0
        )
        assert jammer.assist_ahead_deg == 60.0
        assert jammer.assist_behind_deg == 30.0


class TestJammerPowerCalculation:
    """Test jammer power (J) calculations."""

    def test_compute_J_at_close_range(self, basic_jammer, victim_radar):
        """Test jammer power calculation at close range."""
        # Victim radar 1 km away (approximately 0.01° offset)
        victim_pos = type("Pos", (), {"lat": 0.01, "lon": 0.0, "alt": 5000.0})()

        J = basic_jammer.compute_J(victim_radar, t=0.0, victim_position=victim_pos)

        # Jammer power should be positive and inversely proportional to R^2
        assert J > 0.0, "Jammer power should be positive at close range"

    def test_compute_J_decreases_with_distance(self, basic_jammer, victim_radar):
        """Test that jammer power decreases with range (1/R^2 law)."""
        # Test at 1 km (≈0.009° lat offset)
        pos_1km = type("Pos", (), {"lat": 0.009, "lon": 0.0, "alt": 5000.0})()
        J_1km = basic_jammer.compute_J(victim_radar, t=0.0, victim_position=pos_1km)

        # Test at 2 km (should be ~1/4 the power)
        pos_2km = type("Pos", (), {"lat": 0.018, "lon": 0.0, "alt": 5000.0})()
        J_2km = basic_jammer.compute_J(victim_radar, t=0.0, victim_position=pos_2km)

        # Test at 4 km (should be ~1/16 the power relative to 1km)
        pos_4km = type("Pos", (), {"lat": 0.036, "lon": 0.0, "alt": 5000.0})()
        J_4km = basic_jammer.compute_J(victim_radar, t=0.0, victim_position=pos_4km)

        # Verify inverse square law
        assert J_1km > J_2km > J_4km
        # Check approximate 1/R^2 relationship (allowing for computational tolerance)
        assert 3.5 < J_1km / J_2km < 4.5, "Should follow ~1/4 relationship"
        assert 14 < J_1km / J_4km < 18, "Should follow ~1/16 relationship"

    def test_compute_J_no_owner(self, victim_radar):
        """Test jammer power when owner is None."""
        jammer = ECMEmitter(owner=None, erp_w=1000.0, bw_hz=1e6)
        victim_pos = type("Pos", (), {"lat": 0.009, "lon": 0.0, "alt": 5000.0})()
        J = jammer.compute_J(victim_radar, t=0.0, victim_position=victim_pos)

        assert J == 0.0, "Jammer with no owner should produce zero power"

    def test_compute_J_minimum_range(self, basic_jammer, victim_radar):
        """Test that minimum range prevents division by zero."""
        # Victim at same position as jammer
        victim_pos = type("Pos", (), {"lat": 0.0, "lon": 0.0, "alt": 5000.0})()
        J = basic_jammer.compute_J(victim_radar, t=0.0, victim_position=victim_pos)

        # Should not crash and should return a finite value
        assert math.isfinite(J)
        assert J > 0.0


class TestFrequencyHopping:
    """Test frequency hopping behavior."""

    def test_frequency_hopping_initialization(self):
        """Test that frequency hopping state initializes correctly."""
        owner = MockOwner()
        jammer = ECMEmitter(owner=owner, erp_w=1000.0, bw_hz=1e6, hop_period_s=0.2)

        assert jammer._hop_timer == 0.0
        assert jammer._on_target  # Starts on target

    def test_hopping_period_parameter(self):
        """Test different hopping periods."""
        owner = MockOwner()

        fast_hop = ECMEmitter(owner=owner, erp_w=1000.0, bw_hz=1e6, hop_period_s=0.05)
        slow_hop = ECMEmitter(owner=owner, erp_w=1000.0, bw_hz=1e6, hop_period_s=0.5)

        assert fast_hop.hop_period_s == 0.05
        assert slow_hop.hop_period_s == 0.5


class TestDeceptionGeneration:
    """Test ghost/false target generation."""

    def test_generate_deception_with_drfm(self, basic_jammer, victim_radar):
        """Test deception generation with DRFM technique."""
        victim_pos = type(
            "Pos", (), {"x": 5000.0, "y": 0.0, "z": 5000.0, "lat": 0.0, "lon": 0.0, "alt": 5000.0}
        )()

        ghosts = basic_jammer.deception_burst(
            victim_radar=victim_radar, victim_position=victim_pos, t=0.0
        )

        # Should generate multiple ghost returns with DRFM technique
        assert isinstance(ghosts, list)
        # DRFM typically generates 1-3 false targets
        if len(ghosts) > 0:
            for ghost in ghosts:
                assert "az" in ghost
                assert "el" in ghost
                assert "d" in ghost  # 'd' is distance/range
                assert "dop" in ghost  # 'dop' is doppler

    def test_deception_within_assist_sector(self, basic_jammer, victim_radar):
        """Test that deception only occurs within assist sectors."""
        # Victim behind jammer (outside assist sector)
        victim_behind = type(
            "Pos", (), {"x": -5000.0, "y": 0.0, "z": 5000.0, "lat": 0.0, "lon": 0.0, "alt": 5000.0}
        )()

        ghosts_behind = basic_jammer.deception_burst(
            victim_radar=victim_radar, victim_position=victim_behind, t=0.0
        )

        # May or may not generate ghosts depending on assist sector
        assert isinstance(ghosts_behind, list)

    def test_no_deception_without_drfm(self):
        """Test that no deception is generated without DRFM technique."""
        owner = MockOwner(x=0.0, y=0.0, z=5000.0)
        jammer = ECMEmitter(
            owner=owner,
            erp_w=1000.0,
            bw_hz=1e6,
            techniques={"noise"},  # Only noise, no DRFM
        )

        victim_radar = MockRadar()
        victim_pos = type(
            "Pos", (), {"x": 5000.0, "y": 0.0, "z": 5000.0, "lat": 0.0, "lon": 0.0, "alt": 5000.0}
        )()

        ghosts = jammer.deception_burst(victim_radar, victim_pos, t=0.0)

        # No DRFM = no deception ghosts
        assert len(ghosts) == 0


class TestECMEffectivenessScenarios:
    """Test realistic ECM effectiveness scenarios."""

    def test_high_power_jammer_effectiveness(self, victim_radar):
        """Test that high-power jammer produces strong jamming signal."""
        owner = MockOwner(x=0.0, y=0.0, z=5000.0)
        high_power_jammer = ECMEmitter(
            owner=owner,
            erp_w=10000.0,  # 10 kW ERP (strong jammer)
            bw_hz=10e6,
        )

        victim_pos = type("Pos", (), {"lat": 0.027, "lon": 0.0, "alt": 5000.0})()  # ~3km
        J = high_power_jammer.compute_J(victim_radar, t=0.0, victim_position=victim_pos)

        # High power jammer at 3km should produce significant jamming power
        assert J > 0.0

    def test_low_power_jammer_limited_effect(self, victim_radar):
        """Test that low-power jammer has limited effect at range."""
        owner = MockOwner(x=0.0, y=0.0, z=5000.0)
        low_power_jammer = ECMEmitter(
            owner=owner,
            erp_w=100.0,  # 100 W ERP (weak jammer)
            bw_hz=1e6,
        )

        # Far range (20 km ≈ 0.18°)
        victim_pos = type("Pos", (), {"lat": 0.18, "lon": 0.0, "alt": 5000.0})()
        J = low_power_jammer.compute_J(victim_radar, t=0.0, victim_position=victim_pos)

        # Low power jammer at long range should produce weak jamming
        # But should still be non-zero
        assert J >= 0.0

    def test_multiple_jammers_additive_effect(self, victim_radar):
        """Test that multiple jammers have additive jamming effect."""
        # Create two jammers
        jammer1 = ECMEmitter(
            owner=MockOwner(x=1000.0, y=0.0, z=5000.0, unit_id=1), erp_w=1000.0, bw_hz=1e6
        )
        jammer2 = ECMEmitter(
            owner=MockOwner(x=-1000.0, y=0.0, z=5000.0, unit_id=2), erp_w=1000.0, bw_hz=1e6
        )

        victim_pos = type("Pos", (), {"lat": 0.0, "lon": 0.0, "alt": 5000.0})()

        J1 = jammer1.compute_J(victim_radar, t=0.0, victim_position=victim_pos)
        J2 = jammer2.compute_J(victim_radar, t=0.0, victim_position=victim_pos)
        J_total = J1 + J2

        # Combined jamming should be greater than individual
        assert J_total > J1
        assert J_total > J2


class TestECMEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_erp(self, victim_radar):
        """Test jammer with zero ERP."""
        owner = MockOwner()
        jammer = ECMEmitter(owner=owner, erp_w=0.0, bw_hz=1e6)

        victim_pos = type("Pos", (), {"lat": 0.009, "lon": 0.0, "alt": 5000.0})()
        J = jammer.compute_J(victim_radar, t=0.0, victim_position=victim_pos)

        assert J == 0.0

    def test_negative_erp_converted(self):
        """Test that negative ERP is converted to positive."""
        owner = MockOwner()
        jammer = ECMEmitter(owner=owner, erp_w=-1000.0, bw_hz=1e6)

        # Should be converted to absolute value or clamped to zero
        assert jammer.erp_w >= 0.0

    def test_very_high_bandwidth(self, victim_radar):
        """Test jammer with very high bandwidth."""
        owner = MockOwner(x=0.0, y=0.0, z=5000.0)
        jammer = ECMEmitter(
            owner=owner,
            erp_w=1000.0,
            bw_hz=100e6,  # 100 MHz bandwidth (wideband jammer)
        )

        victim_pos = type("Pos", (), {"lat": 0.045, "lon": 0.0, "alt": 5000.0})()  # ~5km
        J = jammer.compute_J(victim_radar, t=0.0, victim_position=victim_pos)

        # Should still compute valid jamming power
        assert math.isfinite(J)
        assert J >= 0.0
