"""
Tests for geometry_sampler.py - Controlled scenario geometry.
"""

import numpy as np
import pytest

from bvr_marl_core.rl.environment.scenarios.geometry_sampler import (
    STUDY_REGIMES,
    GeometryConfig,
    GeometrySampler,
)


class TestGeometryConfig:
    """Tests for GeometryConfig dataclass."""

    def test_default_values(self):
        """Should have sensible default values."""
        config = GeometryConfig(
            range_m=100000.0,
            aspect="hot",
            altitude_split="co_alt",
        )
        assert config.range_m == 100000.0
        assert config.aspect == "hot"
        assert config.altitude_split == "co_alt"
        assert config.offset_deg == 0.0
        assert config.agent_altitude_m == 8000.0
        assert config.opponent_altitude_m == 8000.0
        assert config.agent_speed_mps == 250.0
        assert config.opponent_speed_mps == 250.0

    def test_custom_values(self):
        """Should accept custom values."""
        config = GeometryConfig(
            range_m=150000.0,
            aspect="beam",
            altitude_split="high_low",
            offset_deg=30.0,
            agent_altitude_m=12000.0,
            opponent_altitude_m=5000.0,
            agent_speed_mps=300.0,
            opponent_speed_mps=200.0,
        )
        assert config.range_m == 150000.0
        assert config.aspect == "beam"
        assert config.altitude_split == "high_low"
        assert config.offset_deg == 30.0
        assert config.agent_altitude_m == 12000.0
        assert config.opponent_altitude_m == 5000.0


class TestStudyRegimes:
    """Tests for predefined study regimes."""

    def test_hot_80km_regime(self):
        """hot_80km regime should have correct values."""
        regime = STUDY_REGIMES["hot_80km"]
        assert regime.range_m == 80000.0
        assert regime.aspect == "hot"
        assert regime.altitude_split == "co_alt"
        assert regime.offset_deg == 0.0

    def test_hot_120km_regime(self):
        """hot_120km regime should have correct values."""
        regime = STUDY_REGIMES["hot_120km"]
        assert regime.range_m == 120000.0
        assert regime.aspect == "hot"

    def test_hot_160km_regime(self):
        """hot_160km regime should have correct values."""
        regime = STUDY_REGIMES["hot_160km"]
        assert regime.range_m == 160000.0
        assert regime.aspect == "hot"

    def test_offset_30deg_regime(self):
        """offset_30deg regime should have correct offset."""
        regime = STUDY_REGIMES["offset_30deg"]
        assert regime.offset_deg == 30.0
        assert regime.aspect == "hot"

    def test_high_low_split_regime(self):
        """high_low_split should have altitude difference."""
        regime = STUDY_REGIMES["high_low_split"]
        assert regime.agent_altitude_m > regime.opponent_altitude_m
        assert regime.altitude_split == "high_low"

    def test_low_high_split_regime(self):
        """low_high_split should have opposite altitude difference."""
        regime = STUDY_REGIMES["low_high_split"]
        assert regime.agent_altitude_m < regime.opponent_altitude_m
        assert regime.altitude_split == "low_high"

    def test_beam_regime(self):
        """beam_100km should have beam aspect."""
        regime = STUDY_REGIMES["beam_100km"]
        assert regime.aspect == "beam"
        assert regime.range_m == 100000.0

    def test_cold_regime(self):
        """cold_60km should have cold aspect."""
        regime = STUDY_REGIMES["cold_60km"]
        assert regime.aspect == "cold"
        assert regime.range_m == 60000.0

    def test_all_regimes_have_required_fields(self):
        """All regimes should have all required fields."""
        for name, regime in STUDY_REGIMES.items():
            assert hasattr(regime, "range_m"), f"{name} missing range_m"
            assert hasattr(regime, "aspect"), f"{name} missing aspect"
            assert hasattr(regime, "altitude_split"), f"{name} missing altitude_split"
            assert hasattr(regime, "offset_deg"), f"{name} missing offset_deg"


class TestGeometrySampler:
    """Tests for GeometrySampler class."""

    def test_init_default(self):
        """Should initialize with default map size."""
        sampler = GeometrySampler()
        assert sampler.map_size_km == 200.0

    def test_init_custom_map_size(self):
        """Should accept custom map size."""
        sampler = GeometrySampler(map_size_km=300.0)
        assert sampler.map_size_km == 300.0

    def test_sample_valid_regime(self):
        """Should return geometry for valid regime."""
        sampler = GeometrySampler()
        geometry = sampler.sample("hot_120km")

        assert isinstance(geometry, GeometryConfig)
        assert geometry.range_m == 120000.0
        assert geometry.aspect == "hot"

    def test_sample_invalid_regime_raises(self):
        """Should raise ValueError for invalid regime."""
        sampler = GeometrySampler()
        with pytest.raises(ValueError) as exc_info:
            sampler.sample("invalid_regime")
        assert "invalid_regime" in str(exc_info.value)

    def test_list_regimes(self):
        """Should list all available regimes."""
        regimes = GeometrySampler.list_regimes()

        assert "hot_80km" in regimes
        assert "hot_120km" in regimes
        assert "offset_30deg" in regimes
        assert len(regimes) == len(STUDY_REGIMES)


class TestComputePositions:
    """Tests for compute_positions method."""

    def test_compute_basic_positions(self):
        """Should compute positions for basic scenario."""
        sampler = GeometrySampler()
        geometry = sampler.sample("hot_120km")
        result = sampler.compute_positions(geometry, num_agents=2, num_opponents=2)

        assert "agent_positions" in result
        assert "opponent_positions" in result
        assert "agent_headings" in result
        assert "opponent_headings" in result
        assert len(result["agent_positions"]) == 2
        assert len(result["opponent_positions"]) == 2

    def test_agents_on_west_side(self):
        """Agents should spawn on west (negative longitude) side."""
        sampler = GeometrySampler()
        geometry = sampler.sample("hot_120km")
        result = sampler.compute_positions(geometry)

        for pos in result["agent_positions"]:
            assert pos.lon < 0, "Agents should be on west side"

    def test_opponents_on_east_side(self):
        """Opponents should spawn on east (positive longitude) side."""
        sampler = GeometrySampler()
        geometry = sampler.sample("hot_120km")
        result = sampler.compute_positions(geometry)

        for pos in result["opponent_positions"]:
            assert pos.lon > 0, "Opponents should be on east side"

    def test_hot_headings(self):
        """Hot engagement should have head-on headings."""
        sampler = GeometrySampler()
        geometry = sampler.sample("hot_120km")
        result = sampler.compute_positions(geometry)

        # Agents should face east (90 deg), opponents west (270 deg)
        for heading in result["agent_headings"]:
            assert abs(heading - 90.0) < 1.0 or abs(heading - 90.0 - 360) < 1.0

        for heading in result["opponent_headings"]:
            assert abs(heading - 270.0) < 1.0

    def test_altitude_set_correctly(self):
        """Altitudes should match geometry config."""
        sampler = GeometrySampler()
        geometry = sampler.sample("high_low_split")
        result = sampler.compute_positions(geometry)

        for pos in result["agent_positions"]:
            assert pos.alt == geometry.agent_altitude_m

        for pos in result["opponent_positions"]:
            assert pos.alt == geometry.opponent_altitude_m

    def test_formation_spread(self):
        """Formation members should be spread laterally."""
        sampler = GeometrySampler()
        geometry = sampler.sample("hot_120km")
        result = sampler.compute_positions(geometry, num_agents=2, formation_spread_m=5000.0)

        # Two agents should have some separation
        pos1 = result["agent_positions"][0]
        pos2 = result["agent_positions"][1]

        # Calculate approximate distance
        dlat = (pos2.lat - pos1.lat) * 111000  # meters
        dlon = (pos2.lon - pos1.lon) * 111000  # meters
        dist = np.sqrt(dlat**2 + dlon**2)

        assert dist > 1000, "Formation members should be spread apart"

    def test_includes_geometry_in_result(self):
        """Result should include original geometry config."""
        sampler = GeometrySampler()
        geometry = sampler.sample("hot_120km")
        result = sampler.compute_positions(geometry)

        assert "geometry" in result
        assert result["geometry"] == geometry


class TestSampleWithNoise:
    """Tests for sample_with_noise method."""

    def test_sample_with_zero_noise(self):
        """Zero noise should return original geometry."""
        sampler = GeometrySampler()
        rng = np.random.default_rng(42)

        base = sampler.sample("hot_120km")
        noisy = sampler.sample_with_noise(
            "hot_120km",
            position_noise_m=0.0,
            altitude_noise_m=0.0,
            heading_noise_deg=0.0,
            rng=rng,
        )

        assert noisy.range_m == base.range_m
        assert noisy.aspect == base.aspect
        assert noisy.altitude_split == base.altitude_split

    def test_sample_with_position_noise(self):
        """Position noise should vary the range."""
        sampler = GeometrySampler()
        rng = np.random.default_rng(42)

        results = []
        for _ in range(10):
            noisy = sampler.sample_with_noise(
                "hot_120km",
                position_noise_m=10000.0,
                rng=rng,
            )
            results.append(noisy.range_m)

        # Not all results should be identical
        assert len(set(results)) > 1, "Noise should create variation"

    def test_sample_with_altitude_noise(self):
        """Altitude noise should vary the altitudes."""
        sampler = GeometrySampler()
        rng = np.random.default_rng(42)

        results = []
        for _ in range(10):
            noisy = sampler.sample_with_noise(
                "hot_120km",
                altitude_noise_m=1000.0,
                rng=rng,
            )
            results.append(noisy.agent_altitude_m)

        assert len(set(results)) > 1, "Altitude noise should create variation"

    def test_altitude_clamped_to_valid_range(self):
        """Altitudes should be clamped to valid range."""
        sampler = GeometrySampler()
        rng = np.random.default_rng(42)

        for _ in range(100):
            noisy = sampler.sample_with_noise(
                "hot_120km",
                altitude_noise_m=50000.0,  # Very large noise
                rng=rng,
            )
            assert 1000.0 <= noisy.agent_altitude_m <= 20000.0
            assert 1000.0 <= noisy.opponent_altitude_m <= 20000.0

    def test_range_has_minimum(self):
        """Range should have a minimum value."""
        sampler = GeometrySampler()
        rng = np.random.default_rng(42)

        for _ in range(100):
            noisy = sampler.sample_with_noise(
                "hot_80km",
                position_noise_m=100000.0,  # Very large noise
                rng=rng,
            )
            assert noisy.range_m >= 10000.0, "Range should have minimum"

    def test_reproducible_with_same_seed(self):
        """Same RNG seed should produce same results."""
        sampler = GeometrySampler()

        rng1 = np.random.default_rng(42)
        result1 = sampler.sample_with_noise(
            "hot_120km",
            position_noise_m=5000.0,
            altitude_noise_m=1000.0,
            rng=rng1,
        )

        rng2 = np.random.default_rng(42)
        result2 = sampler.sample_with_noise(
            "hot_120km",
            position_noise_m=5000.0,
            altitude_noise_m=1000.0,
            rng=rng2,
        )

        assert result1.range_m == result2.range_m
        assert result1.agent_altitude_m == result2.agent_altitude_m
