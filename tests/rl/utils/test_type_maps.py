"""
Tests for type_maps.py - Aircraft and missile type mappings.
"""

import pytest

from bvr_marl_core.rl.utils.type_maps import (
    AIRCRAFT_TYPE_MAP,
    MISSILE_TYPE_MAP,
    resolve_aircraft_config,
)


class TestAircraftTypeMap:
    """Tests for AIRCRAFT_TYPE_MAP."""

    def test_contains_f22(self):
        """F22 should be in the aircraft type map."""
        assert "F22" in AIRCRAFT_TYPE_MAP

    def test_contains_f35(self):
        """F35 should be in the aircraft type map."""
        assert "F35" in AIRCRAFT_TYPE_MAP

    def test_contains_eurofighter(self):
        """Eurofighter should be in the aircraft type map."""
        assert "Eurofighter" in AIRCRAFT_TYPE_MAP

    def test_contains_su57(self):
        """Su57 should be in the aircraft type map."""
        assert "Su57" in AIRCRAFT_TYPE_MAP

    def test_contains_awacs(self):
        """AWACS should be in the aircraft type map."""
        assert "AWACS" in AIRCRAFT_TYPE_MAP

    def test_contains_debug_plane(self):
        """DebugPlane should be in the aircraft type map."""
        assert "DebugPlane" in AIRCRAFT_TYPE_MAP

    def test_all_values_are_callable(self):
        """All aircraft types should be callable classes."""
        for name, cls in AIRCRAFT_TYPE_MAP.items():
            assert callable(cls), f"{name} should be callable"

    def test_aircraft_count(self):
        """Should have exactly 6 aircraft types."""
        assert len(AIRCRAFT_TYPE_MAP) == 6


class TestMissileTypeMap:
    """Tests for MISSILE_TYPE_MAP."""

    def test_contains_amraam(self):
        """AIM120_AMRAAM should be in the missile type map."""
        assert "AIM120_AMRAAM" in MISSILE_TYPE_MAP

    def test_contains_meteor(self):
        """Meteor should be in the missile type map."""
        assert "Meteor" in MISSILE_TYPE_MAP

    def test_contains_k77m(self):
        """K77M should be in the missile type map."""
        assert "K77M" in MISSILE_TYPE_MAP

    def test_contains_default_missile(self):
        """DefaultMissile should be in the missile type map."""
        assert "DefaultMissile" in MISSILE_TYPE_MAP

    def test_all_values_are_callable(self):
        """All missile types should be callable classes."""
        for name, cls in MISSILE_TYPE_MAP.items():
            assert callable(cls), f"{name} should be callable"

    def test_missile_count(self):
        """Should have exactly 4 missile types."""
        assert len(MISSILE_TYPE_MAP) == 4


class TestResolveAircraftConfig:
    """Tests for resolve_aircraft_config function."""

    def test_resolve_valid_aircraft_types(self):
        """Should resolve valid aircraft type strings to classes."""
        cfg = {
            "aircraft_config": {
                "agent_type": "F22",
                "opponent_type": "Su57",
            }
        }
        result = resolve_aircraft_config(cfg)
        assert "aircraft_types" in result
        assert result["aircraft_types"]["agent"] == AIRCRAFT_TYPE_MAP["F22"]
        assert result["aircraft_types"]["opponent"] == AIRCRAFT_TYPE_MAP["Su57"]

    def test_resolve_default_aircraft_types(self):
        """Should use F22 as default when types not specified."""
        cfg = {"aircraft_config": {}}
        result = resolve_aircraft_config(cfg)
        assert result["aircraft_types"]["agent"] == AIRCRAFT_TYPE_MAP["F22"]
        assert result["aircraft_types"]["opponent"] == AIRCRAFT_TYPE_MAP["F22"]

    def test_resolve_empty_config(self):
        """Should return empty dict when no aircraft_config."""
        result = resolve_aircraft_config({})
        assert result == {}

    def test_resolve_invalid_agent_type_raises(self):
        """Should raise ValueError for invalid agent aircraft type."""
        cfg = {
            "aircraft_config": {
                "agent_type": "InvalidPlane",
                "opponent_type": "F22",
            }
        }
        with pytest.raises(ValueError) as exc_info:
            resolve_aircraft_config(cfg)
        assert "InvalidPlane" in str(exc_info.value)

    def test_resolve_invalid_opponent_type_raises(self):
        """Should raise ValueError for invalid opponent aircraft type."""
        cfg = {
            "aircraft_config": {
                "agent_type": "F22",
                "opponent_type": "InvalidPlane",
            }
        }
        with pytest.raises(ValueError) as exc_info:
            resolve_aircraft_config(cfg)
        assert "InvalidPlane" in str(exc_info.value)

    def test_resolve_awacs_type(self):
        """Should resolve AWACS aircraft type."""
        cfg = {
            "aircraft_config": {
                "agent_type": "F22",
                "opponent_type": "AWACS",
            }
        }
        result = resolve_aircraft_config(cfg)
        assert result["aircraft_types"]["opponent"] == AIRCRAFT_TYPE_MAP["AWACS"]
