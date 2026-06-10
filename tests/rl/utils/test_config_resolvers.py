"""
Tests for config_resolvers.py - Configuration resolution utilities.
"""

import pytest

from bvr_marl_core.rl.utils.config_resolvers import (
    resolve_datalink_config,
    resolve_loadout_config,
    resolve_missile_config,
)
from bvr_marl_core.rl.utils.type_maps import MISSILE_TYPE_MAP


class TestResolveMissileConfig:
    """Tests for resolve_missile_config function."""

    def test_resolve_valid_missile_types(self):
        """Should resolve valid missile type strings to classes."""
        config = {
            "agent_missiles": ["AIM120_AMRAAM", "Meteor"],
            "opponent_missiles": ["K77M"],
        }
        result = resolve_missile_config(config)

        assert len(result["agent_missiles"]) == 2
        assert result["agent_missiles"][0] == MISSILE_TYPE_MAP["AIM120_AMRAAM"]
        assert result["agent_missiles"][1] == MISSILE_TYPE_MAP["Meteor"]
        assert len(result["opponent_missiles"]) == 1
        assert result["opponent_missiles"][0] == MISSILE_TYPE_MAP["K77M"]

    def test_resolve_none_config(self):
        """Should return empty lists when config is None."""
        result = resolve_missile_config(None)
        assert result["agent_missiles"] == []
        assert result["opponent_missiles"] == []

    def test_resolve_empty_config(self):
        """Should return empty lists when config is empty."""
        result = resolve_missile_config({})
        assert result["agent_missiles"] == []
        assert result["opponent_missiles"] == []

    def test_resolve_partial_config(self):
        """Should handle partial config with only one side specified."""
        config = {"agent_missiles": ["AIM120_AMRAAM"]}
        result = resolve_missile_config(config)
        assert len(result["agent_missiles"]) == 1
        assert result["opponent_missiles"] == []

    def test_resolve_invalid_agent_missile_raises(self):
        """Should raise ValueError for invalid agent missile type."""
        config = {
            "agent_missiles": ["InvalidMissile"],
            "opponent_missiles": ["AIM120_AMRAAM"],
        }
        with pytest.raises(ValueError) as exc_info:
            resolve_missile_config(config)
        assert "InvalidMissile" in str(exc_info.value)

    def test_resolve_invalid_opponent_missile_raises(self):
        """Should raise ValueError for invalid opponent missile type."""
        config = {
            "agent_missiles": ["AIM120_AMRAAM"],
            "opponent_missiles": ["InvalidMissile"],
        }
        with pytest.raises(ValueError) as exc_info:
            resolve_missile_config(config)
        assert "InvalidMissile" in str(exc_info.value)

    def test_resolve_single_missile(self):
        """Should handle single missile in list."""
        config = {
            "agent_missiles": ["Meteor"],
            "opponent_missiles": ["K77M"],
        }
        result = resolve_missile_config(config)
        assert result["agent_missiles"][0] == MISSILE_TYPE_MAP["Meteor"]
        assert result["opponent_missiles"][0] == MISSILE_TYPE_MAP["K77M"]


class TestResolveLoadoutConfig:
    """Tests for resolve_loadout_config function."""

    def test_resolve_full_loadout(self):
        """Should resolve all loadout values."""
        config = {
            "max_missiles": 10,
            "flares": 30,
            "chaff": 50,
            "ecm": 5,
            "decoys": 8,
        }
        result = resolve_loadout_config(config)

        assert result["max_missiles"] == 10
        assert result["flares"] == 30
        assert result["chaff"] == 50
        assert result["ecm"] == 5
        assert result["decoys"] == 8

    def test_resolve_none_config(self):
        """Should return empty dict when config is None."""
        result = resolve_loadout_config(None)
        assert result == {}

    def test_resolve_partial_loadout(self):
        """Should handle partial loadout with some values missing."""
        config = {"max_missiles": 12, "flares": 20}
        result = resolve_loadout_config(config)

        assert result["max_missiles"] == 12
        assert result["flares"] == 20
        assert result["chaff"] is None
        assert result["ecm"] is None
        assert result["decoys"] is None

    def test_resolve_empty_config(self):
        """Should return all None values for empty config."""
        result = resolve_loadout_config({})
        assert result["max_missiles"] is None
        assert result["flares"] is None


class TestResolveDataLinkConfig:
    """Tests for resolve_datalink_config function."""

    def test_resolve_full_mode(self):
        """Should resolve 'full' datalink mode."""
        config = {"datalink_mode": "full"}
        result = resolve_datalink_config(config)
        assert result == "full"

    def test_resolve_own_mode(self):
        """Should resolve 'own' datalink mode."""
        config = {"datalink_mode": "own"}
        result = resolve_datalink_config(config)
        assert result == "own"

    def test_resolve_none_mode(self):
        """Should resolve 'none' datalink mode."""
        config = {"datalink_mode": "none"}
        result = resolve_datalink_config(config)
        assert result == "none"

    def test_resolve_msl_support_mode(self):
        """Should resolve 'msl_support' datalink mode."""
        config = {"datalink_mode": "msl_support"}
        result = resolve_datalink_config(config)
        assert result == "msl_support"

    def test_resolve_default_mode(self):
        """Should default to 'full' when not specified."""
        config = {}
        result = resolve_datalink_config(config)
        assert result == "full"

    def test_resolve_missing_key(self):
        """Should default to 'full' when key is missing."""
        config = {"other_config": "value"}
        result = resolve_datalink_config(config)
        assert result == "full"
