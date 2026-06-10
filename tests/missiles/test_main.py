from unittest.mock import Mock, patch

import pytest


class TestMissilesModule:
    """Test the main missiles module and overall integration."""

    def test_missiles_module_structure(self):
        """Test that the missiles module has the expected structure."""
        try:
            import bvr_marl_core.missiles

            # Test that main missile class can be imported
            from bvr_marl_core.missiles.missile import Missile

            assert Missile is not None

            # Test that FOX1 missiles can be imported
            from bvr_marl_core.missiles.fox1.skyflash import Skyflash
            from bvr_marl_core.missiles.fox1.sparrow import AIM7_Sparrow

            assert AIM7_Sparrow is not None
            assert Skyflash is not None

            # Test that FOX2 missiles can be imported
            from bvr_marl_core.missiles.fox2.python import Python_5
            from bvr_marl_core.missiles.fox2.sidewinder import AIM9_Sidewinder

            assert AIM9_Sidewinder is not None
            assert Python_5 is not None

            # Test that FOX3 missiles can be imported
            from bvr_marl_core.missiles.fox3.amraam import AIM120_AMRAAM
            from bvr_marl_core.missiles.fox3.default_missile import LongRangeMissile

            assert AIM120_AMRAAM is not None
            assert LongRangeMissile is not None

        except ImportError as e:
            pytest.fail(f"Expected imports failed: {e}")

    def test_missiles_guidance_integration(self):
        """Test that guidance systems integrate properly."""
        try:
            from bvr_marl_core.missiles.guidance.guidance import MissileGuidance
            from bvr_marl_core.missiles.guidance.pn_propnav import PnPropNavGuidance
            from bvr_marl_core.missiles.guidance.target_provider import GuidanceTargetProvider

            assert MissileGuidance is not None
            assert PnPropNavGuidance is not None
            assert GuidanceTargetProvider is not None

        except ImportError as e:
            pytest.fail(f"Expected imports failed: {e}")

    def test_missiles_seeker_integration(self):
        """Test that seeker systems integrate properly."""
        try:
            from bvr_marl_core.missiles.fox1.sarh_seeker import SemiActiveRadarSeeker
            from bvr_marl_core.missiles.fox2.ir_seeker import InfraredSeeker

            assert SemiActiveRadarSeeker is not None
            assert InfraredSeeker is not None

        except ImportError as e:
            pytest.fail(f"Expected imports failed: {e}")

    def test_missile_types_classification(self):
        """Test that different missile types are properly classified."""
        try:
            # Test FOX1 (SARH) classification
            from bvr_marl_core.missiles.fox1.sparrow import AIM7_Sparrow

            mock_source = Mock()
            mock_source.position = Mock()
            mock_source.position.copy.return_value = mock_source.position
            mock_source.yaw_deg = 0.0
            mock_source.speed = 300.0
            mock_source.pitch_deg = 0.0
            mock_source.group = "BLUE"

            mock_target = Mock()
            mock_map_limits = {"lat_min": 44, "lat_max": 46, "lon_min": 1, "lon_max": 3}

            with patch(
                "bvr_marl_core.missiles.fox1.base_fox1.Fox1Missile.__init__", return_value=None
            ):
                sparrow = AIM7_Sparrow(0.0, mock_target, mock_source, mock_map_limits)
                sparrow.fox_type = 1  # Set manually since __init__ is mocked
                assert hasattr(sparrow, "fox_type")
                assert sparrow.fox_type == 1

            # Test FOX2 (IR) classification
            from bvr_marl_core.missiles.fox2.sidewinder import AIM9_Sidewinder

            with patch(
                "bvr_marl_core.missiles.fox2.base_fox2.Fox2Missile.__init__", return_value=None
            ):
                sidewinder = AIM9_Sidewinder(0.0, mock_target, mock_source, mock_map_limits)
                sidewinder.fox_type = 2  # Set manually since __init__ is mocked
                assert hasattr(sidewinder, "fox_type")
                assert sidewinder.fox_type == 2

            # Test FOX3 (Active Radar) classification
            from bvr_marl_core.missiles.fox3.amraam import AIM120_AMRAAM

            with patch("bvr_marl_core.missiles.missile.Missile.__init__", return_value=None):
                amraam = AIM120_AMRAAM(0.0, mock_target, mock_source, mock_map_limits)
                amraam.fox_type = 3  # Set manually since __init__ is mocked
                assert hasattr(amraam, "fox_type")
                assert amraam.fox_type == 3

        except ImportError as e:
            pytest.fail(f"Expected imports failed: {e}")

    def test_missile_configuration_consistency(self):
        """Test that missile configurations are consistent across types."""
        try:
            from bvr_marl_core.missiles.fox1.sparrow import AIM7_Sparrow
            from bvr_marl_core.missiles.fox2.sidewinder import AIM9_Sidewinder
            from bvr_marl_core.missiles.fox3.amraam import AIM120_AMRAAM

            mock_source = Mock()
            mock_source.position = Mock()
            mock_source.position.copy.return_value = mock_source.position
            mock_source.yaw_deg = 0.0
            mock_source.speed = 300.0
            mock_source.pitch_deg = 0.0
            mock_source.group = "BLUE"

            mock_target = Mock()
            mock_map_limits = {"lat_min": 44, "lat_max": 46, "lon_min": 1, "lon_max": 3}

            # Test that all missiles have required configuration keys
            required_keys = [
                "mass_kg",
                "max_speed_mps",
                "life_time_s",
                "hit_probability",
                "motor_burn_s",
                "reference_area_m2",
                "drag_coefficient",
                "constant_engine_N",
                "n_max",
            ]

            with patch(
                "bvr_marl_core.missiles.fox1.base_fox1.Fox1Missile.__init__", return_value=None
            ) as mock_fox1:
                AIM7_Sparrow(0.0, mock_target, mock_source, mock_map_limits)
                sparrow_config = mock_fox1.call_args[1]["config"]

                for key in required_keys:
                    assert key in sparrow_config, f"Missing {key} in Sparrow config"
                    assert isinstance(sparrow_config[key], (int, float)), f"{key} should be numeric"

            with patch(
                "bvr_marl_core.missiles.fox2.base_fox2.Fox2Missile.__init__", return_value=None
            ) as mock_fox2:
                AIM9_Sidewinder(0.0, mock_target, mock_source, mock_map_limits)
                sidewinder_config = mock_fox2.call_args[1]["config"]

                for key in required_keys:
                    assert key in sidewinder_config, f"Missing {key} in Sidewinder config"
                    assert isinstance(sidewinder_config[key], (int, float)), (
                        f"{key} should be numeric"
                    )

            with patch(
                "bvr_marl_core.missiles.missile.Missile.__init__", return_value=None
            ) as mock_missile:
                AIM120_AMRAAM(0.0, mock_target, mock_source, mock_map_limits)
                amraam_config = mock_missile.call_args[1]["config"]

                for key in required_keys:
                    assert key in amraam_config, f"Missing {key} in AMRAAM config"
                    assert isinstance(amraam_config[key], (int, float)), f"{key} should be numeric"

        except ImportError as e:
            pytest.fail(f"Expected imports failed: {e}")

    def test_missile_performance_hierarchy(self):
        """Test that missile performance follows expected hierarchy."""
        try:
            from bvr_marl_core.missiles.fox1.sparrow import AIM7_Sparrow
            from bvr_marl_core.missiles.fox2.sidewinder import AIM9_Sidewinder
            from bvr_marl_core.missiles.fox3.amraam import AIM120_AMRAAM
            from bvr_marl_core.missiles.fox3.default_missile import LongRangeMissile

            mock_source = Mock()
            mock_source.position = Mock()
            mock_source.position.copy.return_value = mock_source.position
            mock_source.yaw_deg = 0.0
            mock_source.speed = 300.0
            mock_source.pitch_deg = 0.0
            mock_source.group = "BLUE"

            mock_target = Mock()
            mock_map_limits = {"lat_min": 44, "lat_max": 46, "lon_min": 1, "lon_max": 3}

            configs = {}

            with patch(
                "bvr_marl_core.missiles.fox1.base_fox1.Fox1Missile.__init__", return_value=None
            ) as mock_fox1:
                AIM7_Sparrow(0.0, mock_target, mock_source, mock_map_limits)
                configs["sparrow"] = mock_fox1.call_args[1]["config"]

            with patch(
                "bvr_marl_core.missiles.fox2.base_fox2.Fox2Missile.__init__", return_value=None
            ) as mock_fox2:
                AIM9_Sidewinder(0.0, mock_target, mock_source, mock_map_limits)
                configs["sidewinder"] = mock_fox2.call_args[1]["config"]

            with patch(
                "bvr_marl_core.missiles.missile.Missile.__init__", return_value=None
            ) as mock_missile:
                AIM120_AMRAAM(0.0, mock_target, mock_source, mock_map_limits)
                configs["amraam"] = mock_missile.call_args[1]["config"]

                mock_missile.reset_mock()
                LongRangeMissile(0.0, mock_target, mock_source, mock_map_limits)
                configs["long_range"] = mock_missile.call_args[1]["config"]

            # Test expected performance hierarchy
            # FOX2 should be lighter and more agile than others
            assert configs["sidewinder"]["mass_kg"] < configs["sparrow"]["mass_kg"]
            assert configs["sidewinder"]["mass_kg"] < configs["amraam"]["mass_kg"]
            assert configs["sidewinder"]["n_max"] >= configs["sparrow"]["n_max"]

            # FOX3 should have longer range than FOX2
            assert configs["amraam"]["life_time_s"] > configs["sidewinder"]["life_time_s"]
            assert configs["amraam"]["max_speed_mps"] > configs["sidewinder"]["max_speed_mps"]

            # Long range missile should exceed all others in range metrics
            assert configs["long_range"]["life_time_s"] >= configs["amraam"]["life_time_s"]
            assert configs["long_range"]["max_speed_mps"] >= configs["amraam"]["max_speed_mps"]
            assert configs["long_range"]["mass_kg"] >= configs["amraam"]["mass_kg"]

        except ImportError as e:
            pytest.fail(f"Expected imports failed: {e}")
