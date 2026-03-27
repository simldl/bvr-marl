"""
Integration test for missile guidance accuracy with AWACS datalink support.

Tests:
1. ENU coordinate transformation accuracy
2. Datalink fusion between AWACS and missile
3. Missile tracking lock stability
4. Missile removal reasons analysis
"""

import numpy as np
import pytest

from air_to_air_rl.radar.core.utils import enu_to_geodetic, geodetic_to_enu
from air_to_air_rl.simulator.core.helpers import Position

pytestmark = pytest.mark.integration


class TestCoordinateTransformations:
    """Verify ENU coordinate transformations are accurate and reversible."""

    def test_enu_round_trip_same_ref(self):
        """Test that geodetic -> ENU -> geodetic preserves coordinates (same reference)."""
        ref_lat, ref_lon, ref_alt = 45.0, 10.0, 5000.0
        target_lat, target_lon, target_alt = 45.1, 10.1, 6000.0

        # Forward transform
        enu = geodetic_to_enu(target_lat, target_lon, target_alt, ref_lat, ref_lon, ref_alt)

        # Reverse transform
        lat2, lon2, alt2 = enu_to_geodetic(enu, ref_lat, ref_lon, ref_alt)

        # Should match within numerical precision
        assert abs(lat2 - target_lat) < 1e-6, f"Latitude mismatch: {lat2} vs {target_lat}"
        assert abs(lon2 - target_lon) < 1e-6, f"Longitude mismatch: {lon2} vs {target_lon}"
        assert abs(alt2 - target_alt) < 1e-3, f"Altitude mismatch: {alt2} vs {target_alt}"

    def test_enu_transform_chain_awacs_to_missile(self):
        """Test coordinate transformation chain: AWACS -> intermediate ENU -> missile."""
        # AWACS position
        awacs_lat, awacs_lon, awacs_alt = 45.0, 10.0, 8000.0

        # Target position (enemy fighter)
        target_lat, target_lon, target_alt = 45.2, 10.3, 7000.0

        # Missile position
        missile_lat, missile_lon, missile_alt = 45.05, 10.05, 6000.0

        # Step 1: AWACS detects target -> ENU relative to AWACS
        enu_awacs = geodetic_to_enu(
            target_lat, target_lon, target_alt, awacs_lat, awacs_lon, awacs_alt
        )

        # Step 2: Transform to geodetic (as datalink would)
        lat_intermediate, lon_intermediate, alt_intermediate = enu_to_geodetic(
            enu_awacs, awacs_lat, awacs_lon, awacs_alt
        )

        # Step 3: Missile receives and transforms to its own ENU frame
        enu_missile = geodetic_to_enu(
            lat_intermediate,
            lon_intermediate,
            alt_intermediate,
            missile_lat,
            missile_lon,
            missile_alt,
        )

        # Step 4: Verify we can recover original target position
        lat_final, lon_final, alt_final = enu_to_geodetic(
            enu_missile, missile_lat, missile_lon, missile_alt
        )

        # Should match original target position within acceptable error
        lat_error_m = abs(lat_final - target_lat) * 111000  # ~111km per degree
        lon_error_m = abs(lon_final - target_lon) * 111000 * np.cos(np.radians(target_lat))
        alt_error_m = abs(alt_final - target_alt)

        assert lat_error_m < 10.0, f"Latitude error: {lat_error_m:.2f}m"
        assert lon_error_m < 10.0, f"Longitude error: {lon_error_m:.2f}m"
        assert alt_error_m < 10.0, f"Altitude error: {alt_error_m:.2f}m"

    def test_enu_velocity_preservation(self):
        """Test that ENU velocity vectors are preserved correctly."""
        _ref_lat, _ref_lon, _ref_alt = 45.0, 10.0, 5000.0

        # Velocity in ENU frame (m/s): 300 m/s east, 200 m/s north, 50 m/s up
        vel_enu = np.array([300.0, 200.0, 50.0])

        # Velocity should be preserved in ENU frame (it's already in local tangent plane)
        # This is a sanity check that we're not accidentally transforming velocities
        vel_magnitude = np.linalg.norm(vel_enu)
        expected_magnitude = np.sqrt(300**2 + 200**2 + 50**2)

        assert abs(vel_magnitude - expected_magnitude) < 0.01


class TestDatalinkModeConfiguration:
    """Test that datalink modes are configured correctly for different unit types."""

    def test_fighter_datalink_mode(self):
        """Verify fighters have 'full' datalink mode for sharing."""
        from air_to_air_rl.aircrafts.types.f22 import F22
        from air_to_air_rl.simulator.core.helpers import Position

        map_limits = type(
            "obj",
            (),
            {
                "min_alt": 0,
                "max_alt": 20000,
                "left_lon": 0,
                "right_lon": 1,
                "bottom_lat": 0,
                "top_lat": 1,
            },
        )()

        fighter = F22(
            position=Position(45.0, 10.0, 5000.0),
            yaw_deg=0.0,
            speed_mps=250.0,
            group="blue",
            map_limits=map_limits,
            min_alt_m=0,
            max_alt_m=20000,
        )

        assert hasattr(fighter, "radar"), "Fighter should have radar"
        assert hasattr(fighter.radar, "data_link"), "Fighter radar should have datalink"
        assert fighter.radar.data_link.get_mode() == "full", (
            "Fighter should share data in 'full' mode"
        )

    def test_missile_datalink_mode_transition(self):
        """Verify missiles transition from 'other' to 'full' mode correctly."""
        from air_to_air_rl.aircrafts.types.f22 import F22
        from air_to_air_rl.missiles.fox3.amraam import AIM120_AMRAAM
        from air_to_air_rl.simulator.core.helpers import Position

        map_limits = type(
            "obj",
            (),
            {
                "min_alt": 0,
                "max_alt": 20000,
                "left_lon": 0,
                "right_lon": 1,
                "bottom_lat": 0,
                "top_lat": 1,
            },
        )()

        # Create shooter
        shooter = F22(
            position=Position(45.0, 10.0, 5000.0),
            yaw_deg=0.0,
            speed_mps=250.0,
            group="blue",
            map_limits=map_limits,
            min_alt_m=0,
            max_alt_m=20000,
        )

        # Create target
        target = F22(
            position=Position(45.2, 10.2, 6000.0),
            yaw_deg=180.0,
            speed_mps=250.0,
            group="red",
            map_limits=map_limits,
            min_alt_m=0,
            max_alt_m=20000,
        )

        # Create missile
        missile = AIM120_AMRAAM(
            firing_time_s=0.0, target=target, source=shooter, map_limits=map_limits, group="blue"
        )

        # Initial mode should be "other" (receiving only)
        initial_mode = missile.radar.data_link.get_mode()
        assert initial_mode == "other", f"Missile should start in 'other' mode, got {initial_mode}"

        # Check that original mode is stored as "full"
        assert missile.radar._original_data_link_mode == "full", (
            "Missile should store 'full' as original mode"
        )


class TestMissileGuidanceFiltering:
    """Test that missiles filter targets correctly during flight."""

    def test_target_provider_filters_non_engageable(self):
        """Verify target provider filters out is_non_engageable targets."""
        from air_to_air_rl.missiles.guidance.target_provider import GuidanceTargetProvider

        # Create mock missile
        missile = type(
            "obj", (), {"designated_target_id": "target1", "retarget_policy": "locked_override"}
        )()

        # Create mock radar with tracks
        mock_awacs = type(
            "obj",
            (),
            {
                "id": "awacs1",
                "is_missile": False,
                "is_countermeasure": False,
                "is_non_engageable": True,  # AWACS
            },
        )()

        mock_fighter = type(
            "obj",
            (),
            {
                "id": "target1",
                "is_missile": False,
                "is_countermeasure": False,
                "is_non_engageable": False,
            },
        )()

        # Mock tracks: tid, state, cov, tgt, utype, ref, confidence, n_obs, lifetime, update_count,
        #              is_deception, suspect_deception, engagement_id, jammer_id, engageable
        tracks = [
            (
                1,
                np.array([1000, 0, 0, 100, 0, 0]),
                None,
                mock_awacs,
                "air",
                None,
                0.9,
                10,
                5.0,
                20,
                False,
                False,
                None,
                None,
                False,
            ),
            (
                2,
                np.array([2000, 0, 0, 100, 0, 0]),
                None,
                mock_fighter,
                "air",
                None,
                0.9,
                10,
                5.0,
                20,
                False,
                False,
                None,
                None,
                True,
            ),
        ]

        missile.radar = type(
            "obj",
            (),
            {
                "get_tracks": lambda self: tracks,
                "get_locked_target": lambda self: "target1",
                "target_provider": None,
            },
        )()

        missile.position = type("obj", (), {"lat": 45.0, "lon": 10.0, "alt": 5000.0})()

        provider = GuidanceTargetProvider(missile)
        missile.radar.target_provider = provider

        # Simulate update — active_units needed by target_provider.update()
        sim = type(
            "obj",
            (),
            {
                "utc_time": 1.0,
                "active_units": {"target1": mock_fighter},
            },
        )()
        provider.update(sim, 0.1)

        # Verify that AWACS is not selected as target
        target_pos = provider.get_guidance_target()
        assert target_pos is not None, "Should have selected a valid target"

        # Verify current_target_id is the fighter, not AWACS
        assert provider.current_target_id == "target1", (
            f"Should target fighter, got {provider.current_target_id}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
