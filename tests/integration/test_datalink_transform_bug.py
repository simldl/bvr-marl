"""
Test to verify if datalink coordinate transformation introduces errors.

The _transform_detection_to_local_ref method does:
1. geodetic -> ENU (relative to source)
2. ENU -> geodetic (relative to target)

This should be mathematically equivalent to identity (no change),
but may introduce numerical errors.
"""

import numpy as np
import pytest

from air_to_air_rl.radar.core.utils import enu_to_geodetic, geodetic_to_enu
from air_to_air_rl.simulator.core.helpers import Position

pytestmark = pytest.mark.integration


def test_datalink_transform_identity():
    """Test that datalink coordinate transform preserves absolute position."""
    from air_to_air_rl.radar.core.data_link import DataLink

    # Source radar position (e.g., AWACS)
    src_pos = Position(lat=45.0, lon=10.0, alt=8000.0)

    # Target radar position (e.g., missile)
    tgt_pos = Position(lat=45.05, lon=10.05, alt=6000.0)

    # Actual target position (enemy fighter)
    target_lat, target_lon, target_alt = 45.2, 10.3, 7000.0

    # Create detection
    detection = {"lat": target_lat, "lon": target_lon, "alt": target_alt, "T": None}

    # Use actual DataLink method
    datalink = DataLink(mode="full")
    detection_transformed = datalink._transform_detection_to_local_ref(detection, src_pos, tgt_pos)

    # Check if position is preserved (should be unchanged now after fix)
    lat2 = detection_transformed["lat"]
    lon2 = detection_transformed["lon"]
    alt2 = detection_transformed["alt"]

    lat_error_m = abs(lat2 - target_lat) * 111000
    lon_error_m = abs(lon2 - target_lon) * 111000 * np.cos(np.radians(target_lat))
    alt_error_m = abs(alt2 - target_alt)

    print(f"Original: ({target_lat}, {target_lon}, {target_alt})")
    print(f"After transform: ({lat2}, {lon2}, {alt2})")
    print(f"Errors: lat={lat_error_m:.2f}m, lon={lon_error_m:.2f}m, alt={alt_error_m:.2f}m")

    # The transform should preserve position exactly (geodetic coords are absolute)
    assert lat_error_m < 0.01, f"Latitude error: {lat_error_m:.4f}m"
    assert lon_error_m < 0.01, f"Longitude error: {lon_error_m:.4f}m"
    assert alt_error_m < 0.01, f"Altitude error: {alt_error_m:.4f}m"


def test_datalink_transform_logic_error():
    """
    Test that demonstrates the logical error in _transform_detection_to_local_ref.

    The current implementation converts geodetic -> ENU(src) -> geodetic,
    which is mathematically wrong. The ENU vector from step 1 is relative to
    source position, but step 2 treats it as if it's relative to target position.
    """
    # Source radar (AWACS at origin-ish)
    src_pos = Position(lat=0.0, lon=0.0, alt=0.0)

    # Target radar (missile far away)
    tgt_pos = Position(lat=1.0, lon=1.0, alt=0.0)

    # Actual target position
    target_lat, target_lon, target_alt = 0.5, 0.5, 1000.0

    # What _transform_detection_to_local_ref does:
    enu_src = geodetic_to_enu(
        target_lat, target_lon, target_alt, src_pos.lat, src_pos.lon, src_pos.alt
    )
    lat_transformed, lon_transformed, alt_transformed = enu_to_geodetic(
        enu_src, tgt_pos.lat, tgt_pos.lon, tgt_pos.alt
    )

    print("\nLogic error demonstration:")
    print(f"Source position: ({src_pos.lat}, {src_pos.lon}, {src_pos.alt})")
    print(f"Target position: ({tgt_pos.lat}, {tgt_pos.lon}, {tgt_pos.alt})")
    print(f"Actual target: ({target_lat}, {target_lon}, {target_alt})")
    print(f"After transform: ({lat_transformed}, {lon_transformed}, {alt_transformed})")

    # Calculate the ENU vector as it should be (relative to missile)
    enu_correct = geodetic_to_enu(
        target_lat, target_lon, target_alt, tgt_pos.lat, tgt_pos.lon, tgt_pos.alt
    )

    print(f"\nENU relative to source: {enu_src}")
    print(f"ENU relative to target (correct): {enu_correct}")

    # These should be very different!
    enu_diff = np.linalg.norm(enu_src - enu_correct)
    print(f"ENU difference magnitude: {enu_diff:.2f}m")

    # If they're very different (which they should be), then using enu_src
    # in enu_to_geodetic with tgt_pos is mathematically incorrect!
    assert enu_diff > 1000.0, "ENU vectors should be very different (different reference frames)"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
