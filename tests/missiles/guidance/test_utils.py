import pytest
import math
from unittest.mock import Mock, patch


class TestMovingAverageFilter:
    """Test the moving average filter."""

    def test_moving_average_import(self):
        """Test that moving average filter can be imported."""
        try:
            from missiles.guidance.utils import MovingAverageFilter
            assert MovingAverageFilter is not None
        except ImportError:
            pytest.skip("MovingAverageFilter not available")

    def test_moving_average_initialization(self):
        """Test moving average filter initialization."""
        try:
            from missiles.guidance.utils import MovingAverageFilter
            
            filter = MovingAverageFilter(5)
            assert filter.window_size == 5
            assert len(filter.values) == 0
            
        except ImportError:
            pytest.skip("MovingAverageFilter dependencies not available")

    def test_moving_average_single_value(self):
        """Test moving average with single value."""
        try:
            from missiles.guidance.utils import MovingAverageFilter
            
            filter = MovingAverageFilter(3)
            filter.add(10.0)
            assert filter.average() == 10.0
            
        except ImportError:
            pytest.skip("MovingAverageFilter dependencies not available")

    def test_moving_average_multiple_values(self):
        """Test moving average with multiple values."""
        try:
            from missiles.guidance.utils import MovingAverageFilter
            
            filter = MovingAverageFilter(3)
            filter.add(1.0)
            filter.add(2.0)
            filter.add(3.0)
            
            assert filter.average() == 2.0  # (1 + 2 + 3) / 3
            
        except ImportError:
            pytest.skip("MovingAverageFilter dependencies not available")

    def test_moving_average_window_overflow(self):
        """Test moving average with more values than window size."""
        try:
            from missiles.guidance.utils import MovingAverageFilter
            
            filter = MovingAverageFilter(3)
            filter.add(1.0)
            filter.add(2.0)
            filter.add(3.0)
            filter.add(4.0)  # Should push out the first value
            
            assert filter.average() == 3.0  # (2 + 3 + 4) / 3
            
            filter.add(5.0)  # Should push out the second value
            assert filter.average() == 4.0  # (3 + 4 + 5) / 3
            
        except ImportError:
            pytest.skip("MovingAverageFilter dependencies not available")

    def test_moving_average_empty_filter(self):
        """Test moving average with empty filter."""
        try:
            from missiles.guidance.utils import MovingAverageFilter
            
            filter = MovingAverageFilter(5)
            assert filter.average() == 0.0
            
        except ImportError:
            pytest.skip("MovingAverageFilter dependencies not available")

    def test_moving_average_negative_values(self):
        """Test moving average with negative values."""
        try:
            from missiles.guidance.utils import MovingAverageFilter
            
            filter = MovingAverageFilter(4)
            filter.add(-2.0)
            filter.add(-1.0)
            filter.add(1.0)
            filter.add(2.0)
            
            assert filter.average() == 0.0  # (-2 + -1 + 1 + 2) / 4
            
        except ImportError:
            pytest.skip("MovingAverageFilter dependencies not available")


class TestCircularMovingAverageFilter:
    """Test the circular moving average filter for angles."""

    def test_circular_average_import(self):
        """Test that circular moving average filter can be imported."""
        try:
            from missiles.guidance.utils import CircularMovingAverageFilter
            assert CircularMovingAverageFilter is not None
        except ImportError:
            pytest.skip("CircularMovingAverageFilter not available")

    def test_circular_average_initialization(self):
        """Test circular moving average filter initialization."""
        try:
            from missiles.guidance.utils import CircularMovingAverageFilter
            
            filter = CircularMovingAverageFilter(5)
            assert filter.window_size == 5
            assert len(filter._s) == 0
            assert len(filter._c) == 0
            
        except ImportError:
            pytest.skip("CircularMovingAverageFilter dependencies not available")

    def test_circular_average_single_angle(self):
        """Test circular moving average with single angle."""
        try:
            from missiles.guidance.utils import CircularMovingAverageFilter
            
            filter = CircularMovingAverageFilter(3)
            filter.add(90.0)  # East
            result = filter.average()
            assert abs(result - 90.0) < 1e-10
            
        except ImportError:
            pytest.skip("CircularMovingAverageFilter dependencies not available")

    def test_circular_average_simple_angles(self):
        """Test circular moving average with simple angles."""
        try:
            from missiles.guidance.utils import CircularMovingAverageFilter
            
            filter = CircularMovingAverageFilter(4)
            filter.add(0.0)    # North
            filter.add(90.0)   # East
            filter.add(180.0)  # South
            filter.add(270.0)  # West
            
            # These four directions should average to some central direction
            # The exact result depends on the vector averaging
            result = filter.average()
            assert 0.0 <= result < 360.0
            
        except ImportError:
            pytest.skip("CircularMovingAverageFilter dependencies not available")

    def test_circular_average_wraparound(self):
        """Test circular moving average with wraparound angles."""
        try:
            from missiles.guidance.utils import CircularMovingAverageFilter
            
            filter = CircularMovingAverageFilter(2)
            filter.add(350.0)  # 10 degrees west of north
            filter.add(10.0)   # 10 degrees east of north
            
            result = filter.average()
            # Should average to approximately 0 degrees (north)
            # allowing for some numerical precision issues
            assert result < 5.0 or result > 355.0
            
        except ImportError:
            pytest.skip("CircularMovingAverageFilter dependencies not available")

    def test_circular_average_same_direction(self):
        """Test circular moving average with same direction."""
        try:
            from missiles.guidance.utils import CircularMovingAverageFilter
            
            filter = CircularMovingAverageFilter(3)
            filter.add(45.0)
            filter.add(45.0)
            filter.add(45.0)
            
            result = filter.average()
            assert abs(result - 45.0) < 1e-10
            
        except ImportError:
            pytest.skip("CircularMovingAverageFilter dependencies not available")

    def test_circular_average_empty_filter(self):
        """Test circular moving average with empty filter."""
        try:
            from missiles.guidance.utils import CircularMovingAverageFilter
            
            filter = CircularMovingAverageFilter(5)
            assert filter.average() == 0.0
            
        except ImportError:
            pytest.skip("CircularMovingAverageFilter dependencies not available")

    def test_circular_average_angle_normalization(self):
        """Test circular moving average with angles outside 0-360 range."""
        try:
            from missiles.guidance.utils import CircularMovingAverageFilter
            
            filter = CircularMovingAverageFilter(2)
            filter.add(450.0)  # Should be normalized to 90.0
            filter.add(-90.0)  # Should be normalized to 270.0
            
            result = filter.average()
            # 90 and 270 degrees should average to either 0 or 180
            assert 0.0 <= result < 360.0
            
        except ImportError:
            pytest.skip("CircularMovingAverageFilter dependencies not available")

    def test_circular_average_window_overflow(self):
        """Test circular moving average with more values than window size."""
        try:
            from missiles.guidance.utils import CircularMovingAverageFilter
            
            filter = CircularMovingAverageFilter(2)
            filter.add(0.0)
            filter.add(90.0)
            filter.add(180.0)  # Should push out the first value
            
            # Now averaging 90 and 180 degrees
            result = filter.average()
            assert 120.0 < result < 150.0  # Should be around 135 degrees
            
        except ImportError:
            pytest.skip("CircularMovingAverageFilter dependencies not available")


class TestDistanceFunction:
    """Test the distance utility function."""

    def test_distance_import(self):
        """Test that distance function can be imported."""
        try:
            from missiles.guidance.utils import distance_m
            assert distance_m is not None
        except ImportError:
            pytest.skip("distance_m not available")

    def test_distance_calculation(self):
        """Test distance calculation between positions."""
        try:
            from missiles.guidance.utils import distance_m
            
            with patch('missiles.guidance.utils.geodetic_distance_km') as mock_distance:
                mock_distance.return_value = 150.0  # 150 km
                
                pos1 = Mock()
                pos1.lat = 45.0
                pos1.lon = 2.0
                pos1.alt = 8000.0
                
                pos2 = Mock()
                pos2.lat = 46.0
                pos2.lon = 3.0
                pos2.alt = 9000.0
                
                result = distance_m(pos1, pos2)
                
                # Should convert km to meters
                assert result == 150000.0  # 150 km * 1000
                
                # Verify the underlying function was called correctly
                mock_distance.assert_called_once_with(
                    pos1.lat, pos1.lon, pos1.alt,
                    pos2.lat, pos2.lon, pos2.alt
                )
                
        except ImportError:
            pytest.skip("distance_m dependencies not available")

    def test_distance_zero(self):
        """Test distance calculation with same position."""
        try:
            from missiles.guidance.utils import distance_m
            
            with patch('missiles.guidance.utils.geodetic_distance_km') as mock_distance:
                mock_distance.return_value = 0.0
                
                pos = Mock()
                pos.lat = 45.0
                pos.lon = 2.0
                pos.alt = 8000.0
                
                result = distance_m(pos, pos)
                assert result == 0.0
                
        except ImportError:
            pytest.skip("distance_m dependencies not available")

    def test_distance_long_range(self):
        """Test distance calculation for long range."""
        try:
            from missiles.guidance.utils import distance_m
            
            with patch('missiles.guidance.utils.geodetic_distance_km') as mock_distance:
                mock_distance.return_value = 1000.0  # 1000 km
                
                pos1 = Mock()
                pos1.lat = 40.0
                pos1.lon = -74.0  # New York area
                pos1.alt = 10000.0
                
                pos2 = Mock()
                pos2.lat = 51.5
                pos2.lon = 0.1  # London area
                pos2.alt = 11000.0
                
                result = distance_m(pos1, pos2)
                assert result == 1000000.0  # 1000 km in meters
                
        except ImportError:
            pytest.skip("distance_m dependencies not available")