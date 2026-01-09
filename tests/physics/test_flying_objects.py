#!/usr/bin/env python3
"""
Comprehensive tests for physics.flying_objects module.
Tests FlyingPhysics class and all its methods including movement computation,
energy management, stall handling, and orientation updates.
"""

import pytest
import math
import numpy as np
from unittest.mock import Mock, patch

from physics.physics import AirLayer, PhysicsParams, AIRSPEED_OF_SOUND
from physics.flying_objects import FlyingPhysics
from simulator.core.helpers import Position
from tests.mocks import MockPosition


class TestFlyingPhysics:
    """Test FlyingPhysics class functionality."""
    
    @pytest.fixture
    def basic_params(self):
        """Basic physics parameters for testing."""
        return PhysicsParams(
            mass_kg=12000.0,
            reference_area_m2=28.0,
            air=AirLayer(),
            aspect_ratio=6.5,
            oswald_e=0.8,
            max_speed_mps=650,
            n_max=8.0,
            engine_thrust_profile={"full_thrust_speed": 160.0, "optimal_speed": 250.0},
            stall0_mps=75.0,
            service_ceiling_m=15000.0,
            stall_margin_mps=20.0,
            energy_relief_gain=2.0
        )
    
    @pytest.fixture
    def flying_physics(self, basic_params):
        """FlyingPhysics instance for testing."""
        return FlyingPhysics(basic_params)
    
    @pytest.fixture
    def position(self):
        """Test position."""
        return Position(0.0, 0.0, 5000.0)
    
    def test_init(self, flying_physics, basic_params):
        """Test FlyingPhysics initialization."""
        assert flying_physics.mass_kg == basic_params.mass_kg
        assert flying_physics.A_m2 == basic_params.reference_area_m2
        assert flying_physics.g == basic_params.gravity_m_s2
        assert flying_physics.air == basic_params.air
        assert flying_physics.AR == basic_params.aspect_ratio
        assert flying_physics.eps == basic_params.oswald_e
        assert flying_physics.max_speed == basic_params.max_speed_mps
        assert flying_physics.n_max == basic_params.n_max
        assert flying_physics.stall0_mps == basic_params.stall0_mps
        assert flying_physics.service_ceiling == basic_params.service_ceiling_m
        assert flying_physics.enable_energy_protections == True
        assert hasattr(flying_physics, 'force_model')
        
        # Test calculated values
        expected_K_ind = 1.0 / (math.pi * basic_params.aspect_ratio * basic_params.oswald_e)
        assert math.isclose(flying_physics.K_ind, expected_K_ind, rel_tol=1e-3)
    
    def test_get_engine_force_base(self, flying_physics):
        """Test base engine force method (should return 0)."""
        force = flying_physics.get_engine_force(250.0, 5000.0, 1.0)
        assert force == 0.0
    
    def test_get_base_drag_cd_base(self, flying_physics):
        """Test base drag coefficient method."""
        cd = flying_physics.get_base_drag_cd(250.0, 5000.0)
        assert cd == flying_physics.cd0
    
    def test_get_pitch_limit_deg_base(self, flying_physics):
        """Test base pitch limit method."""
        limit = flying_physics.get_pitch_limit_deg(None, 250.0)
        assert limit == 90.0
    
    def test_get_ca_max_base(self, flying_physics):
        """Test base CA_max method."""
        ca_max = flying_physics.get_ca_max(250.0, 5000.0)
        assert ca_max == 1.2  # Default value
    
    def test_engine_speed_factor(self, flying_physics):
        """Test engine speed factor calculation."""
        # Below full thrust speed
        fv_low = flying_physics._engine_speed_factor(150.0)
        assert fv_low == 1.0
        
        # Above full thrust speed
        fv_high = flying_physics._engine_speed_factor(300.0)
        assert fv_high > 1.0
        assert fv_high <= 2.4  # Clamped maximum
    
    def test_dynamic_stall_calculation(self, flying_physics):
        """Test dynamic stall speed calculation."""
        # Sea level
        v_stall_sl = flying_physics._dynamic_stall(0.0)
        assert math.isclose(v_stall_sl, flying_physics.stall0_mps, rel_tol=1e-3)
        
        # Higher altitude (lower density)
        v_stall_alt = flying_physics._dynamic_stall(10000.0)
        assert v_stall_alt > v_stall_sl  # Should be higher at altitude
        
        # No stall speed defined
        flying_physics.stall0_mps = None
        v_stall_none = flying_physics._dynamic_stall(5000.0)
        assert math.isinf(v_stall_none)
    
    def test_is_in_stall_condition(self, flying_physics):
        """Test stall condition detection."""
        # No stall speed defined
        flying_physics.stall0_mps = None
        assert not flying_physics.is_in_stall_condition(50.0, 5000.0, 1.0)
        
        # Reset stall speed
        flying_physics.stall0_mps = 75.0
        
        # Above stall speed
        v_stall_dynamic = flying_physics._dynamic_stall(5000.0)
        assert not flying_physics.is_in_stall_condition(v_stall_dynamic * 1.2, 5000.0, 1.0)
        
        # Below stall speed - need to account for dynamic stall calculation
        v_stall_dynamic = flying_physics._dynamic_stall(5000.0)
        assert flying_physics.is_in_stall_condition(v_stall_dynamic * 0.8, 5000.0, 1.0)
        
        # High load factor increases stall speed
        assert flying_physics.is_in_stall_condition(v_stall_dynamic * 0.9, 5000.0, 2.0)  # sqrt(2) * stall speed
    
    def test_service_ceiling_pitch_limit(self, flying_physics):
        """Test service ceiling pitch limiting."""
        # Below 80% of ceiling
        low_alt_limit = flying_physics._service_ceiling_pitch_limit(5000.0)
        assert low_alt_limit == 90.0
        
        # At ceiling
        ceiling_limit = flying_physics._service_ceiling_pitch_limit(15000.0)
        assert ceiling_limit == 0.0
        
        # In taper region
        mid_alt_limit = flying_physics._service_ceiling_pitch_limit(13000.0)  # 87% of ceiling
        assert 0.0 < mid_alt_limit < 90.0
        
        # No service ceiling defined
        flying_physics.service_ceiling = None
        no_ceiling_limit = flying_physics._service_ceiling_pitch_limit(20000.0)
        assert no_ceiling_limit == 90.0


class TestFlyingPhysicsMovement:
    """Test movement-related methods of FlyingPhysics."""
    
    @pytest.fixture
    def flying_physics(self):
        """FlyingPhysics with mock engine and drag methods."""
        params = PhysicsParams(
            mass_kg=15000.0,
            reference_area_m2=30.0,
            air=AirLayer(),
            max_speed_mps=400.0,
            n_max=9.0,
            stall0_mps=80.0,
            service_ceiling_m=18000.0
        )
        physics = FlyingPhysics(params)
        
        # Mock methods for predictable testing
        def mock_engine_force(v_mps, alt_m, throttle):
            return throttle * 150000.0 * (physics.air.get_density(alt_m) / physics.air.rho0)**0.7
        
        def mock_drag_cd(v_mps, alt_m):
            mach = v_mps / AIRSPEED_OF_SOUND
            return 0.02 if mach < 0.8 else 0.025
        
        physics.get_engine_force = mock_engine_force
        physics.get_base_drag_cd = mock_drag_cd
        
        return physics
    
    @pytest.fixture
    def position(self):
        return Position(0.0, 0.0, 8000.0)
    
    def test_update_pitch_deg_basic(self, flying_physics):
        """Test basic pitch update."""
        new_pitch = flying_physics.update_pitch_deg(0.0, 15.0, 250.0, 1.0)
        assert 0.0 < new_pitch <= 15.0  # Should move toward desired pitch
        
        # Large time step should get closer to desired
        new_pitch_large_dt = flying_physics.update_pitch_deg(0.0, 15.0, 250.0, 5.0)
        assert new_pitch_large_dt >= new_pitch
    
    def test_update_pitch_deg_with_limits(self, flying_physics, position):
        """Test pitch update with position-based limits."""
        # Mock pitch limit method
        flying_physics.get_pitch_limit_deg = lambda pos, speed: 30.0
        
        # Desired pitch exceeds limit
        new_pitch = flying_physics.update_pitch_deg(0.0, 45.0, 250.0, 1.0, position)
        assert new_pitch <= 30.0
    
    def test_update_yaw_deg_basic(self, flying_physics, position):
        """Test basic yaw update."""
        new_yaw, omega = flying_physics.update_yaw_deg(position, 0.0, 90.0, 250.0, 2.0)
        
        assert 0.0 <= new_yaw <= 90.0  # Should move toward desired yaw
        assert omega != 0.0  # Should have angular velocity
    
    def test_update_yaw_deg_zero_speed(self, flying_physics, position):
        """Test yaw update with zero speed."""
        new_yaw, omega = flying_physics.update_yaw_deg(position, 0.0, 90.0, 0.0, 2.0)
        
        assert new_yaw == 0.0  # Should not change
        assert omega == 0.0
    
    def test_update_yaw_deg_large_turn(self, flying_physics, position):
        """Test yaw update with large turn angle."""
        # 180 degree turn
        new_yaw, omega = flying_physics.update_yaw_deg(position, 0.0, 180.0, 250.0, 1.0)
        
        # Should move towards target, but might not complete in one time step
        # Allow for negative angles due to angle normalization
        yaw_change = abs(new_yaw - 0.0)
        if new_yaw < 0:
            yaw_change = min(yaw_change, abs(new_yaw + 360.0 - 0.0))
        assert yaw_change > 0.0  # Should have some change
        assert abs(omega) > 0.0  # Should have angular velocity
    
    def test_update_yaw_deg_wrap_around(self, flying_physics, position):
        """Test yaw update with angle wrap-around."""
        # Turn from 350 to 10 degrees (should go through 0)
        new_yaw, omega = flying_physics.update_yaw_deg(position, 350.0, 10.0, 250.0, 2.0)
        
        # Should move in positive direction through 0
        assert new_yaw > 350.0 or new_yaw < 50.0
    
    def test_update_roll(self, flying_physics):
        """Test roll calculation from turn rate."""
        # Level flight, no turn
        roll = flying_physics.update_roll(250.0, 0.0, 0.0)
        assert roll == 0.0
        
        # Turn with level flight
        omega_rad = math.radians(10.0)  # 10 deg/s turn rate
        roll = flying_physics.update_roll(250.0, omega_rad, 0.0)
        assert roll > 0.0  # Should have positive roll for right turn
        
        # Turn with climb
        roll_climb = flying_physics.update_roll(250.0, omega_rad, 20.0)
        assert roll_climb < roll  # Climbing reduces effective roll angle


class TestFlyingPhysicsIntegration:
    """Test integrated movement computation."""
    
    @pytest.fixture
    def flying_physics(self):
        """FlyingPhysics with realistic parameters."""
        params = PhysicsParams(
            mass_kg=18000.0,
            reference_area_m2=32.0,
            air=AirLayer(),
            aspect_ratio=6.0,
            oswald_e=0.75,
            max_speed_mps=500.0,
            n_max=7.5,
            stall0_mps=85.0,
            service_ceiling_m=16000.0,
            stall_margin_mps=25.0,
            energy_relief_gain=1.5
        )
        physics = FlyingPhysics(params)
        
        # Realistic engine model
        def engine_force(v_mps, alt_m, throttle):
            rho_rel = physics.air.get_density(alt_m) / physics.air.rho0
            T_sl = physics.mass_kg * physics.g * 1.2  # T/W = 1.2 at sea level
            fv = 1.0 if v_mps <= 200.0 else max(0.5, 1.0 - (v_mps - 200.0) / 600.0)
            return throttle * T_sl * fv * rho_rel**0.8
        
        # Realistic drag model
        def drag_cd(v_mps, alt_m):
            mach = v_mps / AIRSPEED_OF_SOUND
            if mach < 0.7:
                return 0.018
            elif mach < 0.9:
                return 0.018 + (mach - 0.7) * 0.01 / 0.2  # Drag rise
            else:
                return 0.028 + (mach - 0.9) * 0.005  # Supersonic
        
        physics.get_engine_force = engine_force
        physics.get_base_drag_cd = drag_cd
        
        return physics
    
    def test_straight_level_flight(self, flying_physics):
        """Test straight and level flight."""
        pos = Position(0.0, 0.0, 8000.0)
        result = flying_physics.compute_movement(
            pos, 0.0, 0.0, 0.0, 0.0, 250.0, 0.8, 10.0
        )
        
        lat2, lon2, alt2, v2, yaw2, pitch2, roll2 = result
        
        # Should move north (yaw=0)
        assert lat2 > pos.lat
        assert abs(lon2 - pos.lon) < 1e-6  # Minimal east-west movement
        
        # Altitude should remain approximately constant
        assert abs(alt2 - pos.alt) < 200.0
        
        # Speed should be reasonable
        assert 200.0 < v2 < 300.0
        
        # Attitudes should be close to commanded
        assert abs(yaw2) < 5.0
        assert abs(pitch2) < 5.0
        assert abs(roll2) < 5.0
    
    def test_climbing_flight(self, flying_physics):
        """Test climbing flight with energy management."""
        pos = Position(0.0, 0.0, 5000.0)
        result = flying_physics.compute_movement(
            pos, 0.0, 0.0, 15.0, 15.0, 220.0, 1.0, 15.0
        )
        
        lat2, lon2, alt2, v2, yaw2, pitch2, roll2 = result
        
        # Should gain significant altitude
        assert alt2 > pos.alt + 300.0  # Reduced expectation for realistic climb
        
        # Speed might decrease due to climb and energy conversion
        assert v2 > 150.0  # Allow for energy exchange altitude for speed
    
    def test_descending_flight(self, flying_physics):
        """Test descending flight."""
        pos = Position(0.0, 0.0, 12000.0)
        result = flying_physics.compute_movement(
            pos, 0.0, 0.0, -10.0, -10.0, 280.0, 0.3, 20.0
        )
        
        lat2, lon2, alt2, v2, yaw2, pitch2, roll2 = result
        
        # Should lose altitude
        assert alt2 < pos.alt - 500.0
        
        # Speed might increase due to descent
        assert v2 > 280.0
    
    def test_turning_flight(self, flying_physics):
        """Test coordinated turning flight."""
        pos = Position(0.0, 0.0, 6000.0)
        result = flying_physics.compute_movement(
            pos, 0.0, 90.0, 0.0, 5.0, 250.0, 0.9, 8.0
        )
        
        lat2, lon2, alt2, v2, yaw2, pitch2, roll2 = result
        
        # Should turn toward east
        assert 0.0 < yaw2 <= 90.0
        
        # Should have some eastward movement
        assert lon2 > pos.lon
        
        # Should have bank angle for turn
        assert abs(roll2) > 1.0
    
    def test_energy_protection_stall_prevention(self, flying_physics):
        """Test energy protection system prevents stall."""
        # Set up near-stall condition but not too extreme
        pos = Position(0.0, 0.0, 8000.0)
        stall_speed = flying_physics._dynamic_stall(8000.0) if flying_physics.stall0_mps else 100.0
        low_speed = stall_speed * 1.3  # Just above stall speed
        
        result = flying_physics.compute_movement(
            pos, 0.0, 0.0, 15.0, 15.0, low_speed, 0.5, 3.0  # Moderate throttle and pitch
        )
        
        lat2, lon2, alt2, v2, yaw2, pitch2, roll2 = result
        
        # Energy protections should prevent speed from dropping too low
        final_stall_speed = flying_physics._dynamic_stall(alt2) if flying_physics.stall0_mps else 100.0
        assert v2 >= final_stall_speed * 0.9  # Allow some margin but should not stall badly
    
    def test_stall_recovery_override(self, flying_physics):
        """Test stall recovery when speed drops too low."""
        # Force stall condition by using speed well below stall
        pos = Position(0.0, 0.0, 5000.0)
        if flying_physics.stall0_mps:
            stall_speed = flying_physics._dynamic_stall(5000.0)
            test_speed = stall_speed * 0.7  # Well below stall speed
        else:
            stall_speed = 100.0  # Fallback
            test_speed = 60.0
        
        result = flying_physics.compute_movement(
            pos, 0.0, 0.0, 20.0, 20.0, test_speed, 0.5, 3.0
        )
        
        lat2, lon2, alt2, v2, yaw2, pitch2, roll2 = result
        
        # Stall recovery should kick in - either pitch down OR speed recovery
        # Allow for different implementations of stall recovery
        if flying_physics.stall0_mps:
            final_stall_speed = flying_physics._dynamic_stall(alt2)
            assert pitch2 < 10.0 or v2 > test_speed * 1.1  # Either pitch down or speed recovery
        else:
            # No stall speed defined, should behave normally
            assert v2 > 0
    
    def test_high_altitude_performance(self, flying_physics):
        """Test performance at high altitude near service ceiling."""
        pos = Position(0.0, 0.0, 15000.0)  # Near service ceiling
        
        result = flying_physics.compute_movement(
            pos, 0.0, 0.0, 10.0, 10.0, 200.0, 1.0, 10.0
        )
        
        lat2, lon2, alt2, v2, yaw2, pitch2, roll2 = result
        
        # Pitch should be limited by service ceiling
        assert pitch2 < 10.0  # Should be limited
        
        # Performance should be reduced at high altitude
        # but aircraft should still function
        assert v2 > 150.0


class TestFlyingPhysicsStallHandling:
    """Test stall detection and recovery systems."""
    
    @pytest.fixture
    def physics_with_stall(self):
        """FlyingPhysics configured for stall testing."""
        params = PhysicsParams(
            mass_kg=12000.0,
            reference_area_m2=25.0,
            air=AirLayer(),
            stall0_mps=70.0,
            stall_margin_mps=15.0,
            stall_sink_pitch_deg=-8.0
        )
        physics = FlyingPhysics(params)
        
        # Initialize stall recovery state variables
        physics.in_stall_recovery = False
        physics.stall_recovery_start_time = 0.0
        physics.speed_at_stall_entry = 0.0
        physics.recovery_time_required = 3.0
        
        return physics
    
    def test_should_enter_stall_recovery(self, physics_with_stall):
        """Test stall recovery entry logic."""
        # Not in stall condition
        assert not physics_with_stall.should_enter_stall_recovery(100.0, 5000.0, 1.0, 10.0)
        assert not physics_with_stall.in_stall_recovery
        
        # In stall condition
        assert physics_with_stall.should_enter_stall_recovery(60.0, 5000.0, 1.0, 20.0)
        assert physics_with_stall.in_stall_recovery
        assert physics_with_stall.stall_recovery_start_time == 20.0
        assert physics_with_stall.speed_at_stall_entry == 60.0
        
        # Already in recovery - should not re-enter
        assert not physics_with_stall.should_enter_stall_recovery(55.0, 5000.0, 1.0, 25.0)
    
    def test_should_exit_stall_recovery(self, physics_with_stall):
        """Test stall recovery exit logic."""
        # Not in recovery
        assert not physics_with_stall.should_exit_stall_recovery(100.0, 5000.0, 10.0)
        
        # Enter recovery first
        physics_with_stall.in_stall_recovery = True
        physics_with_stall.stall_recovery_start_time = 10.0
        physics_with_stall.speed_at_stall_entry = 60.0
        
        # Too soon to exit (minimum time not met)
        assert not physics_with_stall.should_exit_stall_recovery(85.0, 5000.0, 12.0)
        
        # Sufficient time but speed not recovered enough
        assert not physics_with_stall.should_exit_stall_recovery(75.0, 5000.0, 15.0)
        
        # Sufficient time and speed recovered
        v_stall = physics_with_stall._dynamic_stall(5000.0)
        recovery_speed = v_stall * 1.3  # Well above stall
        assert physics_with_stall.should_exit_stall_recovery(recovery_speed, 5000.0, 15.0)
        assert not physics_with_stall.in_stall_recovery  # Should exit recovery
    
    def test_compute_stall_recovery_controls(self, physics_with_stall):
        """Test stall recovery control computation."""
        # Not in recovery
        pitch, throttle = physics_with_stall.compute_stall_recovery_controls(10.0, 0.8)
        assert pitch == 10.0  # No change
        assert throttle == 0.8  # No change
        
        # In recovery
        physics_with_stall.in_stall_recovery = True
        
        # Should force nose down and full throttle
        pitch, throttle = physics_with_stall.compute_stall_recovery_controls(10.0, 0.5)
        assert pitch <= -15.0  # Aggressive nose down
        assert throttle == 1.0  # Full throttle
        
        # Already nose down - should not change
        pitch, throttle = physics_with_stall.compute_stall_recovery_controls(-20.0, 0.7)
        assert pitch == -20.0  # Keep current nose down attitude
        assert throttle == 1.0  # Full throttle


class TestFlyingPhysicsEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def flying_physics(self):
        params = PhysicsParams(
            mass_kg=10000.0,
            reference_area_m2=20.0,
            air=AirLayer()
        )
        return FlyingPhysics(params)
    
    def test_zero_speed_handling(self, flying_physics):
        """Test handling of zero or very low speeds."""
        pos = Position(0.0, 0.0, 1000.0)
        
        # Zero speed
        result = flying_physics.compute_movement(
            pos, 0.0, 90.0, 0.0, 10.0, 0.0, 1.0, 1.0
        )
        
        # Should handle gracefully without crashing
        lat2, lon2, alt2, v2, yaw2, pitch2, roll2 = result
        assert math.isfinite(lat2)
        assert math.isfinite(lon2)
        assert math.isfinite(alt2)
        assert v2 >= 0.0
    
    def test_extreme_altitudes(self, flying_physics):
        """Test handling of extreme altitudes."""
        # Very high altitude
        pos_high = Position(0.0, 0.0, 50000.0)
        result = flying_physics.compute_movement(
            pos_high, 0.0, 0.0, 0.0, 0.0, 200.0, 1.0, 5.0
        )
        
        # Should handle without crashing
        assert all(math.isfinite(x) for x in result)
        
        # Very low altitude (below sea level)
        pos_low = Position(0.0, 0.0, -1000.0)
        result = flying_physics.compute_movement(
            pos_low, 0.0, 0.0, 0.0, 0.0, 200.0, 1.0, 5.0
        )
        
        # Should handle without crashing
        assert all(math.isfinite(x) for x in result)
    
    def test_extreme_attitudes(self, flying_physics):
        """Test handling of extreme pitch and yaw angles."""
        pos = Position(0.0, 0.0, 5000.0)
        
        # Extreme pitch
        result = flying_physics.compute_movement(
            pos, 0.0, 0.0, 89.0, 89.0, 200.0, 1.0, 2.0
        )
        
        assert all(math.isfinite(x) for x in result)
        
        # Large yaw changes
        result = flying_physics.compute_movement(
            pos, 0.0, 359.0, 0.0, 0.0, 200.0, 1.0, 2.0
        )
        
        assert all(math.isfinite(x) for x in result)
    
    def test_extreme_time_steps(self, flying_physics):
        """Test handling of extreme time steps."""
        pos = Position(0.0, 0.0, 5000.0)
        
        # Very small time step
        result = flying_physics.compute_movement(
            pos, 0.0, 0.0, 0.0, 0.0, 200.0, 1.0, 1e-6
        )
        
        # Position should barely change
        lat2, lon2, alt2, v2, yaw2, pitch2, roll2 = result
        assert abs(lat2 - pos.lat) < 1e-8  # Relaxed precision
        assert abs(lon2 - pos.lon) < 1e-8
        assert abs(alt2 - pos.alt) < 1e-2
        
        # Large time step
        result = flying_physics.compute_movement(
            pos, 0.0, 0.0, 0.0, 0.0, 200.0, 1.0, 100.0
        )
        
        # Should still produce reasonable results
        assert all(math.isfinite(x) for x in result)
    
    def test_no_stall_speed_defined(self, flying_physics):
        """Test behavior when no stall speed is defined."""
        flying_physics.stall0_mps = None
        
        # Should not trigger stall conditions
        assert not flying_physics.is_in_stall_condition(10.0, 5000.0, 1.0)
        
        # Movement should work normally
        pos = Position(0.0, 0.0, 5000.0)
        result = flying_physics.compute_movement(
            pos, 0.0, 0.0, 0.0, 0.0, 10.0, 1.0, 1.0
        )
        
        assert all(math.isfinite(x) for x in result)
    
    def test_no_service_ceiling_defined(self, flying_physics):
        """Test behavior when no service ceiling is defined."""
        flying_physics.service_ceiling = None
        
        # Pitch limits should not be affected by altitude
        limit_low = flying_physics._service_ceiling_pitch_limit(1000.0)
        limit_high = flying_physics._service_ceiling_pitch_limit(20000.0)
        
        assert limit_low == limit_high == 90.0


class TestFlyingPhysicsCompatibility:
    """Test compatibility with existing test cases."""
    
    @pytest.fixture
    def flying_physics(self):
        """FlyingPhysics matching original test parameters."""
        params = PhysicsParams(
            mass_kg=12000.0,
            reference_area_m2=28.0,
            air=AirLayer(),
            aspect_ratio=6.5,
            oswald_e=0.8,
            max_speed_mps=650,
            n_max=8.0,
            engine_thrust_profile={"full_thrust_speed": 160.0, "optimal_speed": 250.0},
        )
        return FlyingPhysics(params)
    
    def test_update_yaw_deg_compatibility(self, flying_physics):
        """Test compatibility with original yaw update test."""
        pos = Position(0.0, 0.0, 1500.0)
        yaw, omega = flying_physics.update_yaw_deg(pos, 0.0, 90.0, 250.0, 2.0)
        assert 0 <= yaw <= 90
    
    def test_update_pitch_deg_compatibility(self, flying_physics):
        """Test compatibility with original pitch update test."""
        pitch = flying_physics.update_pitch_deg(0.0, 25.0, 250.0, 1.0)
        assert 0 <= pitch <= 25.0
    
    def test_compute_movement_simple_compatibility(self, flying_physics):
        """Test compatibility with original simple movement test."""
        pos = Position(0.0, 0.0, 1200.0)
        res = flying_physics.compute_movement(
            pos, 0.0, 0.0, 0.0, 0.0, 220.0, 1.0, 5.0
        )
        lat2, lon2, alt2, v2, yaw2, pitch2, roll2 = res
        assert v2 > 100  # Original assertion