#!/usr/bin/env python3
"""
Comprehensive tests for physics.missiles module.
Tests MissilePhysics class and all its methods including drag modeling,
engine modeling, and flight characteristics specific to missiles.
"""

import pytest
import math
import numpy as np
from physics.physics import AirLayer, AIRSPEED_OF_SOUND, get_speed_of_sound
from physics.missiles import MissilePhysics
from simulator.core.helpers import Position
from tests.mocks import MockPosition


class TestMissilePhysicsParams:
    """Test MissilePhysics.Params dataclass."""
    
    def test_default_params(self):
        """Test default parameter values for missiles."""
        params = MissilePhysics.Params(
            mass_kg=500.0,
            reference_area_m2=0.3
        )
        
        assert params.mass_kg == 500.0
        assert params.reference_area_m2 == 0.3
        assert params.aspect_ratio == 3.0
        assert params.oswald_e == 0.75
        assert params.n_max == 30.0
        assert params.max_speed_mps == 1440.0
        assert params.cd0_s1 == 0.08
        assert params.cd0_s2 == 0.33
        assert params.cd0_s3 == 0.30
        assert params.cd0_s4 == 0.23
        assert params.cd0_hi == 0.17
        assert params.mach_t1 == 1.1
        assert params.mach_t2 == 1.6
        assert params.mach_t3 == 2.4
        assert params.mach_t4 == 3.2
        assert params.constant_engine_F == 20000.0
        assert params.stall0_mps == 129.0
        assert params.service_ceiling_m is None
    
    def test_custom_params(self):
        """Test custom parameter values."""
        params = MissilePhysics.Params(
            mass_kg=750.0,
            reference_area_m2=0.4,
            aspect_ratio=3.5,
            oswald_e=0.80,
            n_max=35.0,
            max_speed_mps=1600.0,
            constant_engine_F=25000.0,
            stall0_mps=140.0
        )
        
        assert params.mass_kg == 750.0
        assert params.reference_area_m2 == 0.4
        assert params.aspect_ratio == 3.5
        assert params.oswald_e == 0.80
        assert params.n_max == 35.0
        assert params.max_speed_mps == 1600.0
        assert params.constant_engine_F == 25000.0
        assert params.stall0_mps == 140.0
    
    def test_engine_thrust_profile(self):
        """Test engine thrust profile configuration."""
        params = MissilePhysics.Params(
            mass_kg=500.0,
            reference_area_m2=0.3,
        )
        
        expected_profile = {"full_thrust_speed": 1030.0, "optimal_speed": 1280.0}
        assert params.engine_thrust_profile == expected_profile


class TestMissilePhysicsInit:
    """Test MissilePhysics initialization."""
    
    @pytest.fixture
    def basic_params(self):
        return MissilePhysics.Params(
            mass_kg=600.0,
            reference_area_m2=0.35,
            air=AirLayer()
        )
    
    @pytest.fixture
    def missile(self, basic_params):
        return MissilePhysics(basic_params)
    
    def test_init_basic(self, missile, basic_params):
        """Test basic initialization."""
        assert missile.mass_kg == basic_params.mass_kg
        assert missile.A_m2 == basic_params.reference_area_m2
        assert missile.cd0_s1 == basic_params.cd0_s1
        assert missile.cd0_s2 == basic_params.cd0_s2
        assert missile.cd0_s3 == basic_params.cd0_s3
        assert missile.cd0_s4 == basic_params.cd0_s4
        assert missile.cd0_hi == basic_params.cd0_hi
        assert missile.mach_t1 == basic_params.mach_t1
        assert missile.mach_t2 == basic_params.mach_t2
        assert missile.mach_t3 == basic_params.mach_t3
        assert missile.mach_t4 == basic_params.mach_t4
        assert missile.constant_engine_F == basic_params.constant_engine_F
    
    def test_energy_protections_disabled(self, missile):
        """Test that missiles have energy protections disabled."""
        assert missile.enable_energy_protections == False
    
    def test_inherited_properties(self, missile):
        """Test properties inherited from FlyingPhysics."""
        assert hasattr(missile, 'force_model')
        assert hasattr(missile, 'air')
        assert hasattr(missile, 'n_max')
        assert hasattr(missile, 'stall0_mps')
        assert missile.n_max == 30.0  # High G capability for missiles


class TestMissilePhysicsSmoothstep:
    """Test smoothstep interpolation function."""
    
    @pytest.fixture
    def missile(self):
        params = MissilePhysics.Params(
            mass_kg=500.0,
            reference_area_m2=0.3
        )
        return MissilePhysics(params)
    
    def test_smoothstep_bounds(self, missile):
        """Test smoothstep boundary conditions."""
        assert missile.smoothstep(0.0) == 0.0
        assert missile.smoothstep(1.0) == 1.0
        assert missile.smoothstep(-0.5) == 0.0  # Clamped to 0
        assert missile.smoothstep(1.5) == 1.0   # Clamped to 1
    
    def test_smoothstep_monotonic(self, missile):
        """Test smoothstep is monotonically increasing."""
        values = [missile.smoothstep(t) for t in np.linspace(0, 1, 11)]
        
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1]
    
    def test_smoothstep_continuity(self, missile):
        """Test smoothstep continuity."""
        # Test at several points
        test_points = [0.25, 0.5, 0.75]
        eps = 1e-6
        
        for point in test_points:
            val1 = missile.smoothstep(point - eps)
            val2 = missile.smoothstep(point + eps)
            assert abs(val2 - val1) < 1e-3  # Should be continuous


class TestMissilePhysicsDragModel:
    """Test missile drag modeling."""
    
    @pytest.fixture
    def missile(self):
        params = MissilePhysics.Params(
            mass_kg=500.0,
            reference_area_m2=0.3,
            air=AirLayer(),
            cd0_s1=0.10,
            cd0_s2=0.35,
            cd0_s3=0.32,
            cd0_s4=0.25,
            cd0_hi=0.18,
            mach_t1=1.2,
            mach_t2=1.8,
            mach_t3=2.5,
            mach_t4=3.5
        )
        return MissilePhysics(params)
    
    def test_get_base_drag_cd_subsonic(self, missile):
        """Test drag coefficient in subsonic range."""
        # Mach < 1.2
        cd = missile.get_base_drag_cd(300.0, 5000.0)  # ~Mach 0.88
        
        # Should be interpolating between cd0_s1 and cd0_s2
        assert missile.cd0_s1 <= cd <= missile.cd0_s2
    
    def test_get_base_drag_cd_transonic(self, missile):
        """Test drag coefficient in transonic range."""
        # Mach between t1 and t2 (1.2 - 1.8)
        cd = missile.get_base_drag_cd(500.0, 5000.0)  # ~Mach 1.47
        
        # Should be decreasing from peak
        assert cd <= missile.cd0_s2
        assert cd >= missile.cd0_s3
    
    def test_get_base_drag_cd_supersonic_mid(self, missile):
        """Test drag coefficient in mid-supersonic range."""
        # Mach between t2 and t3 (1.8 - 2.5)
        cd = missile.get_base_drag_cd(750.0, 5000.0)  # ~Mach 2.2
        
        # Should be in the t3 region, decreasing toward t4
        assert cd >= missile.cd0_s4
        assert cd <= missile.cd0_s3
    
    def test_get_base_drag_cd_high_supersonic(self, missile):
        """Test drag coefficient in high supersonic range."""
        # Mach between t3 and t4 (2.5 - 3.5)
        cd = missile.get_base_drag_cd(1100.0, 5000.0)  # ~Mach 3.23
        
        # Should be transitioning to high-speed value
        assert cd >= missile.cd0_hi
        assert cd <= missile.cd0_s4
    
    def test_get_base_drag_cd_hypersonic(self, missile):
        """Test drag coefficient at hypersonic speeds."""
        # Mach > 3.5
        altitude = 5000.0
        speed = 1400.0
        cd = missile.get_base_drag_cd(speed, altitude)  # ~Mach 4.1

        # Should have additional reduction for hypersonic flight
        mach = speed / get_speed_of_sound(altitude)
        expected = missile.cd0_hi - 0.01 * (mach - missile.mach_t4)
        assert math.isclose(cd, expected, rel_tol=1e-3)
    
    def test_drag_coefficient_continuity(self, missile):
        """Test continuity of drag coefficient across Mach regions."""
        # Test points around transition Mach numbers
        mach_points = [1.19, 1.21, 1.79, 1.81, 2.49, 2.51, 3.49, 3.51]
        speeds = [m * AIRSPEED_OF_SOUND for m in mach_points]
        cds = [missile.get_base_drag_cd(v, 5000.0) for v in speeds]
        
        # Check for reasonable continuity (no large jumps)
        for i in range(len(cds) - 1):
            if i % 2 == 0:  # Only check across transition points
                jump = abs(cds[i+1] - cds[i])
                assert jump < 0.02  # Should be relatively continuous
    
    def test_drag_peak_behavior(self, missile):
        """Test that drag peaks appropriately in transonic region."""
        # Check that cd0_s2 is indeed the peak value
        assert missile.cd0_s2 >= missile.cd0_s1
        assert missile.cd0_s2 >= missile.cd0_s3
        assert missile.cd0_s2 >= missile.cd0_s4
        assert missile.cd0_s2 >= missile.cd0_hi


class TestMissilePhysicsEngineModel:
    """Test missile engine modeling."""
    
    @pytest.fixture
    def missile(self):
        params = MissilePhysics.Params(
            mass_kg=400.0,
            reference_area_m2=0.25,
            air=AirLayer(),
            constant_engine_F=18000.0
        )
        return MissilePhysics(params)
    
    def test_get_engine_force_constant(self, missile):
        """Test engine force is constant regardless of conditions."""
        # Different speeds
        thrust1 = missile.get_engine_force(300.0, 5000.0, 1.0)
        thrust2 = missile.get_engine_force(800.0, 5000.0, 1.0)
        thrust3 = missile.get_engine_force(1200.0, 5000.0, 1.0)
        
        assert thrust1 == thrust2 == thrust3 == missile.constant_engine_F
    
    def test_get_engine_force_altitude_independent(self, missile):
        """Test engine force is independent of altitude."""
        # Different altitudes
        thrust_sl = missile.get_engine_force(500.0, 0.0, 1.0)
        thrust_mid = missile.get_engine_force(500.0, 10000.0, 1.0)
        thrust_high = missile.get_engine_force(500.0, 20000.0, 1.0)
        
        assert thrust_sl == thrust_mid == thrust_high == missile.constant_engine_F
    
    def test_get_engine_force_throttle_independent(self, missile):
        """Test engine force is independent of throttle (rocket motor behavior)."""
        # Different throttle settings
        thrust_idle = missile.get_engine_force(500.0, 5000.0, 0.0)
        thrust_half = missile.get_engine_force(500.0, 5000.0, 0.5)
        thrust_full = missile.get_engine_force(500.0, 5000.0, 1.0)
        
        assert thrust_idle == thrust_half == thrust_full == missile.constant_engine_F


class TestMissilePhysicsLimitsAndConstraints:
    """Test missile limits and constraints."""
    
    @pytest.fixture
    def missile(self):
        params = MissilePhysics.Params(
            mass_kg=500.0,
            reference_area_m2=0.3,
            air=AirLayer(),
            n_max=40.0
        )
        return MissilePhysics(params)
    
    def test_get_pitch_limit_deg_high(self, missile):
        """Test pitch limits are high for missiles."""
        pos = Position(0.0, 0.0, 5000.0)
        limit = missile.get_pitch_limit_deg(pos, 600.0)
        
        assert limit == 75.0  # High pitch authority for missiles
    
    def test_get_pitch_limit_deg_position_independent(self, missile):
        """Test pitch limits are position independent."""
        pos1 = Position(0.0, 0.0, 1000.0)
        pos2 = Position(0.0, 0.0, 15000.0)
        pos3 = None
        
        limit1 = missile.get_pitch_limit_deg(pos1, 500.0)
        limit2 = missile.get_pitch_limit_deg(pos2, 500.0)
        limit3 = missile.get_pitch_limit_deg(pos3, 500.0)
        
        assert limit1 == limit2 == limit3 == 75.0
    
    def test_get_ca_max_mach_dependent(self, missile):
        """Test CA_max varies with Mach number."""
        # get_ca_max now uses updated realistic values: C_sub=2.9, C_low=1.65, C_high=1.20
        # Subsonic (~Mach 0.88) - should be close to C_sub with some blending
        ca_max_sub = missile.get_ca_max(300.0, 5000.0)  # ~Mach 0.88
        assert 2.0 < ca_max_sub < 2.9  # Blending from subsonic

        # Low supersonic (~Mach 1.76) - should be in low-supersonic regime
        ca_max_sup1 = missile.get_ca_max(600.0, 5000.0)  # ~Mach 1.76
        assert 1.5 < ca_max_sup1 < 1.8  # Around C_low=1.65

        # High supersonic (~Mach 2.94) - should be close to C_high
        ca_max_sup2 = missile.get_ca_max(1000.0, 5000.0)  # ~Mach 2.94
        assert 1.15 < ca_max_sup2 < 1.3  # Around C_high=1.20
    
    def test_ca_max_monotonic_decrease(self, missile):
        """Test CA_max decreases with increasing Mach number."""
        speeds = [250, 500, 750, 1000, 1250]  # Increasing Mach
        ca_max_values = [missile.get_ca_max(v, 5000.0) for v in speeds]
        
        # Should generally decrease with increasing speed
        for i in range(len(ca_max_values) - 1):
            assert ca_max_values[i] >= ca_max_values[i + 1] - 1e-6  # Allow for numerical precision


class TestMissilePhysicsIntegration:
    """Test integrated missile physics behavior."""
    
    @pytest.fixture
    def realistic_missile(self):
        """Missile with realistic parameters for integration testing."""
        params = MissilePhysics.Params(
            mass_kg=180.0,
            reference_area_m2=0.2,
            air=AirLayer(),
            aspect_ratio=3.2,
            oswald_e=0.73,
            n_max=50.0,
            max_speed_mps=1200.0,
            constant_engine_F=15000.0,
            stall0_mps=120.0,
            cd0_s1=0.085,
            cd0_s2=0.32,
            cd0_s3=0.28,
            cd0_s4=0.22,
            cd0_hi=0.16
        )
        return MissilePhysics(params)
    
    def test_missile_straight_flight_compatibility(self, realistic_missile):
        """Test straight missile flight - compatibility with original test."""
        pos = Position(0.0, 0.0, 2000.0)
        speed = 800.0
        yaw = 0.0
        pitch = 0.0
        dt = 2.0

        res = realistic_missile.compute_movement(
            pos,
            yaw_deg=yaw,
            desired_hdg_deg=yaw,
            pitch_deg=pitch,
            desired_pitch_deg=pitch,
            speed_mps=speed,
            throttle=1.0,
            dt=dt
        )
        lat2, lon2, alt2, v2, yaw2, pitch2, roll2 = res

        # Should not slow down too much (has constant thrust)
        assert v2 > speed - 100, f"Missile slowed too much (Start {speed}, End {v2})"
        # Level flight should maintain altitude approximately
        assert abs(alt2 - pos.alt) < 100.0
        # Should move north significantly
        meters_north = (lat2 - pos.lat) * 111_000
        assert meters_north > 1400.0
    
    def test_missile_high_speed_performance(self, realistic_missile):
        """Test missile performance at high speeds."""
        pos = Position(0.0, 0.0, 8000.0)
        result = realistic_missile.compute_movement(
            pos, 0.0, 0.0, 0.0, 0.0, 1100.0, 1.0, 3.0
        )
        
        lat2, lon2, alt2, v2, yaw2, pitch2, roll2 = result
        
        # Should handle high speeds well
        assert v2 > 1000.0  # Should maintain high speed
        
        # Should cover significant distance
        distance = ((lat2 - pos.lat) * 111000)**2 + ((lon2 - pos.lon) * 111000)**2
        distance = math.sqrt(distance)
        assert distance > 3000.0  # Should cover at least 3km in 3 seconds
    
    def test_missile_climb_performance(self, realistic_missile):
        """Test missile climbing performance."""
        pos = Position(0.0, 0.0, 3000.0)
        result = realistic_missile.compute_movement(
            pos, 0.0, 0.0, 45.0, 45.0, 600.0, 1.0, 5.0
        )
        
        lat2, lon2, alt2, v2, yaw2, pitch2, roll2 = result
        
        # Should climb significantly
        assert alt2 > pos.alt + 1000.0
        
        # High thrust should maintain reasonable speed even in climb
        assert v2 > 400.0
    
    def test_missile_maneuverability(self, realistic_missile):
        """Test missile high-G maneuvering capability."""
        pos = Position(0.0, 0.0, 6000.0)
        
        # Large turn command
        result = realistic_missile.compute_movement(
            pos, 0.0, 180.0, 0.0, 0.0, 700.0, 1.0, 2.0
        )
        
        lat2, lon2, alt2, v2, yaw2, pitch2, roll2 = result
        
        # Should turn more aggressively than aircraft
        assert abs(yaw2) > 10.0  # Should make significant turn
        
        # Should have significant bank angle for turn
        assert abs(roll2) > 5.0
    
    def test_missile_no_energy_protections(self, realistic_missile):
        """Test that missiles don't have energy protections."""
        # Try to force low energy condition
        pos = Position(0.0, 0.0, 15000.0)
        low_speed = 150.0  # Close to stall
        
        result = realistic_missile.compute_movement(
            pos, 0.0, 0.0, 60.0, 60.0, low_speed, 1.0, 2.0
        )
        
        lat2, lon2, alt2, v2, yaw2, pitch2, roll2 = result
        
        # Without energy protections, missile might maintain extreme attitude
        # (Unlike aircraft which would force recovery)
        # Exact behavior depends on implementation, but should not crash
        assert all(math.isfinite(x) for x in result)
    
    def test_missile_thrust_to_weight_ratio(self, realistic_missile):
        """Test missile has high thrust-to-weight ratio."""
        # Calculate T/W ratio
        thrust = realistic_missile.constant_engine_F
        weight = realistic_missile.mass_kg * realistic_missile.g
        tw_ratio = thrust / weight
        
        # Missiles typically have T/W > 10
        assert tw_ratio > 8.0, f"T/W ratio too low: {tw_ratio:.2f}"
    
    def test_missile_acceleration_capability(self, realistic_missile):
        """Test missile acceleration from high thrust."""
        pos = Position(0.0, 0.0, 5000.0)
        initial_speed = 300.0
        
        result = realistic_missile.compute_movement(
            pos, 0.0, 0.0, 15.0, 15.0, initial_speed, 1.0, 3.0
        )
        
        lat2, lon2, alt2, v2, yaw2, pitch2, roll2 = result
        
        # Should accelerate significantly due to high thrust
        assert v2 > initial_speed + 50.0, f"Insufficient acceleration: {initial_speed} -> {v2}"


class TestMissilePhysicsEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def missile(self):
        params = MissilePhysics.Params(
            mass_kg=300.0,
            reference_area_m2=0.2,
            air=AirLayer()
        )
        return MissilePhysics(params)
    
    def test_extreme_speeds(self, missile):
        """Test handling of extreme speeds."""
        altitude = 5000.0

        # Very low speed
        cd_low = missile.get_base_drag_cd(10.0, altitude)
        assert math.isfinite(cd_low)
        assert cd_low > 0.0

        # Extremely high speed
        speed_high = 2000.0
        cd_high = missile.get_base_drag_cd(speed_high, altitude)  # ~Mach 5.9
        assert math.isfinite(cd_high)
        assert cd_high > 0.0

        # Should handle hypersonic drag reduction
        mach_high = speed_high / get_speed_of_sound(altitude)
        expected_reduction = 0.01 * (mach_high - missile.mach_t4)
        expected = missile.cd0_hi - expected_reduction
        assert math.isclose(cd_high, expected, rel_tol=1e-3)
    
    def test_extreme_altitudes(self, missile):
        """Test handling of extreme altitudes."""
        # Very high altitude
        pos_high = Position(0.0, 0.0, 30000.0)
        result = missile.compute_movement(
            pos_high, 0.0, 0.0, 0.0, 0.0, 500.0, 1.0, 2.0
        )
        
        # Should handle without crashing
        assert all(math.isfinite(x) for x in result)
        
        # Engine should still work (not altitude dependent)
        thrust = missile.get_engine_force(500.0, 30000.0, 1.0)
        assert thrust == missile.constant_engine_F
    
    def test_zero_speed_handling(self, missile):
        """Test handling of zero speed."""
        pos = Position(0.0, 0.0, 5000.0)
        result = missile.compute_movement(
            pos, 0.0, 90.0, 0.0, 15.0, 0.0, 1.0, 1.0
        )
        
        # Should handle gracefully
        lat2, lon2, alt2, v2, yaw2, pitch2, roll2 = result
        assert all(math.isfinite(x) for x in result)
        
        # Missile should accelerate from rest due to constant thrust
        # But initial movement might be minimal in first timestep
        assert v2 >= 0.0  # Speed should not go negative
    
    def test_extreme_mach_numbers(self, missile):
        """Test drag model at extreme Mach numbers."""
        # Test continuity and finite values at extreme Mach
        extreme_speeds = [50.0, 3000.0, 5000.0]  # Very low to hypersonic
        
        for speed in extreme_speeds:
            cd = missile.get_base_drag_cd(speed, 5000.0)
            assert math.isfinite(cd)
            assert cd > 0.0
            assert cd < 2.0  # Reasonable upper bound


class TestMissilePhysicsCompatibility:
    """Test compatibility with original missile physics tests."""
    
    @pytest.fixture
    def original_missile(self):
        """Missile matching original test parameters."""
        params = MissilePhysics.Params(
            mass_kg=500.0,
            reference_area_m2=0.3,
            aspect_ratio=3.0,
            oswald_e=0.75,
            n_max=30.0,
            max_speed_mps=1_400.0,
            stall0_mps=110.0,
            cd0_s1=0.08,
            cd0_s2=0.33,
            cd0_s3=0.30,
            cd0_s4=0.23,
            cd0_hi=0.17,
            constant_engine_F=20_000.0,
            air=AirLayer(),
        )
        return MissilePhysics(params)
    
    def test_ca_max_missile_compatibility(self, original_missile):
        """Test CA_max values - compatibility test."""
        # get_ca_max now uses updated realistic values: C_sub=2.9, C_low=1.65, C_high=1.20
        # Mach ~0.88 at sea level - should be in subsonic regime with blending
        assert 2.0 < original_missile.get_ca_max(300.0, 0) < 2.9
        # Mach ~2.06 at sea level - should be in low-supersonic regime
        assert 1.5 < original_missile.get_ca_max(700.0, 0) < 1.8
        # Mach ~3.53 at sea level - should be in high-supersonic regime
        assert 1.15 < original_missile.get_ca_max(1200.0, 0) < 1.3
    
    def test_pitch_limit_deg_missile_compatibility(self, original_missile):
        """Test pitch limits - compatibility test."""
        pos = Position(0.0, 0.0, 2000.0)
        limit = original_missile.get_pitch_limit_deg(pos, 400.0)
        assert 60.0 < limit <= 90.0  # Original test expected > 60°
        assert limit == 75.0  # Specific implementation value
    
    def test_smoothstep_missile_compatibility(self, original_missile):
        """Test smoothstep edge cases - compatibility test."""
        assert original_missile.smoothstep(0.0) == 0.0
        assert original_missile.smoothstep(1.0) == 1.0


class TestMissilePhysicsPerformanceEnvelope:
    """Test missile performance envelope characteristics."""
    
    @pytest.fixture
    def performance_missile(self):
        """Missile configured for performance testing."""
        params = MissilePhysics.Params(
            mass_kg=220.0,
            reference_area_m2=0.15,
            air=AirLayer(),
            n_max=45.0,
            constant_engine_F=18000.0,
            stall0_mps=135.0
        )
        return MissilePhysics(params)
    
    def test_high_g_capability(self, performance_missile):
        """Test missile high-G maneuvering capability."""
        # Missiles should sustain much higher G than aircraft
        pos = Position(0.0, 0.0, 8000.0)
        
        # Simulate high-G turn
        result = performance_missile.compute_movement(
            pos, 0.0, 120.0, 0.0, 0.0, 800.0, 1.0, 1.0
        )
        
        lat2, lon2, alt2, v2, yaw2, pitch2, roll2 = result
        
        # Should be able to make sharp turns
        yaw_change = abs(yaw2 - 0.0)
        assert yaw_change > 5.0  # Should turn more than aircraft
    
    def test_speed_envelope(self, performance_missile):
        """Test missile speed envelope."""
        # Should operate efficiently across wide speed range
        speeds = [200, 500, 800, 1100, 1400]
        
        for speed in speeds:
            pos = Position(0.0, 0.0, 6000.0)
            result = performance_missile.compute_movement(
                pos, 0.0, 0.0, 0.0, 0.0, speed, 1.0, 2.0
            )
            
            lat2, lon2, alt2, v2, yaw2, pitch2, roll2 = result
            
            # Should handle all speeds without issues
            assert all(math.isfinite(x) for x in result)
            assert v2 > speed * 0.8  # Should not slow down dramatically
    
    def test_altitude_envelope(self, performance_missile):
        """Test missile altitude envelope."""
        # Should operate across wide altitude range
        altitudes = [0, 5000, 10000, 15000, 20000, 25000]
        
        for alt in altitudes:
            pos = Position(0.0, 0.0, alt)
            result = performance_missile.compute_movement(
                pos, 0.0, 0.0, 10.0, 10.0, 600.0, 1.0, 3.0
            )
            
            lat2, lon2, alt2, v2, yaw2, pitch2, roll2 = result
            
            # Should handle all altitudes
            assert all(math.isfinite(x) for x in result)
            
            # Thrust should be constant regardless of altitude
            thrust = performance_missile.get_engine_force(600.0, alt, 1.0)
            assert thrust == performance_missile.constant_engine_F
    
    def test_energy_characteristics(self, performance_missile):
        """Test missile energy characteristics."""
        # High T/W should give excellent energy performance
        pos = Position(0.0, 0.0, 5000.0)
        
        # Simultaneous climb and acceleration
        result = performance_missile.compute_movement(
            pos, 0.0, 0.0, 30.0, 30.0, 300.0, 1.0, 4.0
        )
        
        lat2, lon2, alt2, v2, yaw2, pitch2, roll2 = result
        
        # Should both climb and accelerate
        assert alt2 > pos.alt + 500.0  # Significant climb
        assert v2 > 350.0  # Also accelerate
    
    def test_missile_vs_aircraft_comparison(self, performance_missile):
        """Compare missile characteristics to typical aircraft."""
        # Missiles should have:
        # 1. Higher G limits
        assert performance_missile.n_max > 30.0  # Much higher than aircraft ~9G
        
        # 2. Higher T/W ratio
        tw_ratio = performance_missile.constant_engine_F / (performance_missile.mass_kg * performance_missile.g)
        assert tw_ratio > 5.0  # Much higher than aircraft ~1.2
        
        # 3. No energy protections
        assert performance_missile.enable_energy_protections == False
        
        # 4. Higher pitch authority
        pitch_limit = performance_missile.get_pitch_limit_deg(None, 500.0)
        assert pitch_limit > 60.0  # Higher than typical aircraft limits
        
        # 5. Different drag characteristics (optimized for supersonic)
        drag_subsonic = performance_missile.get_base_drag_cd(300.0, 5000.0)
        drag_supersonic = performance_missile.get_base_drag_cd(600.0, 5000.0)
        # Should handle supersonic drag efficiently
        assert math.isfinite(drag_supersonic)
        assert drag_supersonic > 0.0