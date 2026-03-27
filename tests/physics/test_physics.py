#!/usr/bin/env python3
"""
Comprehensive tests for physics.physics module.
Tests all classes and functions including AirLayer, ForceModel, PhysicsParams, and BasePhysics.
"""

import math

import numpy as np
import pytest

from air_to_air_rl.physics.physics import (
    AIRSPEED_OF_SOUND,
    SEA_LEVEL_DENSITY,
    STANDARD_GRAVITY,
    AirLayer,
    BasePhysics,
    ForceModel,
    PhysicsParams,
)


class TestAirLayer:
    """Test AirLayer class functionality."""

    @pytest.fixture
    def air_layer(self):
        return AirLayer()

    @pytest.fixture
    def custom_air_layer(self):
        return AirLayer(sea_level_density=1.3)

    def test_init_default(self, air_layer):
        """Test AirLayer initialization with default parameters."""
        assert air_layer.rho0 == SEA_LEVEL_DENSITY
        assert hasattr(air_layer, "_alt_table")
        assert hasattr(air_layer, "_rho_table")
        assert len(air_layer._alt_table) == len(air_layer._rho_table)

    def test_init_custom_density(self, custom_air_layer):
        """Test AirLayer initialization with custom sea level density."""
        assert custom_air_layer.rho0 == 1.3

    def test_density_scalar_sea_level(self):
        """Test density calculation at sea level."""
        density = AirLayer._density_scalar(0.0)
        assert math.isclose(density, SEA_LEVEL_DENSITY, rel_tol=1e-3)

    def test_density_scalar_troposphere(self):
        """Test density calculation in troposphere (< 11km)."""
        density = AirLayer._density_scalar(5000.0)
        expected = SEA_LEVEL_DENSITY * math.exp(-5000.0 / 8500.0)
        assert math.isclose(density, expected, rel_tol=1e-3)

    def test_density_scalar_stratosphere_lower(self):
        """Test density calculation in lower stratosphere (11-20km)."""
        density = AirLayer._density_scalar(15000.0)
        rho11 = SEA_LEVEL_DENSITY * math.exp(-11000.0 / 8500.0)
        expected = rho11 * math.exp(-(15000.0 - 11000.0) / 6500.0)
        assert math.isclose(density, expected, rel_tol=1e-3)

    def test_density_scalar_stratosphere_upper(self):
        """Test density calculation in upper stratosphere (> 20km)."""
        density = AirLayer._density_scalar(25000.0)
        rho11 = SEA_LEVEL_DENSITY * math.exp(-11000.0 / 8500.0)
        rho20 = rho11 * math.exp(-(20000.0 - 11000.0) / 6500.0)
        expected = rho20 * math.exp(-(25000.0 - 20000.0) / 6000.0)
        assert math.isclose(density, expected, rel_tol=1e-3)

    @pytest.mark.parametrize(
        "altitude, rho_min, rho_max",
        [
            (0, 1.10, 1.30),
            (5_000, 0.60, 0.90),
            (11_000, 0.25, 0.40),
            (20_000, 0.05, 0.15),
            (25_000, 0.02, 0.07),
        ],
    )
    def test_density_range(self, air_layer, altitude, rho_min, rho_max):
        """Test density values are within expected ranges."""
        rho = air_layer.get_density(altitude)
        assert rho_min <= rho <= rho_max, f"ρ({altitude} m) = {rho:.3f} kg/m³"

    def test_density_monotonic_decrease(self, air_layer):
        """Test that density decreases monotonically with altitude."""
        altitudes = np.linspace(0, 25000, 100)
        densities = [air_layer.get_density(alt) for alt in altitudes]
        assert all(x > y for x, y in zip(densities, densities[1:])), (
            "Density must decrease monotonically"
        )

    def test_get_density_interpolation(self, air_layer):
        """Test density interpolation within table range."""
        density_0 = air_layer.get_density(0.0)
        density_100 = air_layer.get_density(100.0)
        density_50 = air_layer.get_density(50.0)

        assert density_0 > density_50 > density_100

    def test_get_density_outside_table(self, air_layer):
        """Test density calculation outside table range."""
        density_high = air_layer.get_density(30000.0)
        expected = air_layer._density_scalar(30000.0, air_layer.rho0)
        assert math.isclose(density_high, expected, rel_tol=1e-3)

    def test_compute_density_vec(self, air_layer):
        """Test vectorized density computation."""
        altitudes = np.array([0, 5000, 10000, 15000, 20000])
        densities = air_layer._compute_density_vec(altitudes)

        assert len(densities) == len(altitudes)
        assert all(densities[i] > densities[i + 1] for i in range(len(densities) - 1))


class TestForceModel:
    """Test ForceModel class functionality."""

    @pytest.fixture
    def air_layer(self):
        return AirLayer()

    @pytest.fixture
    def force_model(self, air_layer):
        def get_base_drag_cd(v_mps, alt_m):
            return 0.02

        def get_engine_force(v_mps, alt_m, throttle):
            return 100_000 * throttle

        return ForceModel(
            mass_kg=10000,
            reference_area_m2=27.0,
            gravity=STANDARD_GRAVITY,
            air=air_layer,
            cd0=0.02,
            K_ind=0.05,
            get_base_drag_cd=get_base_drag_cd,
            get_engine_force=get_engine_force,
        )

    def test_init(self, force_model, air_layer):
        """Test ForceModel initialization."""
        assert force_model.mass_kg == 10000
        assert force_model.A_m2 == 27.0
        assert force_model.g == STANDARD_GRAVITY
        assert force_model.air == air_layer
        assert force_model.cd0 == 0.02
        assert force_model.K_ind == 0.05
        assert callable(force_model.get_base_drag_cd)
        assert callable(force_model.get_engine_force)

    def test_compute_total_drag_level_flight(self, force_model):
        """Test total drag computation for level flight (load factor = 1)."""
        drag = force_model.compute_total_drag(200, 1000, 1.0)
        assert drag > 0.0

        # Drag should increase with speed
        drag_higher_speed = force_model.compute_total_drag(300, 1000, 1.0)
        assert drag_higher_speed > drag

    def test_compute_total_drag_high_g(self, force_model):
        """Test total drag computation with high load factor."""
        drag_1g = force_model.compute_total_drag(200, 1000, 1.0)
        drag_7g = force_model.compute_total_drag(200, 1000, 7.0)

        # Higher load factor should increase drag due to induced drag
        assert drag_7g > drag_1g

    def test_compute_total_drag_zero_dynamic_pressure(self, force_model):
        """Test total drag computation with very low speed."""
        drag = force_model.compute_total_drag(0.001, 1000, 3.0)
        # Should handle near-zero dynamic pressure gracefully
        assert math.isfinite(drag)
        assert drag >= 0.0

    def test_compute_specific_energy_rate_level(self, force_model):
        """Test specific energy rate for level flight."""
        Ps = force_model.compute_specific_energy_rate(200, 1000, 1.0, 1.0)
        assert math.isfinite(Ps)

        # With full throttle and reasonable drag, should be positive
        assert Ps > 0.0

    def test_compute_specific_energy_rate_climb(self, force_model):
        """Test specific energy rate for climbing flight."""
        # Level flight: n=1.0 (no extra lift needed)
        Ps_level = force_model.compute_specific_energy_rate(250, 5000, 1.0, 1.0)
        # Climbing: n>1.0 increases drag, reducing energy rate
        Ps_climb = force_model.compute_specific_energy_rate(250, 5000, 1.2, 1.0)

        # Climbing (higher load factor = more drag) should reduce specific energy rate
        assert Ps_climb < Ps_level

    def test_compute_specific_energy_rate_descent(self, force_model):
        """Test specific energy rate for descending flight."""
        # Level flight: n=1.0
        Ps_level = force_model.compute_specific_energy_rate(250, 5000, 1.0, 1.0)
        # Descending: n<1.0 reduces drag, increasing energy rate
        Ps_descent = force_model.compute_specific_energy_rate(250, 5000, 0.8, 1.0)

        # Descending (lower load factor = less drag) should increase specific energy rate
        assert Ps_descent > Ps_level

    def test_throttle_effect(self, force_model):
        """Test effect of throttle on specific energy rate."""
        Ps_idle = force_model.compute_specific_energy_rate(200, 1000, 1.0, 0.0)
        Ps_half = force_model.compute_specific_energy_rate(200, 1000, 1.0, 0.5)
        Ps_full = force_model.compute_specific_energy_rate(200, 1000, 1.0, 1.0)

        assert Ps_idle < Ps_half < Ps_full


class TestPhysicsParams:
    """Test PhysicsParams dataclass."""

    def test_default_params(self):
        """Test PhysicsParams with minimal required parameters."""
        params = PhysicsParams(mass_kg=10000.0, reference_area_m2=25.0)

        assert params.mass_kg == 10000.0
        assert params.reference_area_m2 == 25.0
        assert params.gravity_m_s2 == STANDARD_GRAVITY
        assert isinstance(params.air, AirLayer)
        assert params.cd0 == 0.01
        assert params.aspect_ratio == 7.5
        assert params.oswald_e == 0.82

    def test_custom_params(self):
        """Test PhysicsParams with custom values."""
        air = AirLayer(1.3)
        params = PhysicsParams(
            mass_kg=15000.0,
            reference_area_m2=30.0,
            gravity_m_s2=9.81,
            air=air,
            cd0=0.015,
            aspect_ratio=6.0,
            oswald_e=0.75,
            max_speed_mps=400.0,
            n_max=12.0,
        )

        assert params.mass_kg == 15000.0
        assert params.reference_area_m2 == 30.0
        assert params.gravity_m_s2 == 9.81
        assert params.air == air
        assert params.cd0 == 0.015
        assert params.aspect_ratio == 6.0
        assert params.oswald_e == 0.75
        assert params.max_speed_mps == 400.0
        assert params.n_max == 12.0

    def test_engine_thrust_profile(self):
        """Test engine thrust profile parameter."""
        profile = {"full_thrust_speed": 200.0, "optimal_speed": 300.0}
        params = PhysicsParams(
            mass_kg=10000.0, reference_area_m2=25.0, engine_thrust_profile=profile
        )

        assert params.engine_thrust_profile == profile

    def test_stall_parameters(self):
        """Test stall-related parameters."""
        params = PhysicsParams(
            mass_kg=10000.0,
            reference_area_m2=25.0,
            stall0_mps=65.0,
            service_ceiling_m=15000.0,
            stall_sink_pitch_deg=-10.0,
            stall_margin_mps=15.0,
        )

        assert params.stall0_mps == 65.0
        assert params.service_ceiling_m == 15000.0
        assert params.stall_sink_pitch_deg == -10.0
        assert params.stall_margin_mps == 15.0


class TestBasePhysics:
    """Test BasePhysics class functionality."""

    @pytest.fixture
    def base_physics(self):
        params = PhysicsParams(mass_kg=12000.0, reference_area_m2=28.0)
        return BasePhysics(params)

    def test_init(self, base_physics):
        """Test BasePhysics initialization."""
        assert base_physics.params is not None
        assert base_physics.params.mass_kg == 12000.0
        assert base_physics.params.reference_area_m2 == 28.0


class TestConstants:
    """Test physics constants."""

    def test_constants_values(self):
        """Test that physics constants have expected values."""
        assert SEA_LEVEL_DENSITY == 1.225
        assert STANDARD_GRAVITY == 9.80665
        assert AIRSPEED_OF_SOUND == 340.29

    def test_constants_types(self):
        """Test that constants are numeric."""
        assert isinstance(SEA_LEVEL_DENSITY, (int, float))
        assert isinstance(STANDARD_GRAVITY, (int, float))
        assert isinstance(AIRSPEED_OF_SOUND, (int, float))


class TestIntegration:
    """Integration tests for physics module components."""

    @pytest.fixture
    def integrated_system(self):
        """Create an integrated physics system for testing."""
        air = AirLayer()
        params = PhysicsParams(
            mass_kg=15000.0, reference_area_m2=30.0, air=air, aspect_ratio=6.0, oswald_e=0.8
        )

        def get_base_drag_cd(v_mps, alt_m):
            mach = v_mps / AIRSPEED_OF_SOUND
            if mach < 0.8:
                return 0.02
            else:
                return 0.025

        def get_engine_force(v_mps, alt_m, throttle):
            rho_rel = air.get_density(alt_m) / air.rho0
            return throttle * 120000.0 * rho_rel**0.7

        force_model = ForceModel(
            mass_kg=params.mass_kg,
            reference_area_m2=params.reference_area_m2,
            gravity=params.gravity_m_s2,
            air=params.air,
            cd0=params.cd0,
            K_ind=1.0 / (math.pi * params.aspect_ratio * params.oswald_e),
            get_base_drag_cd=get_base_drag_cd,
            get_engine_force=get_engine_force,
        )

        return {"air": air, "params": params, "force_model": force_model}

    def test_altitude_effects_on_performance(self, integrated_system):
        """Test how altitude affects aircraft performance."""
        force_model = integrated_system["force_model"]

        # Test at different altitudes
        altitudes = [0, 5000, 10000, 15000]
        speeds = [250.0] * len(altitudes)

        drag_values = []
        thrust_values = []

        for alt, speed in zip(altitudes, speeds):
            drag = force_model.compute_total_drag(speed, alt, 1.0)
            thrust = force_model.get_engine_force(speed, alt, 1.0)

            drag_values.append(drag)
            thrust_values.append(thrust)

        # Drag should decrease with altitude (lower air density)
        assert drag_values[0] > drag_values[-1]

        # Thrust should decrease with altitude
        assert thrust_values[0] > thrust_values[-1]

    def test_speed_effects_on_drag(self, integrated_system):
        """Test how speed affects drag characteristics."""
        force_model = integrated_system["force_model"]

        speeds = [100, 200, 300, 400, 500]
        drags = [force_model.compute_total_drag(speed, 5000, 1.0) for speed in speeds]

        # Drag should generally increase with speed, but allow for transonic effects
        # Check that drag is positive and reasonable
        for drag in drags:
            assert drag > 0.0

        # At least the overall trend should show drag increasing with speed
        assert drags[-1] > drags[0]  # Drag at 500 m/s > drag at 100 m/s

    def test_energy_balance_level_flight(self, integrated_system):
        """Test energy balance for steady level flight."""
        force_model = integrated_system["force_model"]

        # Find approximate equilibrium throttle for level flight
        speed = 250.0
        altitude = 8000.0

        # In steady level flight, thrust should approximately equal drag
        drag = force_model.compute_total_drag(speed, altitude, 1.0)

        # Find throttle that gives thrust close to drag
        for throttle in np.linspace(0.1, 1.0, 10):
            thrust = force_model.get_engine_force(speed, altitude, throttle)
            if abs(thrust - drag) < 1000:  # Within 1kN
                Ps = force_model.compute_specific_energy_rate(speed, altitude, 1.0, throttle)
                assert abs(Ps) < 10.0  # Should be close to zero for steady flight
                break
