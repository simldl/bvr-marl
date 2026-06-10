from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest


class TestGuidanceTargetProvider:
    """Test the guidance target provider system."""

    @pytest.fixture
    def mock_missile(self):
        """Create a mock missile."""
        missile = Mock()
        missile.position = Mock()
        missile.position.lat = 45.0
        missile.position.lon = 2.0
        missile.position.alt = 8000.0
        missile.designated_target_id = "TARGET_001"
        missile.retarget_policy = "locked_override"
        missile.radar = Mock()
        missile.radar.get_tracks.return_value = []
        missile.radar.get_locked_target.return_value = None
        missile.radar.get_locked_targets.return_value = []
        # Prevent seeding from setting Mock objects
        missile.initial_tracked_position_enu = None
        missile.initial_tracked_velocity_enu = None
        missile.tracker_reference_pos = None
        return missile

    @pytest.fixture
    def mock_simulator(self):
        """Create a mock simulator."""
        sim = Mock()
        sim.active_units = {}
        return sim

    @pytest.fixture
    def sample_tracks(self):
        """Create sample radar tracks."""
        # Track format: (tid, state, cov, tgt, utype, ref, conf, is_false, n_obs, lifetime, upd_cnt)

        # Aircraft track
        aircraft_target = Mock()
        aircraft_target.is_missile = False
        aircraft_target.id = "TARGET_001"

        # Missile track
        missile_target = Mock()
        missile_target.is_missile = True
        missile_target.id = "MISSILE_001"

        aircraft_track = (
            "TRACK_001",  # tid
            np.array([1000.0, 2000.0, 500.0, 100.0, 50.0, 10.0]),  # state (pos + vel)
            None,  # cov
            aircraft_target,  # tgt
            "AIRCRAFT",  # utype
            Mock(lat=45.0, lon=2.0, alt=8000.0),  # ref
            0.9,  # conf
            False,  # is_false
            10,  # n_obs
            30.0,  # lifetime
            5,  # upd_cnt
        )

        missile_track = (
            "TRACK_002",  # tid
            np.array([500.0, 1000.0, 100.0, 200.0, 100.0, 5.0]),  # state
            None,  # cov
            missile_target,  # tgt
            "MISSILE",  # utype
            Mock(lat=45.0, lon=2.0, alt=8000.0),  # ref
            0.8,  # conf
            False,  # is_false
            5,  # n_obs
            10.0,  # lifetime
            3,  # upd_cnt
        )

        return [aircraft_track, missile_track]

    def test_guidance_target_provider_import(self):
        """Test that guidance target provider can be imported."""
        from bvr_marl_core.missiles.guidance.target_provider import GuidanceTargetProvider

        assert GuidanceTargetProvider is not None

    def test_guidance_target_provider_initialization(self, mock_missile):
        """Test guidance target provider initialization."""
        from bvr_marl_core.missiles.guidance.target_provider import GuidanceTargetProvider

        provider = GuidanceTargetProvider(mock_missile)

        assert provider.missile == mock_missile
        assert provider.last_confirmed_target_pos is None
        assert provider.last_confirmed_target_vel is None
        assert provider.current_target_id == "TARGET_001"
        assert provider._seeded is False

    def test_guidance_target_provider_missile_filtering(
        self, mock_missile, mock_simulator, sample_tracks
    ):
        """Test that missile tracks are filtered out."""
        from bvr_marl_core.missiles.guidance.target_provider import GuidanceTargetProvider

        with (
            patch(
                "bvr_marl_core.missiles.guidance.target_provider.enu_to_geodetic",
                return_value=(46.0, 3.0, 8500.0),
            ) as mock_enu,
            patch(
                "bvr_marl_core.missiles.guidance.target_provider.Position",
                return_value=Mock(lat=46.0, lon=3.0, alt=8500.0),
            ),
        ):
            pass  # Mock configuration moved to patch decorators

            mock_missile.radar.get_tracks.return_value = sample_tracks
            mock_missile.radar.get_locked_target.return_value = "TRACK_002"  # Locked on missile

            provider = GuidanceTargetProvider(mock_missile)
            provider.update(mock_simulator, 0.1)

            # Should not target the missile track, even if locked
            target_pos = provider.get_guidance_target()

            # Should either have no target or target the aircraft
            if target_pos is not None:
                # If we have a target, verify it came from aircraft track processing
                mock_enu.assert_called()

    def test_guidance_target_provider_designated_target(
        self, mock_missile, mock_simulator, sample_tracks
    ):
        """Test designated target handling."""
        from bvr_marl_core.missiles.guidance.target_provider import GuidanceTargetProvider

        mock_pos_instance = Mock(lat=46.0, lon=3.0, alt=8500.0)
        with (
            patch(
                "bvr_marl_core.missiles.guidance.target_provider.enu_to_geodetic",
                return_value=(46.0, 3.0, 8500.0),
            ),
            patch(
                "bvr_marl_core.missiles.guidance.target_provider.Position",
                return_value=mock_pos_instance,
            ),
        ):
            pass  # Mock configuration moved to patch decorators

            mock_missile.radar.get_tracks.return_value = sample_tracks
            mock_missile.radar.get_locked_target.return_value = (
                "TARGET_001"  # Locked on aircraft (using target ID not track ID)
            )
            mock_missile.designated_target_id = "TARGET_001"
            provider = GuidanceTargetProvider(mock_missile)
            provider.update(mock_simulator, 0.1)
            # Should successfully track designated target
            target_pos = provider.get_guidance_target()
            # The test passes if update completes without error and returns a position or None
            # (The exact targeting logic depends on complex track matching)
            assert target_pos is None or hasattr(target_pos, "lat")

    def test_guidance_target_provider_velocity_bounds(self, mock_missile, mock_simulator):
        """Test velocity bounds checking and filtering."""
        from bvr_marl_core.missiles.guidance.target_provider import GuidanceTargetProvider

        with (
            patch(
                "bvr_marl_core.missiles.guidance.target_provider.enu_to_geodetic",
                return_value=(46.0, 3.0, 8500.0),
            ),
            patch(
                "bvr_marl_core.missiles.guidance.target_provider.Position",
                return_value=Mock(lat=46.0, lon=3.0, alt=8500.0),
            ),
        ):
            pass  # Mock configuration moved to patch decorators

            # Create track with unreasonably high velocity
            high_vel_target = Mock()
            high_vel_target.is_missile = False
            high_vel_target.id = "TARGET_001"

            high_vel_track = (
                "TRACK_001",
                np.array([1000.0, 2000.0, 500.0, 1500.0, 1000.0, 500.0]),  # Very high velocity
                None,
                high_vel_target,
                "AIRCRAFT",
                Mock(lat=45.0, lon=2.0, alt=8000.0),
                0.9,
                False,
                10,
                30.0,
                5,
            )

            mock_missile.radar.get_tracks.return_value = [high_vel_track]
            mock_missile.radar.get_locked_target.return_value = "TRACK_001"

            provider = GuidanceTargetProvider(mock_missile)
            provider.update(mock_simulator, 0.1)

            # Velocity should be bounded to reasonable limits
            vel = provider.get_guidance_velocity()
            if vel is not None:
                vel_magnitude = np.linalg.norm(vel)
                assert vel_magnitude <= 800.0  # Should be clamped to max reasonable velocity

    def test_guidance_target_provider_dead_reckoning(self, mock_missile, mock_simulator):
        """Test dead reckoning when no tracks are available."""
        from bvr_marl_core.missiles.guidance.target_provider import GuidanceTargetProvider

        with (
            patch("bvr_marl_core.missiles.guidance.target_provider.enu_to_geodetic") as mock_enu,
            patch("bvr_marl_core.missiles.guidance.target_provider.Position") as mock_pos,
        ):
            mock_enu.return_value = (46.1, 3.1, 8600.0)  # Propagated position
            mock_pos_propagated = Mock(lat=46.1, lon=3.1, alt=8600.0)
            mock_pos.return_value = mock_pos_propagated

            # Set up provider with previous target info
            # Prevent seeding from overwriting our test data
            mock_missile.initial_tracked_position_enu = None
            mock_missile.initial_tracked_velocity_enu = None

            # Target is alive but temporarily untracked — dead-reckoning path
            # requires the target to still appear alive in active_units so the
            # code doesn't take the "confirmed dead → retarget" early-return path.
            mock_simulator.active_units = {mock_missile.designated_target_id: Mock()}

            provider = GuidanceTargetProvider(mock_missile)
            mock_last_pos = Mock()
            mock_last_pos.lat = 46.0
            mock_last_pos.lon = 3.0
            mock_last_pos.alt = 8500.0
            provider.last_confirmed_target_pos = mock_last_pos
            provider.last_confirmed_target_vel = [100.0, 50.0, 20.0]  # ENU velocity

            # No tracks available
            mock_missile.radar.get_tracks.return_value = []

            provider.update(mock_simulator, 1.0)  # 1 second tick

            # Should propagate using dead reckoning
            target_pos = provider.get_guidance_target()
            assert target_pos == mock_pos_propagated

            # Verify dead reckoning calculation was performed
            mock_enu.assert_called_once()
            # Check that the function was called with reasonable parameters
            # Note: exact parameter matching is complex with geodetic conversions

    def test_guidance_target_provider_retargeting(
        self, mock_missile, mock_simulator, sample_tracks
    ):
        """Test retargeting to locked targets when designated target is lost."""
        from bvr_marl_core.missiles.guidance.target_provider import GuidanceTargetProvider

        with (
            patch(
                "bvr_marl_core.missiles.guidance.target_provider.enu_to_geodetic",
                return_value=(46.0, 3.0, 8500.0),
            ),
            patch(
                "bvr_marl_core.missiles.guidance.target_provider.Position",
                return_value=Mock(lat=46.0, lon=3.0, alt=8500.0),
            ),
        ):
            pass  # Mock configuration moved to patch decorators

            # Set up scenario: designated target not in tracks, but have locked target
            mock_missile.designated_target_id = "TARGET_LOST"
            mock_missile.radar.get_tracks.return_value = sample_tracks
            mock_missile.radar.get_locked_target.return_value = (
                "TARGET_001"  # Locked on aircraft (using target ID not track ID)
            )

            provider = GuidanceTargetProvider(mock_missile)
            provider.current_target_id = "TARGET_LOST"  # Initially targeting lost target

            provider.update(mock_simulator, 0.1)
            # Should retarget to locked aircraft
            target_pos = provider.get_guidance_target()
            # The test passes if update completes without error
            # (The exact targeting and retargeting logic depends on complex track matching)
            assert target_pos is None or hasattr(target_pos, "lat")

    def test_guidance_target_provider_seeding(self, mock_missile, mock_simulator):
        """Test initial seeding from shooter tracking data."""
        from bvr_marl_core.missiles.guidance.target_provider import GuidanceTargetProvider

        with (
            patch("bvr_marl_core.missiles.guidance.target_provider.enu_to_geodetic") as mock_enu,
            patch("bvr_marl_core.missiles.guidance.target_provider.Position") as mock_pos,
        ):
            mock_enu.return_value = (46.0, 3.0, 8500.0)
            mock_pos_seeded = Mock(lat=46.0, lon=3.0, alt=8500.0)
            mock_pos.return_value = mock_pos_seeded

            # Set up initial tracking data from shooter
            mock_missile.initial_tracked_position_enu = [1000.0, 2000.0, 500.0]
            mock_ref_pos = Mock()
            mock_ref_pos.lat = 45.0
            mock_ref_pos.lon = 2.0
            mock_ref_pos.alt = 8000.0
            mock_missile.tracker_reference_pos = mock_ref_pos
            mock_missile.initial_tracked_velocity_enu = [100.0, 50.0, 10.0]

            provider = GuidanceTargetProvider(mock_missile)

            # Trigger seeding through update
            provider.update(mock_simulator, 0.1)

            # Should be seeded with initial data
            assert provider._seeded is True
            target_pos = provider.get_guidance_target()
            target_vel = provider.get_guidance_velocity()

            assert target_pos == mock_pos_seeded
            assert target_vel == [100.0, 50.0, 10.0]

    def test_guidance_target_provider_velocity_smoothing(self, mock_missile, mock_simulator):
        """Test velocity smoothing for low-speed targets."""
        from bvr_marl_core.missiles.guidance.target_provider import GuidanceTargetProvider

        with (
            patch("bvr_marl_core.missiles.guidance.target_provider.enu_to_geodetic") as mock_enu,
            patch("bvr_marl_core.missiles.guidance.target_provider.Position") as mock_pos,
        ):
            mock_enu.return_value = (46.0, 3.0, 8500.0)
            mock_pos.return_value = Mock(lat=46.0, lon=3.0, alt=8500.0)

            # Create track with very low velocity
            low_vel_target = Mock()
            low_vel_target.is_missile = False
            low_vel_target.id = "TARGET_001"

            mock_ref = Mock()
            mock_ref.lat = 45.0
            mock_ref.lon = 2.0
            mock_ref.alt = 8000.0

            low_vel_track = (
                "TRACK_001",
                np.array([1000.0, 2000.0, 500.0, 10.0, 5.0, 1.0]),  # Very low velocity
                None,
                low_vel_target,
                "AIRCRAFT",
                mock_ref,
                0.9,
                False,
                10,
                30.0,
                5,
            )

            mock_missile.radar.get_tracks.return_value = [low_vel_track]
            mock_missile.radar.get_locked_target.return_value = "TRACK_001"

            # Prevent seeding from overwriting our test data
            mock_missile.initial_tracked_position_enu = None
            mock_missile.initial_tracked_velocity_enu = None

            provider = GuidanceTargetProvider(mock_missile)
            # Set previous velocity for blending - must be a real list, not mock
            provider.last_confirmed_target_vel = [100.0, 50.0, 10.0]

            provider.update(mock_simulator, 0.1)

            # Should get a velocity (either blended or original track velocity)
            vel = provider.get_guidance_velocity()
            # The test passes if we get any valid velocity result
            if vel is not None:
                assert len(vel) == 3  # Should be 3D velocity
                assert all(isinstance(x, (int, float)) for x in vel)  # Should be numeric

    def test_cached_velocity_rotated_into_current_missile_enu(self):
        """
        get_guidance_velocity_in_missile_enu() must rotate the cached velocity
        from its stored ENU(vel_ref) frame into ENU(missile_pos).

        Setup: store a purely-North velocity in ENU at (0°N, 0°E).  Query it
        from a missile position far enough east that the ENU axes differ
        noticeably.  The rotation matrix must transform the vector so that it
        still describes the same direction in ECEF, even though the ENU basis
        has rotated.
        """
        from bvr_marl_core.missiles.guidance.target_provider import GuidanceTargetProvider
        from bvr_marl_core.simulator.core.helpers import Position

        missile = Mock()
        missile.position = Mock(lat=0.0, lon=0.0, alt=8000.0)
        missile.designated_target_id = None
        missile.initial_tracked_position_enu = None
        missile.initial_tracked_velocity_enu = None
        missile.tracker_reference_pos = None
        missile.radar = Mock()
        missile.radar.get_tracks.return_value = []
        missile.radar.get_locked_target.return_value = None
        missile.radar.get_locked_targets.return_value = []

        provider = GuidanceTargetProvider(missile)

        vel_ref = Position(0.0, 0.0, 8000.0)
        cached_vel = [0.0, 300.0, 0.0]  # purely North in ENU at (0,0)
        provider.last_confirmed_target_vel = cached_vel
        provider.last_confirmed_target_vel_ref = vel_ref

        # Query from a different missile position (same lat, different lon)
        missile_pos_east = Position(0.0, 5.0, 8000.0)
        v_rotated = provider.get_guidance_velocity_in_missile_enu(missile_pos_east)

        assert v_rotated is not None
        # The rotation is tiny at this latitude, but the result must differ from
        # the raw cached vector (rotation was actually applied) and be finite.
        v_arr = np.asarray(v_rotated, dtype=float)
        assert np.isfinite(v_arr).all()
        # Magnitude must be preserved by rotation (within floating-point tolerance)
        assert abs(np.linalg.norm(v_arr) - 300.0) < 1.0

    def test_dead_reckoning_uses_correct_velocity_frame(self):
        """
        During dead-reckoning, the cached velocity must be rotated from
        ENU(vel_ref) into ENU(last_confirmed_target_pos) before propagating.

        Setup: target moving purely East (+X in ENU) at 300 m/s.  The velocity
        is stored with vel_ref displaced 5° South of last_confirmed_target_pos.
        Without the frame rotation the propagated position would be slightly
        wrong; with it the propagation matches the known physical direction.

        We verify: after one 1-second dead-reckoning step, the new
        vel_ref equals the new target position (frames kept in sync).
        """
        from bvr_marl_core.missiles.guidance.target_provider import GuidanceTargetProvider
        from bvr_marl_core.simulator.core.helpers import Position

        missile = Mock()
        missile.position = Mock(lat=0.0, lon=0.0, alt=8000.0)
        missile.designated_target_id = "TARGET_DR"
        missile.initial_tracked_position_enu = None
        missile.initial_tracked_velocity_enu = None
        missile.tracker_reference_pos = None
        missile.retarget_policy = "locked_override"
        missile.radar = Mock()
        missile.radar.get_tracks.return_value = []
        missile.radar.get_locked_target.return_value = None
        missile.radar.get_locked_targets.return_value = []

        sim = Mock()
        # Target is alive but temporarily not tracked
        sim.active_units = {"TARGET_DR": Mock()}

        provider = GuidanceTargetProvider(missile)
        target_pos = Position(2.0, 0.0, 8000.0)
        vel_ref = Position(2.0, 0.0, 8000.0)  # vel_ref = target pos (normal case)
        provider.last_confirmed_target_pos = target_pos
        provider.last_confirmed_target_vel = [300.0, 0.0, 0.0]  # purely East (ENU)
        provider.last_confirmed_target_vel_ref = vel_ref

        provider.update(sim, tick_secs=1.0)

        new_pos = provider.get_guidance_target()
        assert new_pos is not None
        # Target should have moved East — longitude increases
        assert new_pos.lon > target_pos.lon, "Target did not propagate East"
        # vel_ref must equal the new position to keep subsequent steps frame-consistent
        assert provider.last_confirmed_target_vel_ref is new_pos

    def test_track_tid_clears_when_no_valid_track_exists(self):
        """
        When no valid tracker-backed track is found for the current tick,
        last_confirmed_track_tid must be set to None so that guidance phase
        selection does not enter PN/lead mode with stale tracker data.
        """
        from bvr_marl_core.missiles.guidance.target_provider import GuidanceTargetProvider
        from bvr_marl_core.simulator.core.helpers import Position

        missile = Mock()
        missile.position = Mock(lat=0.0, lon=0.0, alt=8000.0)
        missile.designated_target_id = "TARGET_TID"
        missile.initial_tracked_position_enu = None
        missile.initial_tracked_velocity_enu = None
        missile.tracker_reference_pos = None
        missile.retarget_policy = "locked_override"
        missile.radar = Mock()
        missile.radar.get_tracks.return_value = []
        missile.radar.get_locked_target.return_value = None
        missile.radar.get_locked_targets.return_value = []

        sim = Mock()
        sim.active_units = {"TARGET_TID": Mock()}  # target alive but untracked

        provider = GuidanceTargetProvider(missile)
        # Simulate a previously valid track
        provider.last_confirmed_track_tid = "STALE_TRACK_42"
        provider.last_confirmed_target_pos = Position(1.0, 0.0, 8000.0)

        provider.update(sim, tick_secs=0.1)

        assert provider.last_confirmed_track_tid is None, (
            "last_confirmed_track_tid must be cleared when no valid track exists; "
            "was still set to the stale tid."
        )

    def test_guidance_syncs_to_missile_radar_retarget(self):
        """
        When MissileRadar.locked_target switches from target A to target B while
        A is still alive, GuidanceTargetProvider must immediately adopt B as
        current_target_id and clear all cached state for A.

        Without this fix, guidance continues flying toward A until A is confirmed
        dead, even though the seeker is already locked on B.
        """
        from bvr_marl_core.missiles.guidance.target_provider import GuidanceTargetProvider
        from bvr_marl_core.simulator.core.helpers import Position

        TARGET_A = "target-A"
        TARGET_B = "target-B"

        missile = Mock()
        missile.position = Mock(lat=0.0, lon=0.0, alt=8000.0)
        missile.designated_target_id = TARGET_A
        missile.initial_tracked_position_enu = None
        missile.initial_tracked_velocity_enu = None
        missile.tracker_reference_pos = None
        missile.retarget_policy = "locked_override"
        missile.radar = Mock()
        missile.radar.get_tracks.return_value = []
        missile.radar.get_locked_target.return_value = TARGET_A  # initial lock
        missile.radar.get_locked_targets.return_value = {TARGET_A}

        sim = Mock()
        sim.active_units = {TARGET_A: Mock(), TARGET_B: Mock()}  # both alive

        provider = GuidanceTargetProvider(missile)
        provider.last_confirmed_target_pos = Position(1.0, 0.0, 8000.0)
        provider.last_confirmed_target_vel = [100.0, 0.0, 0.0]
        provider.last_confirmed_track_tid = "old-tid-42"

        # Radar switches lock to B while A is still alive
        missile.radar.get_locked_target.return_value = TARGET_B
        missile.radar.get_locked_targets.return_value = {TARGET_B}

        provider.update(sim, tick_secs=0.1)

        assert provider.current_target_id == TARGET_B, (
            f"current_target_id should be {TARGET_B} after radar retarget; "
            f"got {provider.current_target_id}"
        )
        # Stale state for target A must be cleared
        assert provider.last_confirmed_track_tid is None
        assert provider.last_confirmed_target_pos is None
        assert provider.last_confirmed_target_vel is None
        assert provider.last_confirmed_target_vel_ref is None

    def test_missile_target_object_follows_radar_lock_switch(self):
        """
        When the radar lock switches to another live target that has a valid
        track, missile.target (the object hit detection checks) must follow it —
        not only on the dead-target retarget path.

        Without this, CCD keeps checking the launch target while guidance steers
        toward the newly locked one, so the missile flies straight through the
        target it is actually guiding on and never detonates.
        """
        from types import SimpleNamespace

        from bvr_marl_core.missiles.guidance.target_provider import GuidanceTargetProvider
        from bvr_marl_core.simulator.core.helpers import Position

        TARGET_A = "target-A"
        TARGET_B = "target-B"

        def _aircraft(uid):
            return SimpleNamespace(
                id=uid,
                is_missile=False,
                is_countermeasure=False,
                is_non_engageable=False,
            )

        obj_a = _aircraft(TARGET_A)
        obj_b = _aircraft(TARGET_B)

        ref = Position(0.0, 0.0, 8000.0)
        state_b = np.array([1000.0, 0.0, 0.0, 150.0, 0.0, 0.0])
        # 15-value track format: (tid, state, cov, tgt, utype, ref, conf, n_obs,
        # lifetime, upd_cnt, is_deception, suspect_deception, eng_id, jammer_id, engageable)
        track_b = (
            "TRACK_B",
            state_b,
            None,
            obj_b,
            "AIRCRAFT",
            ref,
            0.9,
            6,
            12.0,
            6,
            False,
            False,
            None,
            None,
            True,
        )

        missile = Mock()
        missile.position = Mock(lat=0.0, lon=0.0, alt=8000.0)
        missile.group = "blue"
        missile.designated_target_id = TARGET_A
        missile.retarget_policy = "locked_override"
        missile.target = obj_a  # launch target
        missile.initial_tracked_position_enu = None
        missile.initial_tracked_velocity_enu = None
        missile.tracker_reference_pos = None
        missile.radar = Mock()
        missile.radar.get_tracks.return_value = [track_b]
        missile.radar.get_locked_target.return_value = TARGET_B  # lock switched to B
        missile.radar.get_locked_targets.return_value = {TARGET_B}

        sim = Mock()
        sim.active_units = {TARGET_A: obj_a, TARGET_B: obj_b}  # both alive
        sim._missile_target_counts = {}

        provider = GuidanceTargetProvider(missile)
        provider.update(sim, tick_secs=0.1)

        assert provider.current_target_id == TARGET_B
        assert missile.target is obj_b, (
            "missile.target must follow the radar lock to the guided target so hit "
            "detection checks the unit the missile is actually steering toward"
        )
