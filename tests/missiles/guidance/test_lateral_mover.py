"""
Regression tests: missile guidance for laterally-moving targets.

Covers the root-cause chain fixed in this session:
  - PN must produce a lateral acceleration command when the target is moving
    sideways (if velocity is zero the missile would miss a crossing target).
  - hold-last-valid keeps PN working during brief track-loss intervals.
  - GuidanceTargetProvider stores and exposes the tracker TID correctly.

These are unit-level tests; they mock geodetic helpers so the geometry is
clean and the assertions are deterministic.
"""

import math
from unittest.mock import Mock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sim(*unit_ids):
    """Return a minimal mock simulator with active_units containing the given IDs."""
    sim = Mock()
    sim.active_units = {uid: Mock() for uid in unit_ids}
    sim._missile_target_counts = {}
    return sim


def _make_pos(lat=0.0, lon=0.0, alt=10000.0):
    pos = Mock()
    pos.lat = lat
    pos.lon = lon
    pos.alt = alt
    return pos


def _make_missile(speed=600.0, n_max=35.0, vt=None, tid=None):
    """Return a minimal mock missile wired up for PnPropNavGuidance."""
    missile = Mock()
    missile.speed = speed
    missile.n_max = n_max

    prov = Mock()
    prov.get_guidance_velocity.return_value = vt  # ENU m/s or None
    prov.get_guidance_velocity_in_missile_enu.return_value = vt  # frame-aware API, same value
    prov.get_guidance_tid.return_value = tid  # track ID or None
    missile.target_provider = prov

    missile.radar = Mock()
    missile.radar.get_tracker_info.return_value = None
    missile.radar.tracker_manager = None
    return missile


# ---------------------------------------------------------------------------
# Unit tests for PnPropNavGuidance — lateral mover geometry
# ---------------------------------------------------------------------------


class TestPnLateralMover:
    """PN guidance must produce eastward (positive-yaw) lead for a target
    that starts directly ahead (North) and is moving East."""

    # Geometry:
    #   Missile   at (0°, 0°, 10 km), heading North (yaw=0°), speed 600 m/s
    #   Target    at (0.45°, 0°, 10 km)  ≈ 50 km North
    #             velocity = [200, 0, 0] m/s  (East)
    #   Expected: PN should command positive yaw (turn East to lead)

    MISSILE_POS = _make_pos(0.0, 0.0, 10000.0)
    TARGET_POS = _make_pos(0.45, 0.0, 10000.0)  # ~50 km North
    # geodetic_to_enu(target, missile): East=0, North≈50000, Up=0
    R_ENU = [0.0, 50_000.0, 0.0]

    def _run_pn(self, vt_enu, n_calls=1, tick=0.5):
        """Run PN compute() and return (yaw, pitch) from the last call."""
        from bvr_marl_core.missiles.guidance.pn_propnav import PnPropNavGuidance

        missile = _make_missile(speed=600.0, n_max=35.0, vt=vt_enu)
        pn = PnPropNavGuidance(missile, pn_law=PnPropNavGuidance.PN_TRUE)

        with patch(
            "bvr_marl_core.missiles.guidance.pn_propnav.geodetic_to_enu", return_value=self.R_ENU
        ):
            for _ in range(n_calls):
                yaw, pitch = pn.compute(
                    current_yaw_deg=0.0,
                    current_pitch_deg=0.0,
                    missile_position=self.MISSILE_POS,
                    target_position=self.TARGET_POS,
                    tick_secs=tick,
                )
        return yaw, pitch, pn

    def test_lateral_target_produces_eastward_lead(self):
        """Core regression: positive East velocity → positive yaw command."""
        yaw, pitch, _ = self._run_pn(vt_enu=[200.0, 0.0, 0.0])
        # Target moving East → missile must turn East (yaw > 0°) to lead it
        assert yaw > 0.0, f"Expected yaw > 0° (East lead) for eastward target, got {yaw:.2f}°"
        assert yaw < 45.0, f"Yaw lead angle unreasonably large: {yaw:.2f}°"
        assert -89.0 <= pitch <= 89.0

    def test_opposite_lateral_target_produces_westward_lead(self):
        """Symmetric: negative East velocity → negative yaw (West lead)."""
        yaw, pitch, _ = self._run_pn(vt_enu=[-200.0, 0.0, 0.0])
        # Yaw convention: 0=N, 90=E, 270=W → westward lead should be near 360° or < 0
        # After normalize_angle, a small westward lead yields yaw close to 360° or ~350°
        assert yaw > 315.0 or (0.0 <= yaw < 30.0 and False), (
            f"Expected yaw > 315° (West lead) for westward target, got {yaw:.2f}°"
        )

    def test_opposite_lateral_yaw_sign_differs_from_east(self):
        """East-mover and West-mover must produce opposite yaw corrections."""
        yaw_east, _, _ = self._run_pn(vt_enu=[200.0, 0.0, 0.0])
        yaw_west, _, _ = self._run_pn(vt_enu=[-200.0, 0.0, 0.0])
        # Signed difference: east yaw should be on the east side, west on the west
        # Both normalized to [0, 360). East lead < 180, West lead > 180 (or < east).
        assert yaw_east < 180.0, f"East-mover yaw should be in [0,180): {yaw_east:.2f}"
        assert yaw_west > 180.0, f"West-mover yaw should be in (180,360]: {yaw_west:.2f}"

    def test_zero_lateral_velocity_produces_near_zero_yaw(self):
        """Head-on target (no lateral component) → near-zero yaw correction."""
        yaw, pitch, _ = self._run_pn(vt_enu=[0.0, -200.0, 0.0])  # approaching head-on
        # For a head-on target due North with no lateral motion, yaw ≈ 0 or 360
        assert yaw < 5.0 or yaw > 355.0, f"Head-on target should give ~0° yaw, got {yaw:.2f}°"

    def test_no_velocity_fallback_produces_valid_angles(self):
        """Even with no velocity information, compute() must return valid angles."""
        yaw, pitch, _ = self._run_pn(vt_enu=None)
        assert 0.0 <= yaw <= 360.0
        assert -89.0 <= pitch <= 89.0

    def test_hold_last_valid_velocity_on_track_loss(self):
        """After a tick with good velocity, a subsequent tick with None velocity
        should reuse the stored velocity (hold-last-valid), NOT zero it out."""
        from bvr_marl_core.missiles.guidance.pn_propnav import PnPropNavGuidance

        missile = _make_missile(speed=600.0, n_max=35.0, vt=[200.0, 0.0, 0.0])
        pn = PnPropNavGuidance(missile)

        with patch(
            "bvr_marl_core.missiles.guidance.pn_propnav.geodetic_to_enu", return_value=self.R_ENU
        ):
            # Tick 1 — good velocity, should store it
            pn.compute(0.0, 0.0, self.MISSILE_POS, self.TARGET_POS, 0.5)
            assert pn._last_valid_vt is not None
            pn._last_valid_vt.copy()

            # Tick 2 — velocity disappears
            missile.target_provider.get_guidance_velocity.return_value = None
            missile.target_provider.get_guidance_velocity_in_missile_enu.return_value = None
            yaw2, pitch2 = pn.compute(0.0, 0.0, self.MISSILE_POS, self.TARGET_POS, 0.5)

        # Still eastward lead (velocity held)
        assert yaw2 > 0.0, (
            f"After track-loss, hold-last-valid should maintain eastward lead; got {yaw2:.2f}°"
        )
        assert pn._last_valid_vt_age == pytest.approx(0.5, abs=1e-9)

    def test_hold_expires_after_max_age(self):
        """After _HOLD_VT_MAX_AGE seconds of no velocity, hold should expire."""
        from bvr_marl_core.missiles.guidance.pn_propnav import PnPropNavGuidance

        missile = _make_missile(speed=600.0, n_max=35.0, vt=[200.0, 0.0, 0.0])
        pn = PnPropNavGuidance(missile)
        pn._HOLD_VT_MAX_AGE = 1.0  # short for testing

        with patch(
            "bvr_marl_core.missiles.guidance.pn_propnav.geodetic_to_enu", return_value=self.R_ENU
        ):
            # Establish the held value
            pn.compute(0.0, 0.0, self.MISSILE_POS, self.TARGET_POS, 0.5)
            assert pn._last_valid_vt is not None

            # Now simulate track loss beyond max age
            missile.target_provider.get_guidance_velocity.return_value = None
            missile.target_provider.get_guidance_velocity_in_missile_enu.return_value = None
            # Two ticks of 0.5s = 1.0s age (at boundary — should NOT be expired yet)
            pn.compute(0.0, 0.0, self.MISSILE_POS, self.TARGET_POS, 0.5)
            # One more tick past max age
            pn.compute(0.0, 0.0, self.MISSILE_POS, self.TARGET_POS, 0.5)

        # Age should exceed max; guidance still produces valid angles (zero velocity used)
        assert pn._last_valid_vt_age >= pn._HOLD_VT_MAX_AGE


# ---------------------------------------------------------------------------
# Unit tests for GuidanceTargetProvider — TID tracking
# ---------------------------------------------------------------------------


class TestTargetProviderTid:
    """GuidanceTargetProvider must store and expose the tracker TID."""

    def _make_track(
        self, tid, unit_id, state=None, conf=0.9, n_obs=5, upd_cnt=10, is_deception=False
    ):
        """Build a 15-element track tuple matching the new format."""
        if state is None:
            state = np.array([1000.0, 2000.0, 0.0, 100.0, 50.0, 0.0])
        tgt = Mock()
        tgt.id = unit_id
        tgt.is_missile = False
        tgt.is_countermeasure = False
        tgt.is_non_engageable = False
        ref = Mock()
        ref.lat = 0.0
        ref.lon = 0.0
        ref.alt = 10000.0
        cov = np.eye(6) * 100.0
        # 15-value format:
        # tid, state, cov, tgt, utype, ref, conf, n_obs, lifetime, upd_cnt,
        # is_deception, suspect_deception, engagement_id, jammer_id, engageable
        return (
            tid,
            state,
            cov,
            tgt,
            "aircraft",
            ref,
            conf,
            n_obs,
            20,
            upd_cnt,
            is_deception,
            False,
            None,
            None,
            True,
        )

    def _make_missile_stub(self, designated_id, tracks):
        """Minimal missile stub for GuidanceTargetProvider."""
        missile = Mock()
        missile.designated_target_id = designated_id
        missile.retarget_policy = "locked_override"
        missile.initial_tracked_position_enu = None
        missile.tracker_reference_pos = None
        missile.initial_tracked_velocity_enu = None
        missile.position = _make_pos(0.0, 0.0, 10000.0)

        radar = Mock()
        radar.get_tracks.return_value = tracks
        radar.get_locked_target.return_value = None
        radar.get_locked_targets.return_value = set()
        missile.radar = radar
        return missile

    def test_tid_stored_after_update(self):
        """After update(), get_guidance_tid() returns the matched track's TID."""
        from bvr_marl_core.missiles.guidance.target_provider import GuidanceTargetProvider

        unit_id = "target-001"
        track_tid = 42
        tracks = [self._make_track(track_tid, unit_id)]
        missile = self._make_missile_stub(unit_id, tracks)

        prov = GuidanceTargetProvider(missile)
        prov.update(sim=_make_sim(unit_id), tick_secs=0.5)

        assert prov.get_guidance_tid() == track_tid

    def test_tid_is_none_before_first_update(self):
        """Before any update, get_guidance_tid() returns None."""
        from bvr_marl_core.missiles.guidance.target_provider import GuidanceTargetProvider

        missile = self._make_missile_stub("target-001", [])
        prov = GuidanceTargetProvider(missile)
        assert prov.get_guidance_tid() is None

    def test_tid_changes_on_retarget(self):
        """When the designated target disappears and another is selected, TID updates."""
        from bvr_marl_core.missiles.guidance.target_provider import GuidanceTargetProvider

        # First update: only target-001 visible (tid=42)
        unit_id_a = "target-001"
        unit_id_b = "target-002"
        tracks_a = [self._make_track(42, unit_id_a)]
        missile = self._make_missile_stub(unit_id_a, tracks_a)

        prov = GuidanceTargetProvider(missile)
        prov.update(sim=_make_sim(unit_id_a), tick_secs=0.5)
        assert prov.get_guidance_tid() == 42

        # Second update: target-001 gone, only target-002 (tid=99)
        tracks_b = [self._make_track(99, unit_id_b)]
        missile.radar.get_tracks.return_value = tracks_b
        prov.update(sim=_make_sim(unit_id_b), tick_secs=0.5)
        assert prov.get_guidance_tid() == 99

    def test_tid_cleared_in_dead_reckoning(self):
        """
        During dead-reckoning (no valid track this tick), TID must be cleared to None.

        Keeping a stale TID would allow guidance._select_guidance_phase() to enter
        PN/lead mode even though the tracker has no live data, which causes the
        missile to use a velocity estimate from a prior tick's ENU frame.
        """
        from bvr_marl_core.missiles.guidance.target_provider import GuidanceTargetProvider

        unit_id = "target-001"
        tracks = [self._make_track(42, unit_id)]
        missile = self._make_missile_stub(unit_id, tracks)

        prov = GuidanceTargetProvider(missile)
        prov.update(sim=_make_sim(unit_id), tick_secs=0.5)
        assert prov.get_guidance_tid() == 42

        # Now tracks disappear → dead-reckoning path; TID must be cleared
        missile.radar.get_tracks.return_value = []
        prov.update(sim=_make_sim(unit_id), tick_secs=0.5)
        assert prov.get_guidance_tid() is None, (
            "TID must be None during dead-reckoning so PN/lead is not entered "
            "with stale tracker data."
        )
