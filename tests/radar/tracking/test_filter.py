import numpy as np
import pytest

from bvr_marl_core.radar.tracking.filter.filters import (
    ConstantVelocityKFFilter,
    CoordinatedTurnKFFilter,
    IMMFilter,
)


def test_cv_kf_update_predict_cycle():
    meas = np.array([1.0, 2.0, 3.0])
    kf = ConstantVelocityKFFilter(meas, dt=1.0)
    kf.predict(1.0)
    kf.update(np.array([1.1, 2.1, 3.1]))
    state = kf.get_state()
    assert np.allclose(state[:3], [1.1, 2.1, 3.1], atol=0.5)


def test_ct_kf_predict_update():
    state = np.zeros(8)
    kf = CoordinatedTurnKFFilter(state, np.eye(8), np.eye(8) * 0.1, np.eye(3) * 0.1, dt=1.0)
    kf.predict(1.0)
    kf.update(np.zeros(3))
    out = kf.get_state()
    assert out.shape[0] in (6, 8)  # depends on internal state size


def test_imm_blends_filters():
    meas = np.array([2.0, 0.0, -2.0])
    ct = CoordinatedTurnKFFilter(
        np.hstack((meas, [0, 0, 0, 0, 0])), np.eye(8), np.eye(8) * 0.1, np.eye(3) * 0.1, dt=1.0
    )
    cv = ConstantVelocityKFFilter(meas, dt=1.0)
    imm = IMMFilter([ct, cv], np.array([[0.9, 0.1], [0.1, 0.9]]), [0.5, 0.5], np.eye(3) * 0.1)
    imm.predict(1.0)
    imm.update(meas + 0.1)
    s = imm.get_state()
    assert s.shape[0] in (6, 8)
    probs = imm.mode_probabilities
    assert np.isclose(np.sum(probs), 1.0)


def test_cv_kf_accuracy_straight_line():
    """Test that the CV filter correctly estimates uniform straight-line motion."""
    np.random.seed(42)
    true_pos = np.zeros(3)
    true_vel = np.array([5.0, -2.0, 1.0])
    dt = 1.0
    kf = ConstantVelocityKFFilter(
        true_pos, dt=dt, process_noise_std=1e-2, measurement_noise_var=1e-2
    )
    for i in range(30):
        true_pos = true_pos + true_vel * dt
        measurement = true_pos + np.random.normal(0, 0.1, 3)
        kf.predict(dt)
        kf.update(measurement)
    est = kf.get_state()
    assert np.allclose(est[:3], true_pos, atol=0.2), f"CV position: {est[:3]} vs {true_pos}"
    assert np.allclose(est[3:], true_vel, atol=0.2), f"CV velocity: {est[3:]} vs {true_vel}"


def test_ct_kf_accuracy_turn():
    """Test that the CT filter correctly estimates a circular trajectory."""
    np.random.seed(43)
    R = 100.0
    omega = 0.05
    V = omega * R
    dt = 1.0
    T = 10
    yaw = 0.0
    pos = np.array([R * np.cos(yaw), R * np.sin(yaw), 0.0])
    x0 = np.array([pos[0], pos[1], pos[2], V, yaw, 0, omega, 0])
    kf = CoordinatedTurnKFFilter(x0, np.eye(8), np.eye(8) * 1e-3, np.eye(3) * 1e-2, dt=dt)

    print(
        f"{'step':>4} | {'true_x':>9} {'true_y':>9} {'true_vx':>9} {'true_vy':>9} | "
        f"{'kf_x':>9} {'kf_y':>9} {'kf_vx':>9} {'kf_vy':>9} | {'speed_true':>10} {'speed_est':>10}"
    )

    for i in range(T):
        yaw = yaw + omega * dt
        pos = np.array([R * np.cos(yaw), R * np.sin(yaw), 0.0])
        vx_true = -R * omega * np.sin(yaw)
        vy_true = R * omega * np.cos(yaw)
        speed_true = np.sqrt(vx_true**2 + vy_true**2)
        meas = pos + np.random.normal(0, 0.2, 3)
        kf.predict(dt)
        kf.update(meas)
        est = kf.get_state()
        kf_x, kf_y = est[0], est[1]
        kf_vx, kf_vy = est[3], est[4]
        speed_est = np.sqrt(kf_vx**2 + kf_vy**2)
        print(
            f"{i:4d} | {pos[0]:9.3f} {pos[1]:9.3f} {vx_true:9.3f} {vy_true:9.3f} | "
            f"{kf_x:9.3f} {kf_y:9.3f} {kf_vx:9.3f} {kf_vy:9.3f} | "
            f"{speed_true:10.3f} {speed_est:10.3f}"
        )

    # Finaler Check
    assert np.allclose(est[:3], pos, atol=10.0), f"CT-Position: {est[:3]} vs {pos}"
    speed_est = np.linalg.norm(est[3:])
    assert np.allclose(speed_est, V, atol=2.0), f"CT-Speed: {speed_est} vs {V}"


def test_imm_filter_accuracy_mix():
    """Test the IMM filter switching between straight-line and circular motion."""
    np.random.seed(44)
    # Segment 1: straight line, Segment 2: circular arc
    seg1_steps = 15
    seg2_steps = 25
    total_steps = seg1_steps + seg2_steps
    dt = 1.0
    pos = np.zeros(3)
    vel = np.array([4.0, 0.0, 0.0])
    # Circular arc parameters
    R = 60.0
    omega = 0.08
    V_circ = omega * R
    yaw = 0.0

    ct = CoordinatedTurnKFFilter(
        np.hstack((pos, [V_circ, yaw, 0, omega, 0])),
        np.eye(8),
        np.eye(8) * 1e-3,
        np.eye(3) * 1e-2,
        dt,
    )
    cv = ConstantVelocityKFFilter(pos, dt, process_noise_std=1e-2, measurement_noise_var=1e-2)
    imm = IMMFilter([ct, cv], np.array([[0.97, 0.03], [0.03, 0.97]]), [0.5, 0.5], np.eye(3) * 1e-2)

    for i in range(total_steps):
        if i < seg1_steps:
            pos = pos + vel * dt
            meas = pos + np.random.normal(0, 0.15, 3)
        else:
            yaw = yaw + omega * dt
            pos = np.array([R * np.cos(yaw) + vel[0] * seg1_steps, R * np.sin(yaw), 0.0])
            meas = pos + np.random.normal(0, 0.2, 3)
        imm.predict(dt)
        imm.update(meas)
    est = imm.get_state()
    assert np.allclose(est[:3], pos, atol=10), f"IMM position: {est[:3]} vs {pos}"


# =============================================================================
# NIS / get_last_update_stats tests (Confidence v2)
# =============================================================================


def test_cv_filter_nis_cached_after_update():
    """CV filter must cache NIS and expose it via get_last_update_stats()."""
    meas = np.array([100.0, 200.0, 300.0])
    kf = ConstantVelocityKFFilter(meas, dt=1.0, process_noise_std=10.0, measurement_noise_var=100.0)
    kf.predict(1.0)
    kf.update(meas + np.array([5.0, 0.0, 0.0]))

    stats = kf.get_last_update_stats()
    assert "nis" in stats, "get_last_update_stats() must return a dict with 'nis' key"
    nis = stats["nis"]
    assert isinstance(nis, float), f"NIS must be a float, got {type(nis)}"
    assert nis >= 0.0, f"NIS must be non-negative, got {nis}"
    # With a 5m innovation and measurement std=10m, NIS should be moderate
    assert nis < 1000.0, f"NIS sanity check failed: {nis}"


def test_cv_filter_nis_zero_innovation():
    """When the measurement exactly matches the prediction, NIS should be near zero."""
    meas = np.array([0.0, 0.0, 0.0])
    kf = ConstantVelocityKFFilter(meas, dt=1.0, process_noise_std=1.0, measurement_noise_var=1.0)
    # Do one predict then update with the same measurement (no motion, near-zero innovation)
    kf.predict(1.0)
    kf.update(meas)  # innovation ≈ 0 because predicted pos ≈ 0 (no velocity)
    nis = kf.get_last_update_stats()["nis"]
    assert nis >= 0.0
    # NIS should be very small for near-zero innovation
    assert nis < 5.0, f"Expected small NIS for zero-innovation update, got {nis}"


def test_ct_filter_nis_cached_after_update():
    """CT filter must cache NIS and expose it via get_last_update_stats()."""
    state = np.zeros(8)
    kf = CoordinatedTurnKFFilter(
        state, np.eye(8) * 100.0, np.eye(8) * 0.1, np.eye(3) * 25.0, dt=1.0
    )
    kf.predict(1.0)
    kf.update(np.array([10.0, 0.0, 0.0]))

    stats = kf.get_last_update_stats()
    assert "nis" in stats
    nis = stats["nis"]
    assert isinstance(nis, float)
    assert nis >= 0.0


def test_imm_filter_get_last_update_stats_delegates_to_dominant():
    """IMM get_last_update_stats() must return stats from the dominant mode."""
    meas = np.array([0.0, 0.0, 0.0])
    ct = CoordinatedTurnKFFilter(np.zeros(8), np.eye(8), np.eye(8) * 0.1, np.eye(3) * 0.1, dt=1.0)
    cv = ConstantVelocityKFFilter(meas, dt=1.0)
    # Start with CV strongly dominant (prob=[0.1, 0.9])
    imm = IMMFilter([ct, cv], np.array([[0.9, 0.1], [0.1, 0.9]]), [0.1, 0.9], np.eye(3) * 0.1)
    imm.predict(1.0)
    imm.update(meas + 0.1)

    stats = imm.get_last_update_stats()
    assert "nis" in stats, "IMM must expose 'nis' from dominant subfilter"
    assert isinstance(stats["nis"], float)
    assert stats["nis"] >= 0.0
