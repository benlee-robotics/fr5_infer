import numpy as np
import pytest

from fr5_infer.conversion import DecodedChunk
from fr5_infer.trajectory import GripperSchmitt, TemporalEnsemble, Trajectory, align_chunk, blend
from teleop.control_math import euler_deg_to_quat, quat_angle_deg


Q0 = euler_deg_to_quat(180.0, 0.0, 0.0)
Q1 = euler_deg_to_quat(180.0, 0.0, 30.0)


def line(t0_ns, count, start_x, step_x, period_ns=33_333_333, quat=Q0):
    t_ns = t0_ns + np.arange(count, dtype=np.int64) * period_ns
    position = np.zeros((count, 3))
    position[:, 0] = start_x + step_x * np.arange(count)
    return Trajectory(t_ns, position, np.tile(quat, (count, 1)))


def test_align_stamps_rows_from_the_observation_time():
    decoded = DecodedChunk(np.zeros((50, 3)), np.tile(Q0, (50, 1)), np.zeros(50))
    trajectory = align_chunk(decoded, 1_000_000_000, 30)
    assert trajectory.t_ns[0] == 1_000_000_000
    assert trajectory.t_ns[30] == 2_000_000_000
    assert len(trajectory.from_time(1_500_000_000)) == 35


def test_sample_interpolates_and_holds_the_ends():
    trajectory = Trajectory(
        np.array([0, 100], dtype=np.int64),
        np.array([[0.0, 0, 0], [10.0, 0, 0]]),
        np.array([Q0, Q1]),
    )
    position, orientation = trajectory.sample(50)
    assert position[0] == pytest.approx(5.0)
    assert quat_angle_deg(Q0, orientation) == pytest.approx(15.0, abs=1e-6)
    assert trajectory.sample(-10)[0][0] == 0.0
    assert trajectory.sample(500)[0][0] == 10.0


def test_blend_starts_on_the_old_trajectory_and_ends_on_the_new():
    old = line(0, 30, 0.0, 1.0)
    new = line(100_000_000, 30, 50.0, 1.0)  # 47 mm jump vs old at its start
    blended = blend(old, new, 0.15)
    old_at_start = old.sample(int(new.t_ns[0]))[0][0]
    assert blended.position_mm[0, 0] == pytest.approx(old_at_start)
    # After blend_s the new trajectory is used unchanged.
    after = new.t_ns >= new.t_ns[0] + 150_000_000
    np.testing.assert_allclose(blended.position_mm[after], new.position_mm[after])
    # Monotone transition, no overshoot.
    gap = new.position_mm[:, 0] - blended.position_mm[:, 0]
    assert np.all(np.diff(gap) <= 1e-9)


def test_blend_without_previous_returns_new():
    new = line(0, 5, 0.0, 1.0)
    assert blend(None, new, 0.15) is new


def test_schmitt_hysteresis_and_dwell():
    schmitt = GripperSchmitt(0.6, 0.4, min_dwell_s=0.5)
    schmitt.reset(False, 0)
    t_ns = np.arange(10, dtype=np.int64) * 100_000_000
    # 0.5 never crosses the close threshold.
    assert schmitt.first_switch(t_ns, np.full(10, 0.5)) is None
    raw = np.array([0.1, 0.7, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
    # Dwell: no switch before 0.5 s after the last change (t = 0).
    assert schmitt.first_switch(t_ns, raw) == 5
    schmitt.reset(False, -1_000_000_000)
    assert schmitt.first_switch(t_ns, raw) == 1
    schmitt.reset(True, -1_000_000_000)
    assert schmitt.first_switch(t_ns, np.full(10, 0.45)) is None
    assert schmitt.first_switch(t_ns, np.full(10, 0.3)) == 0


PERIOD_NS = 33_333_333


def flat(t0_ns, count, x, quat=Q0):
    return line(t0_ns, count, x, 0.0, quat=quat)


def test_ensemble_first_chunk_is_unchanged():
    ensemble = TemporalEnsemble(decay=0.0)
    new = line(0, 10, 0.0, 1.0)
    merged, gripper = ensemble.add(new, np.full(10, 0.2))
    np.testing.assert_allclose(merged.position_mm, new.position_mm)
    np.testing.assert_allclose(gripper, 0.2)


def test_ensemble_averages_overlapping_predictions():
    ensemble = TemporalEnsemble(decay=0.0)
    ensemble.add(flat(0, 30, 0.0), np.zeros(30))
    # Starts 20 steps later: its first 10 points overlap the first chunk.
    merged, gripper = ensemble.add(flat(20 * PERIOD_NS, 30, 10.0), np.ones(30))
    np.testing.assert_allclose(merged.position_mm[:10, 0], 5.0)
    np.testing.assert_allclose(merged.position_mm[10:, 0], 10.0)
    np.testing.assert_allclose(gripper[:10], 0.5)
    np.testing.assert_allclose(gripper[10:], 1.0)
    # A third prediction for the same times: the mean of all three.
    merged, _ = ensemble.add(flat(25 * PERIOD_NS, 5, 20.0), np.ones(5))
    np.testing.assert_allclose(merged.position_mm[:, 0], 10.0)


def test_ensemble_decay_weighs_older_predictions_more():
    ensemble = TemporalEnsemble(decay=1.0)
    ensemble.add(flat(0, 10, 0.0), np.zeros(10))
    merged, _ = ensemble.add(flat(0, 10, 10.0), np.zeros(10))
    # Weights 1 and exp(-1).
    expected = 10.0 * np.exp(-1.0) / (1.0 + np.exp(-1.0))
    np.testing.assert_allclose(merged.position_mm[:, 0], expected)


def test_ensemble_interpolates_off_grid_and_averages_orientation():
    ensemble = TemporalEnsemble(decay=0.0)
    ensemble.add(line(0, 10, 0.0, 1.0), np.zeros(10))
    # Half a step off the first chunk's grid.
    merged, _ = ensemble.add(flat(PERIOD_NS // 2, 3, 0.5, quat=Q1), np.zeros(3))
    # Old at t = 0.5 step is x = 0.5, so the mean is 0.5 as well.
    assert merged.position_mm[0, 0] == pytest.approx(0.5, abs=1e-6)
    assert quat_angle_deg(Q0, tuple(merged.quat_xyzw[0])) == pytest.approx(15.0, abs=1e-6)


def test_ensemble_reset_starts_over():
    ensemble = TemporalEnsemble(decay=0.0)
    ensemble.add(flat(0, 10, 0.0), np.zeros(10))
    ensemble.reset()
    merged, _ = ensemble.add(flat(0, 10, 10.0), np.zeros(10))
    np.testing.assert_allclose(merged.position_mm[:, 0], 10.0)


def test_rollout_log_pads_short_chunks(tmp_path):
    from fr5_infer.rollout_log import EpisodeLog

    episode_log = EpisodeLog(tmp_path, {"mode": "fake"}, save_images=False)
    for length in (50, 50, 7):
        episode_log.add_inference({"latency_s": 0.1, "chunk": np.ones((length, 10))}, None)
    episode_log.close("trajectory_exhausted")
    saved = np.load(tmp_path / "inferences.npz")
    assert saved["chunk"].shape == (3, 50, 10)
    assert np.isnan(saved["chunk"][2, 7:]).all()
