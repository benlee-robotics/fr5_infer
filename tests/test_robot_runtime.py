"""RobotRuntime state machine against FakeRobotClient, ticked in real time."""

import dataclasses
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pytest

from fr5_infer.config import InferConfig
from fr5_infer.robot_process import RobotRuntime, make_client
from fr5_infer.shm import RobotStateSlot, TrajectoryMailbox
from fr5_infer.trajectory import Trajectory
from teleop.control_math import euler_deg_to_quat


CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"
TICK_S = 0.008


@pytest.fixture
def config(tmp_path):
    # Self-contained: no dataset needed (the Docker test stage has none).
    conversion = tmp_path / "fr5_conversion.json"
    conversion.write_text(
        '{"options": {"fps": 30, "rotation": "rot6d", "action_mode": "command"},'
        ' "cameras": ["top", "wrist"]}'
    )
    return InferConfig.from_yaml(
        CONFIG,
        overrides={
            "mode": "fake",
            "policy.backend": "replay",
            "policy.conversion": str(conversion),
        },
    )


def make_runtime(config, client=None):
    context = mp.get_context("spawn")
    runtime = RobotRuntime(
        config,
        client or make_client(config),
        TrajectoryMailbox(context, config.execution.max_points),
        RobotStateSlot(context),
    )
    runtime.connect()
    return runtime


def run_ticks(runtime, seconds, until=None):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        runtime.tick(time.monotonic_ns())
        if until is not None and until(runtime):
            return True
        time.sleep(TICK_S)
    return False


def trajectory_from(tcp, offset_mm=(0.0, 0.0, 0.0), velocity_mm_s=(0.0, 0.0, 0.0),
                    duration_s=1.0, t0_ns=None):
    t0_ns = time.monotonic_ns() if t0_ns is None else t0_ns
    count = int(duration_s * 30) + 1
    t = np.arange(count) / 30.0
    position = np.asarray(tcp[:3]) + np.asarray(offset_mm) + np.outer(t, velocity_mm_s)
    quat = np.tile(euler_deg_to_quat(*tcp[3:6]), (count, 1))
    return Trajectory(t0_ns + (t * 1e9).astype(np.int64), position, quat)


def start(runtime):
    runtime.handle_command(("START", None), time.monotonic_ns())
    assert runtime.state == "STARTING"


def test_start_refused_when_first_target_is_far(config):
    runtime = make_runtime(config)
    start(runtime)
    runtime.mailbox.write(trajectory_from(runtime.measured_tcp, (50.0, 0, 0)), time.monotonic_ns(), False, -1)
    run_ticks(runtime, 0.1, until=lambda r: r.state != "STARTING")
    assert runtime.state == "IDLE"
    assert runtime.reason.startswith("start_offset_")
    assert not runtime.client.servo_active


def test_trajectory_written_before_start_is_ignored(config):
    runtime = make_runtime(config)
    old = time.monotonic_ns()
    runtime.mailbox.write(trajectory_from(runtime.measured_tcp, t0_ns=old), old, False, -1)
    time.sleep(0.01)
    start(runtime)
    run_ticks(runtime, 0.1)
    assert runtime.state == "STARTING"


def test_runs_then_stops_when_trajectory_is_exhausted(config):
    runtime = make_runtime(config)
    start(runtime)
    home = runtime.measured_tcp
    runtime.mailbox.write(
        trajectory_from(home, velocity_mm_s=(20.0, 0, 0), duration_s=0.3),
        time.monotonic_ns(), False, -1,
    )
    assert run_ticks(runtime, 0.2, until=lambda r: r.state == "RUNNING")
    assert runtime.client.servo_active
    assert run_ticks(runtime, 1.0, until=lambda r: r.state == "IDLE")
    assert runtime.reason == "trajectory_exhausted"
    assert not runtime.client.servo_active
    moved = runtime.client.current_tcp[0] - home[0]
    assert 3.0 < moved < 7.0  # 20 mm/s for 0.3 s


def test_velocity_limit_caps_each_step(config):
    config = dataclasses.replace(config, safety=dataclasses.replace(config.safety, max_velocity_mm_s=50.0))
    runtime = make_runtime(config)
    start(runtime)
    home = runtime.measured_tcp
    # Accepted start offset (15 mm) that the limiter must walk off slowly.
    runtime.mailbox.write(trajectory_from(home, (15.0, 0, 0), duration_s=2.0), time.monotonic_ns(), False, -1)
    positions, stamps = [], []
    end = time.monotonic() + 0.5
    while time.monotonic() < end:
        now = time.monotonic_ns()
        runtime.tick(now)
        if runtime.state == "RUNNING":
            positions.append(runtime.client.current_tcp[0])
            stamps.append(now)
        time.sleep(TICK_S)
    steps = np.diff(positions)
    dt = np.diff(stamps) / 1e9
    assert np.all(steps <= 50.0 * dt + 1e-6)
    assert positions[-1] - home[0] < 50.0 * 0.5 + 1e-6


def test_gripper_stops_servo_and_resumes_only_on_a_new_trajectory(config):
    runtime = make_runtime(config)
    start(runtime)
    home = runtime.measured_tcp
    now = time.monotonic_ns()
    runtime.mailbox.write(trajectory_from(home, duration_s=0.1, t0_ns=now), now, True, now + 50_000_000)
    assert run_ticks(runtime, 0.3, until=lambda r: r.state == "GRIPPER")
    assert not runtime.client.servo_active
    assert runtime.client.gripper_position == config.teleop.gripper.closed_position
    assert run_ticks(runtime, 1.5, until=lambda r: r.state == "STARTING")
    assert runtime.gripper_closed
    done_ns = runtime.gripper_done_ns
    # The trajectory from before the gripper action does not restart motion.
    run_ticks(runtime, 0.1)
    assert runtime.state == "STARTING"
    now = time.monotonic_ns()
    assert now > done_ns
    runtime.mailbox.write(trajectory_from(home, duration_s=0.5, t0_ns=now), now, True, -1)
    assert run_ticks(runtime, 0.2, until=lambda r: r.state == "RUNNING")
    assert runtime.client.servo_active


def test_shadow_mode_never_commands_the_robot(config):
    shadow = dataclasses.replace(config, mode="shadow")
    client = make_client(config)  # fake client standing in for the real robot
    runtime = make_runtime(shadow, client)
    start(runtime)
    home = runtime.measured_tcp
    now = time.monotonic_ns()
    runtime.mailbox.write(
        trajectory_from(home, velocity_mm_s=(20.0, 0, 0), duration_s=0.3, t0_ns=now),
        now, True, now + 200_000_000,
    )
    assert run_ticks(runtime, 0.2, until=lambda r: r.state == "RUNNING")
    runtime.tick(time.monotonic_ns())
    assert runtime.tick_cmd is not None  # computed and logged
    run_ticks(runtime, 0.3)
    assert not client.servo_active
    assert client.current_tcp == home
    assert client.gripper_position == config.teleop.gripper.open_position
    runtime.handle_command(("HOME",), time.monotonic_ns())
    assert runtime.reason == "home_refused_shadow_mode"


def test_stop_command_ends_the_servo(config):
    runtime = make_runtime(config)
    start(runtime)
    runtime.mailbox.write(trajectory_from(runtime.measured_tcp, duration_s=2.0), time.monotonic_ns(), False, -1)
    assert run_ticks(runtime, 0.2, until=lambda r: r.state == "RUNNING")
    runtime.handle_command(("STOP",), time.monotonic_ns())
    assert runtime.state == "IDLE"
    assert runtime.reason == "stopped_by_operator"
    assert not runtime.client.servo_active
