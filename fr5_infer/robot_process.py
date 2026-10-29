"""8 ms ServoCart loop in its own process (no torch, never waits on inference).

States::

    IDLE ─HOME─▶ HOMING ─arrived/stop─▶ IDLE
    IDLE ─START─▶ STARTING ─trajectory ok─▶ RUNNING ⇄ GRIPPER ─▶ STARTING ...
    any ─STOP / exhausted / timeout─▶ IDLE,  errors ─▶ FAULT ─RESET─▶ IDLE

Every servo target passes the fr5_teleop limits (``limit_home_step``: step,
angular step, workspace) plus an absolute orientation range around home.
A trajectory that runs out stops the servo instead of holding the last
target. The gripper follows the fr5_teleop sequence: servo end, MoveGripper,
wait, and resume only on a trajectory inferred after the gripper finished.

In shadow mode the robot is connected but never commanded: targets are
computed, limited and logged as if they were sent.
"""

from __future__ import annotations

import logging
import time
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

import numpy as np

from teleop.control_math import (
    HomeTrajectory,
    Quaternion,
    TcpPose,
    Vector3,
    clamp_orientation_deviation,
    euler_deg_to_quat,
    limit_home_step,
    quat_angle_deg,
    quat_to_euler_deg,
)
from teleop.robot.client import RobotClient, RobotStateSnapshot

from fr5_infer.config import InferConfig
from fr5_infer.conversion import normalize_gripper
from fr5_infer.shm import MailboxContents, RobotState, RobotStateSlot, TrajectoryMailbox


log = logging.getLogger(__name__)

# Same margins as the fr5_teleop worker home return.
HOME_LIMIT_MARGIN = 1.5
HOME_ARRIVE_MM = 0.01
HOME_ARRIVE_DEG = 0.01
HOME_SETTLE_TIMEOUT_S = 3.0
# Without a pushed state packet, poll GetActualTCPPose this often.
TCP_POLL_PERIOD_NS = 50_000_000
EPISODE_STATES = {"STARTING", "RUNNING", "GRIPPER"}

TICK_DTYPE = np.dtype(
    [
        ("t_ns", "<i8"),
        ("state", "|u1"),
        ("tcp", "<f8", (6,)),
        ("gripper_position", "<f8"),
        ("gripper_closed", "|u1"),
        ("desired", "<f8", (6,)),
        ("cmd", "<f8", (6,)),
        ("cmd_valid", "|u1"),
        ("traj_seq", "<i8"),
    ]
)
STATE_CODES = ("IDLE", "HOMING", "STARTING", "RUNNING", "GRIPPER", "FAULT")


def make_client(config: InferConfig) -> RobotClient:
    if config.mode == "fake":
        from teleop.robot.fake_client import FakeRobotBehavior, FakeRobotClient

        home = config.teleop.home
        client = FakeRobotClient(
            FakeRobotBehavior(joints_deg=home.joints_deg, gripper_motion_s=0.5)
        )
        client.current_tcp = home.tcp  # stand at home so a start is accepted
        return client
    from teleop.robot.fairino_client import FairinoRobotClient

    return FairinoRobotClient(config.teleop)


class RobotRuntime:
    def __init__(
        self,
        config: InferConfig,
        client: RobotClient,
        mailbox: TrajectoryMailbox,
        state_slot: RobotStateSlot,
    ) -> None:
        self.config = config
        self.teleop = config.teleop
        self.client = client
        self.mailbox = mailbox
        self.state_slot = state_slot
        self.shadow = config.mode == "shadow"
        self.period_ns = int(self.teleop.servo_period_s * 1e9)
        self.home_q = euler_deg_to_quat(*self.teleop.home.tcp[3:6])

        self.state = "IDLE"
        self.reason = "ready"
        self.servo_active = False
        self.measured_tcp: TcpPose = (0.0,) * 6  # type: ignore[assignment]
        self.measured_joints: tuple[float, ...] = (0.0,) * 6
        self.gripper_position = 0.0
        self.state_valid = False
        self.state_ns = 0
        self.last_tcp_poll_ns = 0
        self.gripper_closed = self.teleop.gripper.initially_closed
        self.gripper_done_ns = 0

        # Last commanded (or, in shadow mode, would-be commanded) target.
        self.cmd_position: Vector3 | None = None
        self.cmd_orientation: Quaternion | None = None
        self.last_cmd_ns = 0
        self.tick_desired: TcpPose | None = None
        self.tick_cmd: TcpPose | None = None

        self.mailbox_contents: MailboxContents | None = None
        self.episode = 0
        self.episode_start_ns = 0
        self.wait_since_ns = 0
        self.home: HomeTrajectory | None = None
        self.home_started_ns = 0
        self.home_arrived_ns: int | None = None
        self.gripper_target: bool | None = None
        self.gripper_deadline_ns = 0
        self.next_gripper_poll_ns = 0

        self.run_dir: Path | None = None
        self.ticks: list[tuple[Any, ...]] = []
        self.shutdown = False

    # ── lifecycle ────────────────────────────────────────────────────

    def connect(self) -> None:
        self.state = "STARTING_UP"
        self._publish(time.monotonic_ns())
        self.client.connect()
        if not self.shadow:
            # Reset errors, enable, activate the gripper. Shadow mode leaves
            # the controller exactly as it is.
            self.client.initialize()
        self._read_state(time.monotonic_ns(), force_rpc=True)
        open_position = self.teleop.gripper.open_position
        closed_position = self.teleop.gripper.closed_position
        if self.state_valid:
            self.gripper_closed = (
                normalize_gripper(self.gripper_position, open_position, closed_position) > 0.5
            )
        self.state = "IDLE"
        self.reason = "shadow_ready" if self.shadow else "ready"
        log.info(
            "robot ready mode=%s tcp=%s gripper_closed=%s",
            self.config.mode,
            tuple(round(v, 1) for v in self.measured_tcp),
            self.gripper_closed,
        )

    def close(self) -> None:
        if self.servo_active:
            self._servo_end()
        self._end_episode("shutdown")
        try:
            self.client.close()
        except Exception as exc:  # noqa: BLE001
            log.warning("robot close failed: %s", exc)
        self.state = "SHUTDOWN"
        self._publish(time.monotonic_ns())

    def run(self, conn: Connection) -> None:
        next_deadline = time.monotonic_ns()
        while not self.shutdown:
            while conn.poll():
                self.handle_command(conn.recv(), time.monotonic_ns())
            if self.shutdown:
                break
            self.tick(time.monotonic_ns())
            next_deadline += self.period_ns
            delay = next_deadline - time.monotonic_ns()
            if delay > 0:
                time.sleep(delay / 1e9)
            elif -delay > self.period_ns:
                next_deadline = time.monotonic_ns()

    # ── commands ─────────────────────────────────────────────────────

    def handle_command(self, command: tuple[Any, ...], now_ns: int) -> None:
        name = command[0]
        log.info("robot command %s state=%s", name, self.state)
        if name == "SHUTDOWN":
            self.shutdown = True
        elif name == "STOP":
            if self.state in EPISODE_STATES or self.state == "HOMING":
                self._stop(now_ns, "stopped_by_operator")
        elif name == "RESET":
            if self.state == "FAULT":
                try:
                    if not self.shadow:
                        self.client.reset_fault()
                    self._set(now_ns, "IDLE", "fault_reset")
                except Exception as exc:  # noqa: BLE001
                    self._fault(now_ns, "fault_reset_failed", exc)
        elif name == "HOME":
            self._begin_home(now_ns)
        elif name == "START":
            self._begin_episode(now_ns, Path(command[1]) if command[1] else None)

    def _begin_episode(self, now_ns: int, run_dir: Path | None) -> None:
        if self.state != "IDLE":
            self._refuse(now_ns, f"start_refused_{self.state.lower()}")
            return
        if not self.state_valid:
            self._refuse(now_ns, "start_refused_no_robot_state")
            return
        if not self._in_workspace(self.measured_tcp):
            self._refuse(now_ns, "start_refused_outside_workspace")
            return
        self.episode += 1
        self.episode_start_ns = now_ns
        self.wait_since_ns = now_ns
        self.run_dir = run_dir
        self.ticks = []
        self._set(now_ns, "STARTING", "waiting_for_trajectory")

    # ── tick ─────────────────────────────────────────────────────────

    def tick(self, now_ns: int) -> None:
        self.tick_desired = None
        self.tick_cmd = None
        self._read_state(now_ns)
        try:
            if self.state == "HOMING":
                self._tick_home(now_ns)
            elif self.state == "STARTING":
                self._tick_starting(now_ns)
            elif self.state == "RUNNING":
                self._tick_running(now_ns)
            elif self.state == "GRIPPER":
                self._tick_gripper(now_ns)
        except Exception as exc:  # noqa: BLE001
            self._fault(time.monotonic_ns(), f"{self.state.lower()}_failed", exc)
        if self.state in EPISODE_STATES:
            self._log_tick()
        self._publish(now_ns)

    def _read_state(self, now_ns: int, *, force_rpc: bool = False) -> None:
        snapshot: RobotStateSnapshot | None = None
        try:
            snapshot = self.client.get_state_snapshot()
        except Exception as exc:  # noqa: BLE001
            log.debug("state snapshot failed: %s", exc)
        if snapshot is not None:
            self.measured_tcp = tuple(snapshot.tcp)  # type: ignore[assignment]
            self.measured_joints = tuple(snapshot.joint_pos_deg)
            self.gripper_position = float(snapshot.gripper_position)
            self.state_ns = time.monotonic_ns()
            self.state_valid = True
            return
        if force_rpc or now_ns - self.last_tcp_poll_ns >= TCP_POLL_PERIOD_NS:
            self.last_tcp_poll_ns = now_ns
            try:
                self.measured_tcp = self.client.get_current_tcp()
                self.measured_joints = self.client.get_current_joints()
                self.state_ns = time.monotonic_ns()
                self.state_valid = True
            except Exception as exc:  # noqa: BLE001
                log.debug("GetActualTCPPose failed: %s", exc)

    def _tick_starting(self, now_ns: int) -> None:
        contents = self.mailbox.read()
        # Only a trajectory inferred after the start (or after the gripper
        # finished) counts: earlier ones saw a different scene.
        fresh = contents.obs_ns >= self.wait_since_ns and len(contents.trajectory) > 0
        if not fresh:
            if now_ns - self.wait_since_ns > self.config.execution.start_timeout_s * 1e9:
                self._stop(now_ns, "start_timeout")
            return
        trajectory = contents.trajectory
        if now_ns > trajectory.end_ns + self.config.execution.trajectory_grace_s * 1e9:
            return  # stale already; wait for the next one
        position, orientation = trajectory.sample(now_ns + self._lookahead_ns)
        tcp = self.measured_tcp
        offset_mm = float(np.linalg.norm(np.subtract(position, tcp[:3])))
        offset_deg = quat_angle_deg(euler_deg_to_quat(*tcp[3:6]), orientation)
        safety = self.config.safety
        if offset_mm > safety.max_start_offset_mm or offset_deg > safety.max_start_offset_deg:
            log.warning(
                "start refused: first target is %.1f mm / %.1f deg from the robot",
                offset_mm,
                offset_deg,
            )
            self._stop(now_ns, f"start_offset_{offset_mm:.0f}mm_{offset_deg:.0f}deg")
            return
        if not self.shadow:
            self.client.servo_start()
            self.servo_active = True
        self.cmd_position = (tcp[0], tcp[1], tcp[2])
        self.cmd_orientation = euler_deg_to_quat(*tcp[3:6])
        self.last_cmd_ns = now_ns
        self.mailbox_contents = contents
        self._set(now_ns, "RUNNING", "running")

    def _tick_running(self, now_ns: int) -> None:
        if now_ns - self.episode_start_ns > self.config.safety.max_episode_s * 1e9:
            self._stop(now_ns, "max_episode_time")
            return
        if self.mailbox_contents is None or self.mailbox.seq() != self.mailbox_contents.seq:
            self.mailbox_contents = self.mailbox.read()
        contents = self.mailbox_contents
        if (
            contents.gripper_change_ns >= 0
            and contents.gripper_closed != self.gripper_closed
            and now_ns + self._lookahead_ns >= contents.gripper_change_ns
        ):
            self._begin_gripper(now_ns, contents.gripper_closed)
            return
        trajectory = contents.trajectory
        grace_ns = self.config.execution.trajectory_grace_s * 1e9
        if len(trajectory) == 0 or now_ns > trajectory.end_ns + grace_ns:
            self._stop(now_ns, "trajectory_exhausted")
            return
        position, orientation = trajectory.sample(now_ns + self._lookahead_ns)
        self.tick_desired = _tcp(position, orientation)
        orientation = clamp_orientation_deviation(
            self.home_q, orientation, self.config.safety.max_orientation_deviation_deg
        )
        assert self.cmd_position is not None and self.cmd_orientation is not None
        elapsed_s = max(0.0, (now_ns - self.last_cmd_ns) / 1e9)
        target, limited_q = limit_home_step(
            self.cmd_position,
            self.cmd_orientation,
            position,
            orientation,
            step_limit_mm=min(self.teleop.max_step_mm, self.config.max_velocity_mm_s * elapsed_s),
            angular_step_limit_deg=min(
                self.teleop.orientation.max_step_deg,
                self.config.max_angular_velocity_deg_s * elapsed_s,
            ),
            config=self.teleop,
        )
        if not self.shadow:
            self.client.servo_cart(target)
        self.cmd_position = (target[0], target[1], target[2])
        self.cmd_orientation = limited_q
        self.last_cmd_ns = now_ns
        self.tick_cmd = target

    # ── gripper ──────────────────────────────────────────────────────

    def _begin_gripper(self, now_ns: int, closed: bool) -> None:
        self._servo_end()
        self.gripper_target = closed
        self._set(now_ns, "GRIPPER", "gripper_closing" if closed else "gripper_opening")
        self._publish(now_ns)
        if self.shadow:
            self._finish_gripper(time.monotonic_ns())
            return
        gripper = self.teleop.gripper
        # Non-blocking (SDK block=1); _tick_gripper polls GetGripperMotionDone.
        # Restarting the servo before the gripper really finishes blocks
        # ServoMoveStart for the rest of the motion (~1.1 s).
        self.client.move_gripper(gripper.closed_position if closed else gripper.open_position)
        done_ns = time.monotonic_ns()
        self.gripper_deadline_ns = done_ns + int(gripper.action_timeout_s * 1e9)
        self.next_gripper_poll_ns = done_ns

    def _tick_gripper(self, now_ns: int) -> None:
        if now_ns >= self.gripper_deadline_ns:
            self._fault(now_ns, "gripper_timeout", RuntimeError("gripper did not finish"))
            return
        if now_ns < self.next_gripper_poll_ns:
            return
        self.next_gripper_poll_ns = now_ns + int(self.teleop.gripper.poll_period_s * 1e9)
        motion = self.client.get_gripper_motion_state()
        if motion.fault != 0:
            self._fault(now_ns, "gripper_fault", RuntimeError(f"gripper fault={motion.fault}"))
        elif motion.done:
            self._finish_gripper(now_ns)

    def _finish_gripper(self, now_ns: int) -> None:
        assert self.gripper_target is not None
        self.gripper_closed = self.gripper_target
        self.gripper_target = None
        self.gripper_done_ns = now_ns
        self.wait_since_ns = now_ns
        self._set(now_ns, "STARTING", "gripper_done_waiting_for_trajectory")

    # ── home return (same profile and limits as the fr5_teleop worker) ──

    def _begin_home(self, now_ns: int) -> None:
        home = self.teleop.home
        if self.shadow:
            self._refuse(now_ns, "home_refused_shadow_mode")
            return
        if not home.enabled:
            self._refuse(now_ns, "home_disabled")
            return
        if self.state != "IDLE":
            self._refuse(now_ns, f"home_refused_{self.state.lower()}")
            return
        try:
            joints = self.client.get_current_joints()
            tcp = self.client.get_current_tcp()
        except Exception as exc:  # noqa: BLE001
            self._fault(now_ns, "home_state_read_failed", exc)
            return
        if home.joints_deg is not None:
            delta = max(abs(a - b) for a, b in zip(joints, home.joints_deg))
            if delta > home.max_start_joint_delta_deg:
                self._refuse(now_ns, "home_refused_joint_delta")
                return
        if not self._in_workspace(tcp):
            self._refuse(now_ns, "home_refused_outside_workspace")
            return
        self.home = HomeTrajectory.plan(
            tcp,
            home.tcp,
            max_velocity_mm_s=home.max_velocity_mm_s,
            max_angular_velocity_deg_s=home.max_angular_velocity_deg_s,
        )
        self.cmd_position = (tcp[0], tcp[1], tcp[2])
        self.cmd_orientation = euler_deg_to_quat(*tcp[3:6])
        try:
            self.client.servo_start()
        except Exception as exc:  # noqa: BLE001
            self._fault(now_ns, "home_start_failed", exc)
            return
        self.servo_active = True
        self.home_started_ns = time.monotonic_ns()
        self.home_arrived_ns = None
        self.last_cmd_ns = self.home_started_ns
        log.info("home return started duration_s=%.2f", self.home.duration_s)
        self._set(now_ns, "HOMING", "home_started")

    def _tick_home(self, now_ns: int) -> None:
        trajectory = self.home
        home = self.teleop.home
        assert trajectory is not None
        assert self.cmd_position is not None and self.cmd_orientation is not None
        elapsed_s = max(0.0, (now_ns - self.home_started_ns) / 1e9)
        since_s = max(0.0, (now_ns - self.last_cmd_ns) / 1e9)
        position, orientation = trajectory.sample(elapsed_s)
        target, limited_q = limit_home_step(
            self.cmd_position,
            self.cmd_orientation,
            position,
            orientation,
            step_limit_mm=min(
                self.teleop.max_step_mm, HOME_LIMIT_MARGIN * home.max_velocity_mm_s * since_s
            ),
            angular_step_limit_deg=min(
                self.teleop.orientation.max_step_deg,
                HOME_LIMIT_MARGIN * home.max_angular_velocity_deg_s * since_s,
            ),
            config=self.teleop,
        )
        self.client.servo_cart(target)
        self.cmd_position = (target[0], target[1], target[2])
        self.cmd_orientation = limited_q
        self.last_cmd_ns = now_ns
        self.tick_cmd = target
        if self.home_arrived_ns is None:
            remaining_mm = max(
                abs(a - b) for a, b in zip(self.cmd_position, trajectory.goal_position)
            )
            remaining_deg = quat_angle_deg(limited_q, trajectory.goal_orientation)
            if (
                elapsed_s >= trajectory.duration_s
                and remaining_mm <= HOME_ARRIVE_MM
                and remaining_deg <= HOME_ARRIVE_DEG
            ):
                self.home_arrived_ns = now_ns
            elif elapsed_s > trajectory.duration_s + HOME_SETTLE_TIMEOUT_S:
                self._stop(now_ns, "home_timeout")
        elif now_ns - self.home_arrived_ns >= home.hold_s * 1e9:
            self._stop(now_ns, "home_reached")

    # ── transitions ──────────────────────────────────────────────────

    def _servo_end(self) -> None:
        if self.servo_active:
            self.servo_active = False
            self.client.servo_end()
        self.cmd_position = None
        self.cmd_orientation = None

    def _stop(self, now_ns: int, reason: str) -> None:
        try:
            self._servo_end()
        except Exception as exc:  # noqa: BLE001
            self._fault(now_ns, f"servo_end_failed_after_{reason}", exc)
            return
        self.home = None
        self._end_episode(reason)
        self._set(now_ns, "IDLE", reason)

    def _fault(self, now_ns: int, reason: str, error: Exception) -> None:
        log.error("robot fault reason=%s error=%s", reason, error, exc_info=error)
        if self.servo_active:
            try:
                self.servo_active = False
                self.client.servo_end()
            except Exception as exc:  # noqa: BLE001
                log.error("servo end after fault failed: %s", exc)
        self.cmd_position = None
        self.cmd_orientation = None
        self.home = None
        self.gripper_target = None
        self._end_episode(reason)
        self._set(now_ns, "FAULT", f"{reason}: {error}")

    def _refuse(self, now_ns: int, reason: str) -> None:
        log.warning("command refused: %s", reason)
        self.reason = reason

    def _set(self, now_ns: int, state: str, reason: str) -> None:
        if state != self.state or reason != self.reason:
            log.info("robot %s -> %s (%s)", self.state, state, reason)
        self.state = state
        self.reason = reason

    def _end_episode(self, reason: str) -> None:
        if self.state not in EPISODE_STATES:
            return
        if self.run_dir is not None and self.ticks:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            np.save(self.run_dir / "ticks.npy", np.array(self.ticks, dtype=TICK_DTYPE))
            (self.run_dir / "robot_end_reason.txt").write_text(reason + "\n", encoding="utf-8")
        self.ticks = []
        self.run_dir = None

    # ── helpers ──────────────────────────────────────────────────────

    @property
    def _lookahead_ns(self) -> int:
        return int(self.config.execution.lookahead_s * 1e9)

    def _in_workspace(self, tcp: TcpPose) -> bool:
        workspace = self.teleop.workspace
        return (
            workspace.x[0] <= tcp[0] <= workspace.x[1]
            and workspace.y[0] <= tcp[1] <= workspace.y[1]
            and workspace.z[0] <= tcp[2] <= workspace.z[1]
        )

    def _log_tick(self) -> None:
        nan6 = (float("nan"),) * 6
        contents = self.mailbox_contents
        self.ticks.append(
            (
                self.state_ns,
                STATE_CODES.index(self.state),
                self.measured_tcp,
                self.gripper_position,
                int(self.gripper_closed),
                self.tick_desired or nan6,
                self.tick_cmd or nan6,
                int(self.tick_cmd is not None),
                contents.seq if contents is not None else -1,
            )
        )

    def _publish(self, now_ns: int) -> None:
        contents = self.mailbox_contents
        trajectory = contents.trajectory if contents is not None else None
        self.state_slot.write(
            RobotState(
                t_ns=self.state_ns,
                state=self.state,
                reason=self.reason,
                tcp=tuple(self.measured_tcp),
                joints_deg=tuple(self.measured_joints),
                gripper_position=self.gripper_position,
                gripper_closed=self.gripper_closed,
                gripper_done_ns=self.gripper_done_ns,
                cmd_tcp=self.tick_cmd or (0.0,) * 6,
                cmd_valid=self.tick_cmd is not None,
                episode=self.episode,
                traj_seq=contents.seq if contents is not None else 0,
                traj_end_ns=(
                    trajectory.end_ns if trajectory is not None and len(trajectory) else 0
                ),
                episode_start_ns=self.episode_start_ns,
                servo_active=self.servo_active,
                valid=self.state_valid,
            )
        )


def _tcp(position: Vector3, orientation: Quaternion) -> TcpPose:
    rx, ry, rz = quat_to_euler_deg(orientation)
    return (position[0], position[1], position[2], rx, ry, rz)


def run_robot_process(
    config: InferConfig,
    conn: Connection,
    mailbox: TrajectoryMailbox,
    state_slot: RobotStateSlot,
) -> None:
    logging.basicConfig(
        level=config.teleop.log_level,
        format="%(asctime)s %(levelname)s [robot] %(name)s: %(message)s",
    )
    runtime = RobotRuntime(config, make_client(config), mailbox, state_slot)
    try:
        runtime.connect()
    except Exception as exc:  # noqa: BLE001
        runtime._fault(time.monotonic_ns(), "connect_failed", exc)
        runtime._publish(time.monotonic_ns())
        # Stay alive so the UI shows the fault; only SHUTDOWN ends the process.
        while not runtime.shutdown:
            if conn.poll(0.1):
                command = conn.recv()
                runtime.shutdown = command[0] == "SHUTDOWN"
        return
    try:
        runtime.run(conn)
    finally:
        runtime.close()
