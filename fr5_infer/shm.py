"""Fixed-size, overwrite-only slots shared between processes.

Same pattern as ``teleop.ipc.LatestPoseMailbox``: one lock, raw ctypes
arrays, readers copy under the lock. The robot loop only ever reads the
trajectory mailbox, so inference never blocks it for more than a copy.
"""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing.context import BaseContext

import numpy as np

from fr5_infer.trajectory import Trajectory


@dataclass(frozen=True)
class MailboxContents:
    seq: int
    # Robot-state time of the observation the trajectory was inferred from.
    obs_ns: int
    trajectory: Trajectory
    gripper_closed: bool
    # When to switch the gripper to gripper_closed; -1 means no switch.
    gripper_change_ns: int


class TrajectoryMailbox:
    def __init__(self, context: BaseContext, max_points: int) -> None:
        self.max_points = max_points
        self._lock = context.Lock()
        self._seq = context.Value("q", 0, lock=False)
        self._obs_ns = context.Value("q", 0, lock=False)
        self._count = context.Value("i", 0, lock=False)
        self._gripper = context.Value("b", 0, lock=False)
        self._change_ns = context.Value("q", -1, lock=False)
        self._t_ns = context.Array("q", max_points, lock=False)
        self._pose = context.Array("d", max_points * 7, lock=False)

    def write(
        self,
        trajectory: Trajectory,
        obs_ns: int,
        gripper_closed: bool,
        gripper_change_ns: int,
    ) -> int:
        count = min(len(trajectory), self.max_points)
        pose = np.concatenate(
            [trajectory.position_mm[:count], trajectory.quat_xyzw[:count]], axis=1
        )
        with self._lock:
            np.frombuffer(self._t_ns, dtype=np.int64)[:count] = trajectory.t_ns[:count]
            np.frombuffer(self._pose, dtype=np.float64)[: count * 7] = pose.ravel()
            self._count.value = count
            self._obs_ns.value = obs_ns
            self._gripper.value = int(gripper_closed)
            self._change_ns.value = gripper_change_ns
            self._seq.value += 1
            return int(self._seq.value)

    def clear(self) -> None:
        with self._lock:
            self._count.value = 0
            self._change_ns.value = -1
            self._seq.value += 1

    def seq(self) -> int:
        with self._lock:
            return int(self._seq.value)

    def read(self) -> MailboxContents:
        with self._lock:
            count = int(self._count.value)
            t_ns = np.frombuffer(self._t_ns, dtype=np.int64)[:count].copy()
            pose = (
                np.frombuffer(self._pose, dtype=np.float64)[: count * 7]
                .copy()
                .reshape(count, 7)
            )
            return MailboxContents(
                seq=int(self._seq.value),
                obs_ns=int(self._obs_ns.value),
                trajectory=Trajectory(t_ns, pose[:, :3], pose[:, 3:]),
                gripper_closed=bool(self._gripper.value),
                gripper_change_ns=int(self._change_ns.value),
            )


# ── robot state ─────────────────────────────────────────────────────

ROBOT_STATES = (
    "STARTING_UP",
    "IDLE",
    "HOMING",
    "STARTING",
    "RUNNING",
    "GRIPPER",
    "FAULT",
    "SHUTDOWN",
)
REASON_BYTES = 96
# float slots: tcp 6, joints 6, cmd 6, gripper_position
_FLOATS = 19
# int slots: t_ns, state, gripper_closed, gripper_done_ns, cmd_valid, episode,
# traj_seq, traj_end_ns, episode_start_ns, servo_active, state_valid
_INTS = 11


@dataclass(frozen=True)
class RobotState:
    t_ns: int
    state: str
    reason: str
    tcp: tuple[float, ...]
    joints_deg: tuple[float, ...]
    gripper_position: float
    # Commanded gripper state and when its last action completed.
    gripper_closed: bool
    gripper_done_ns: int
    cmd_tcp: tuple[float, ...]
    cmd_valid: bool
    episode: int
    traj_seq: int
    traj_end_ns: int
    episode_start_ns: int
    servo_active: bool
    # False until the first robot state was read.
    valid: bool


class RobotStateSlot:
    def __init__(self, context: BaseContext) -> None:
        self._lock = context.Lock()
        self._floats = context.Array("d", _FLOATS, lock=False)
        self._ints = context.Array("q", _INTS, lock=False)
        self._reason = context.Array("c", REASON_BYTES, lock=False)

    def write(self, state: RobotState) -> None:
        floats = (*state.tcp, *state.joints_deg, *state.cmd_tcp, state.gripper_position)
        ints = (
            state.t_ns,
            ROBOT_STATES.index(state.state),
            int(state.gripper_closed),
            state.gripper_done_ns,
            int(state.cmd_valid),
            state.episode,
            state.traj_seq,
            state.traj_end_ns,
            state.episode_start_ns,
            int(state.servo_active),
            int(state.valid),
        )
        reason = state.reason.encode("utf-8")[: REASON_BYTES - 1]
        with self._lock:
            for i, value in enumerate(floats):
                self._floats[i] = value
            for i, value in enumerate(ints):
                self._ints[i] = value
            self._reason.value = reason

    def read(self) -> RobotState:
        with self._lock:
            f = list(self._floats)
            i = list(self._ints)
            reason = self._reason.value.decode("utf-8", errors="replace")
        return RobotState(
            t_ns=i[0],
            state=ROBOT_STATES[i[1]],
            reason=reason,
            tcp=tuple(f[0:6]),
            joints_deg=tuple(f[6:12]),
            gripper_position=f[18],
            gripper_closed=bool(i[2]),
            gripper_done_ns=i[3],
            cmd_tcp=tuple(f[12:18]),
            cmd_valid=bool(i[4]),
            episode=i[5],
            traj_seq=i[6],
            traj_end_ns=i[7],
            episode_start_ns=i[8],
            servo_active=bool(i[9]),
            valid=bool(i[10]),
        )


# ── camera frames ───────────────────────────────────────────────────

ERROR_BYTES = 160


@dataclass(frozen=True)
class Frame:
    seq: int
    capture_ns: int
    rgb: np.ndarray  # (H, W, 3) uint8


class LatestFrameSlot:
    """The newest RGB frame of one camera (~0.9 MB at 640x480)."""

    def __init__(self, context: BaseContext, name: str, height: int, width: int) -> None:
        self.name = name
        self.shape = (height, width, 3)
        self._lock = context.Lock()
        self._seq = context.Value("q", 0, lock=False)
        self._capture_ns = context.Value("q", 0, lock=False)
        self._pixels = context.Array("B", height * width * 3, lock=False)
        self._error = context.Array("c", ERROR_BYTES, lock=False)

    def write(self, rgb: np.ndarray, capture_ns: int) -> None:
        if rgb.shape != self.shape:
            raise ValueError(f"camera {self.name}: frame {rgb.shape} != {self.shape}")
        with self._lock:
            np.frombuffer(self._pixels, dtype=np.uint8)[:] = rgb.reshape(-1)
            self._capture_ns.value = capture_ns
            self._seq.value += 1
            self._error.value = b""

    def set_error(self, message: str) -> None:
        with self._lock:
            self._error.value = message.encode("utf-8")[: ERROR_BYTES - 1]

    def error(self) -> str:
        with self._lock:
            return self._error.value.decode("utf-8", errors="replace")

    def read(self) -> Frame | None:
        with self._lock:
            if self._seq.value == 0:
                return None
            rgb = np.frombuffer(self._pixels, dtype=np.uint8).copy().reshape(self.shape)
            return Frame(int(self._seq.value), int(self._capture_ns.value), rgb)
