"""Timed TCP trajectories: chunk alignment, blending, sampling, gripper.

All times are ``time.monotonic_ns()``, shared by every process on the host,
so a chunk is aligned by stamping row k with ``t_obs + k / fps`` and dropping
what is already in the past; inference latency is absorbed there.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from teleop.control_math import Quaternion, Vector3, quat_slerp

from fr5_infer.conversion import DecodedChunk


@dataclass(frozen=True)
class Trajectory:
    t_ns: np.ndarray  # (N,) int64, increasing
    position_mm: np.ndarray  # (N, 3)
    quat_xyzw: np.ndarray  # (N, 4)

    @classmethod
    def empty(cls) -> Trajectory:
        return cls(
            np.zeros(0, dtype=np.int64), np.zeros((0, 3)), np.zeros((0, 4))
        )

    def __len__(self) -> int:
        return len(self.t_ns)

    @property
    def start_ns(self) -> int:
        return int(self.t_ns[0])

    @property
    def end_ns(self) -> int:
        return int(self.t_ns[-1])

    def select(self, mask_or_slice: np.ndarray | slice) -> Trajectory:
        return Trajectory(
            self.t_ns[mask_or_slice],
            self.position_mm[mask_or_slice],
            self.quat_xyzw[mask_or_slice],
        )

    def from_time(self, t_ns: int) -> Trajectory:
        return self.select(self.t_ns >= t_ns)

    def sample(self, t_ns: int) -> tuple[Vector3, Quaternion]:
        """Linear position and slerp orientation, held at both ends."""
        if len(self) == 0:
            raise ValueError("empty trajectory")
        if t_ns <= self.t_ns[0]:
            return _point(self, 0)
        if t_ns >= self.t_ns[-1]:
            return _point(self, len(self) - 1)
        right = int(np.searchsorted(self.t_ns, t_ns, side="right"))
        left = right - 1
        span = float(self.t_ns[right] - self.t_ns[left])
        s = float(t_ns - self.t_ns[left]) / span
        a = self.position_mm[left]
        b = self.position_mm[right]
        position = tuple(float(v) for v in a + s * (b - a))
        orientation = quat_slerp(
            tuple(float(v) for v in self.quat_xyzw[left]),
            tuple(float(v) for v in self.quat_xyzw[right]),
            s,
        )
        return position, orientation  # type: ignore[return-value]


def _point(trajectory: Trajectory, index: int) -> tuple[Vector3, Quaternion]:
    position = tuple(float(v) for v in trajectory.position_mm[index])
    orientation = tuple(float(v) for v in trajectory.quat_xyzw[index])
    return position, orientation  # type: ignore[return-value]


def align_chunk(decoded: DecodedChunk, t_obs_ns: int, fps: int) -> Trajectory:
    """Row k is the command for t_obs + k/fps (action[t] pairs with state[t])."""
    count = len(decoded.gripper)
    t_ns = t_obs_ns + np.round(np.arange(count) * 1e9 / fps).astype(np.int64)
    return Trajectory(t_ns, decoded.position_mm.copy(), decoded.quat_xyzw.copy())


def blend(old: Trajectory | None, new: Trajectory, blend_s: float) -> Trajectory:
    """Cross-fade the first ``blend_s`` of ``new`` from ``old`` (cosine weight).

    The weight on ``new`` rises 0 -> 1, so the first blended point equals the
    old trajectory and the command stays continuous across a replan.
    """
    if old is None or len(old) == 0 or len(new) == 0 or blend_s <= 0:
        return new
    blend_ns = blend_s * 1e9
    position = new.position_mm.copy()
    quat = new.quat_xyzw.copy()
    for i, t in enumerate(new.t_ns):
        elapsed = float(t - new.t_ns[0])
        if elapsed >= blend_ns or t > old.end_ns:
            break
        weight = 0.5 - 0.5 * math.cos(math.pi * elapsed / blend_ns)
        old_position, old_quat = old.sample(int(t))
        position[i] = (1.0 - weight) * np.asarray(old_position) + weight * position[i]
        quat[i] = quat_slerp(old_quat, tuple(float(v) for v in quat[i]), weight)
    return Trajectory(new.t_ns.copy(), position, quat)


@dataclass
class GripperSchmitt:
    """Hysteresis on the policy's gripper output plus a minimum dwell time."""

    close_threshold: float
    open_threshold: float
    min_dwell_s: float
    closed: bool = False
    last_change_ns: int = 0

    def reset(self, closed: bool, now_ns: int) -> None:
        self.closed = closed
        self.last_change_ns = now_ns

    def first_switch(self, t_ns: np.ndarray, raw: np.ndarray) -> int | None:
        """Index of the first point that flips the current state, if any."""
        dwell_ns = self.min_dwell_s * 1e9
        for i, (t, value) in enumerate(zip(t_ns, raw)):
            if t - self.last_change_ns < dwell_ns:
                continue
            if not self.closed and value > self.close_threshold:
                return i
            if self.closed and value < self.open_threshold:
                return i
        return None
