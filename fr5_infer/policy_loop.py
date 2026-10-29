"""Inference thread: observation -> chunk -> timed trajectory -> mailbox.

1. Trigger while the robot is STARTING (no trajectory for this phase yet) or
   RUNNING and the remaining trajectory is short (latency EMA + margin) or
   ``replan_steps`` chunk steps have passed since the last observation.
2. Observe: the robot state slot (its measurement time is t_obs) and the
   latest frame of each camera; stale inputs mean no inference.
3. Infer (with RTC, continuing the previous chunk), decode, stamp row k
   with t_obs + k/fps, drop past points.
4. Optionally ensemble with the earlier chunks (ACT), cross-fade with the
   previous trajectory, run the gripper hysteresis and, if it flips, cut the
   trajectory at that point and attach the switch.
5. Write the mailbox. No new inference until a written gripper switch has
   completed; the robot then waits for an observation taken after it.
   The robot stood still meanwhile, so neither RTC nor the ensemble carries
   the earlier chunks over a gripper action.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from fr5_infer.backends import Observation, PolicyBackend
from fr5_infer.config import InferConfig
from fr5_infer.conversion import ConversionInfo, decode_chunk, encode_state
from fr5_infer.rollout_log import EpisodeLog
from fr5_infer.shm import LatestFrameSlot, RobotState, RobotStateSlot, TrajectoryMailbox
from fr5_infer.trajectory import (
    GripperSchmitt,
    TemporalEnsemble,
    Trajectory,
    align_chunk,
    blend,
)


log = logging.getLogger(__name__)

EPISODE_STATES = {"STARTING", "RUNNING", "GRIPPER"}
LATENCY_EMA_ALPHA = 0.3
IDLE_SLEEP_S = 0.005


class PolicyLoop(threading.Thread):
    def __init__(
        self,
        config: InferConfig,
        conversion: ConversionInfo,
        backend: PolicyBackend,
        mailbox: TrajectoryMailbox,
        robot_slot: RobotStateSlot,
        frame_slots: dict[str, LatestFrameSlot],
        base_meta: dict[str, Any],
    ) -> None:
        super().__init__(name="policy", daemon=True)
        self.config = config
        self.conversion = conversion
        self.backend = backend
        self.mailbox = mailbox
        self.robot_slot = robot_slot
        self.frame_slots = frame_slots
        self.base_meta = base_meta
        self.stop_event = threading.Event()
        self._lock = threading.Lock()
        self._task = config.policy.task
        self._pending_run_dir: Path | None = None

        gripper = config.gripper
        self.schmitt = GripperSchmitt(
            gripper.close_threshold, gripper.open_threshold, gripper.min_dwell_s
        )
        self.ensemble = (
            TemporalEnsemble(config.execution.ensemble_decay) if config.execution.ensemble else None
        )
        self.episode = 0
        self.episode_log: EpisodeLog | None = None
        self.trajectory: Trajectory | None = None
        self.last_obs_ns = 0
        # (target_closed, written_ns) of a gripper switch sent to the robot.
        self.pending_switch: tuple[bool, int] | None = None
        self.latency_ema_s: float | None = None
        self.status: dict[str, Any] = {"state": "idle", "inferences": 0}

    # ── control from the web thread ──────────────────────────────────

    @property
    def task(self) -> str:
        with self._lock:
            return self._task

    def set_task(self, task: str) -> None:
        with self._lock:
            self._task = task

    def arm(self, run_dir: Path | None) -> None:
        """The next robot episode logs into run_dir."""
        with self._lock:
            self._pending_run_dir = run_dir

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            status = dict(self.status)
        status["task"] = self.task
        status["latency_ema_s"] = self.latency_ema_s
        return status

    def _set_status(self, **values: Any) -> None:
        with self._lock:
            self.status.update(values)

    # ── loop ─────────────────────────────────────────────────────────

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                if not self.step(time.monotonic_ns()):
                    time.sleep(IDLE_SLEEP_S)
            except Exception as exc:  # noqa: BLE001
                log.exception("policy step failed")
                self._set_status(state="error", error=str(exc))
                time.sleep(0.5)
        self._end_episode("shutdown")

    def step(self, now_ns: int) -> bool:
        """One iteration; returns True if an inference ran."""
        robot = self.robot_slot.read()
        if self.episode_log is not None or self.episode:
            if robot.episode != self.episode or robot.state not in EPISODE_STATES:
                self._end_episode(robot.reason)
        if robot.state in EPISODE_STATES and robot.episode != self.episode:
            self._begin_episode(robot, now_ns)
        if robot.state not in EPISODE_STATES or robot.episode != self.episode:
            self._set_status(state="idle")
            return False

        if self.pending_switch is not None:
            target, written_ns = self.pending_switch
            if robot.gripper_closed == target and robot.gripper_done_ns >= written_ns:
                self.pending_switch = None
                self.schmitt.reset(target, robot.gripper_done_ns)
                # The cut trajectory ended at the switch; start over after it.
                self._forget_chunks()
            else:
                self._set_status(state="waiting_for_gripper")
                return False
        if not self._should_infer(robot, now_ns):
            return False
        observation = self._observe(robot, now_ns)
        if observation is None:
            return False
        self._infer(robot, observation)
        return True

    def _should_infer(self, robot: RobotState, now_ns: int) -> bool:
        if robot.state not in {"STARTING", "RUNNING"}:
            return False
        # A start or a finished gripper action needs a fresh observation.
        phase_ns = max(robot.episode_start_ns, robot.gripper_done_ns)
        if self.last_obs_ns < phase_ns or self.trajectory is None or len(self.trajectory) == 0:
            return True
        execution = self.config.execution
        remaining_s = (self.trajectory.end_ns - now_ns) / 1e9
        margin_s = (self.latency_ema_s or 0.0) + execution.refill_margin_s
        executed_steps = (now_ns - self.last_obs_ns) * self.conversion.fps / 1e9
        return remaining_s < margin_s or executed_steps >= execution.replan_steps

    def _observe(self, robot: RobotState, now_ns: int) -> Observation | None:
        safety = self.config.safety
        if not robot.valid or (now_ns - robot.t_ns) / 1e9 > safety.max_state_age_s:
            self._set_status(state="waiting_for_robot_state")
            return None
        images = {}
        for name in self.conversion.cameras:
            frame = self.frame_slots[name].read()
            if frame is None or (now_ns - frame.capture_ns) / 1e9 > safety.max_frame_age_s:
                self._set_status(state=f"waiting_for_camera_{name}")
                return None
            images[name] = frame.rgb
        gripper = self.config.teleop.gripper
        state = encode_state(
            robot.tcp,  # type: ignore[arg-type]
            robot.gripper_position,
            gripper.open_position,
            gripper.closed_position,
        )
        return Observation(
            state=state,
            images=images,
            task=self.task,
            t_obs_ns=robot.t_ns,
            episode_time_s=max(0.0, (robot.t_ns - robot.episode_start_ns) / 1e9),
            joint_pos=np.radians(robot.joints_deg).astype(np.float32),
        )

    def _infer(self, robot: RobotState, observation: Observation) -> None:
        self._set_status(state="inferring")
        execution = self.config.execution
        # Chunk steps that will be in the past once this inference is done.
        delay_s = (self.latency_ema_s or 0.0) + execution.min_lead_s
        started_ns = time.monotonic_ns()
        chunk = np.asarray(self.backend.predict(observation, delay_s), dtype=np.float32)
        done_ns = time.monotonic_ns()
        latency_s = (done_ns - started_ns) / 1e9
        self.latency_ema_s = (
            latency_s
            if self.latency_ema_s is None
            else LATENCY_EMA_ALPHA * latency_s + (1 - LATENCY_EMA_ALPHA) * self.latency_ema_s
        )
        self.last_obs_ns = observation.t_obs_ns

        decoded = decode_chunk(chunk)
        aligned = align_chunk(decoded, observation.t_obs_ns, self.conversion.fps)
        keep = aligned.t_ns >= done_ns + int(execution.min_lead_s * 1e9)
        trajectory = aligned.select(keep)
        gripper_raw = decoded.gripper[keep]
        dropped = int((~keep).sum())
        if len(trajectory) == 0:
            log.warning("inference took %.3f s: the whole chunk is already in the past", latency_s)
            self._log_inference(observation, chunk, latency_s, done_ns, dropped, 0, -1, -1)
            return

        if self.ensemble is not None:
            trajectory, gripper_raw = self.ensemble.add(trajectory, gripper_raw)
        if robot.state == "RUNNING":
            trajectory = blend(self.trajectory, trajectory, execution.blend_s)
        switch = self.schmitt.first_switch(trajectory.t_ns, gripper_raw)
        gripper_closed = self.schmitt.closed
        change_ns = -1
        if switch is not None:
            trajectory = trajectory.select(slice(0, switch + 1))
            gripper_closed = not self.schmitt.closed
            change_ns = int(trajectory.t_ns[switch])
        trajectory = trajectory.select(slice(0, self.mailbox.max_points))
        seq = self.mailbox.write(trajectory, observation.t_obs_ns, gripper_closed, change_ns)
        if switch is not None:
            self.pending_switch = (gripper_closed, done_ns)
        self.trajectory = trajectory
        self._log_inference(
            observation, chunk, latency_s, done_ns, dropped, len(trajectory), change_ns, seq
        )
        with self._lock:
            self.status.update(
                state="running",
                inferences=self.status.get("inferences", 0) + 1,
                last_latency_s=round(latency_s, 4),
                dropped_points=dropped,
                sent_points=len(trajectory),
                gripper_switch=None if switch is None else ("close" if gripper_closed else "open"),
                gripper_raw_first=round(float(gripper_raw[0]), 3),
            )

    def _forget_chunks(self) -> None:
        self.trajectory = None
        self.backend.break_continuation()
        if self.ensemble is not None:
            self.ensemble.reset()

    # ── episode bookkeeping ──────────────────────────────────────────

    def _begin_episode(self, robot: RobotState, now_ns: int) -> None:
        self.episode = robot.episode
        self._forget_chunks()
        self.last_obs_ns = 0
        self.pending_switch = None
        self.schmitt.reset(robot.gripper_closed, 0)
        self.backend.reset()
        with self._lock:
            run_dir, self._pending_run_dir = self._pending_run_dir, None
            self.status = {"state": "starting", "inferences": 0, "episode": robot.episode}
        if run_dir is not None:
            meta = dict(self.base_meta, episode=robot.episode, task=self.task)
            self.episode_log = EpisodeLog(run_dir, meta, self.config.log.save_images)
            self._set_status(run_dir=str(run_dir))
        log.info("episode %d started run_dir=%s", robot.episode, run_dir)

    def _end_episode(self, reason: str) -> None:
        episode_log, self.episode_log = self.episode_log, None
        if episode_log is not None:
            try:
                episode_log.close(reason)
                log.info("episode %d ended: %s (%s)", self.episode, reason, episode_log.run_dir)
            except Exception:  # noqa: BLE001
                log.exception("saving the episode log in %s failed", episode_log.run_dir)
        if self.episode:
            self._set_status(last_end_reason=reason)
        self.episode = 0
        self._forget_chunks()

    def _log_inference(
        self,
        observation: Observation,
        chunk: np.ndarray,
        latency_s: float,
        done_ns: int,
        dropped: int,
        sent: int,
        change_ns: int,
        seq: int,
    ) -> None:
        if self.episode_log is None:
            return
        rtc = getattr(self.backend, "last_rtc", None) or {}
        self.episode_log.add_inference(
            {
                "t_obs_ns": observation.t_obs_ns,
                "t_done_ns": done_ns,
                "latency_s": latency_s,
                "state": observation.state,
                "chunk": chunk,
                "dropped_points": dropped,
                "sent_points": sent,
                "gripper_change_ns": change_ns,
                "mailbox_seq": seq,
                # RTC prefix length and frozen steps (0 = chunk not guided).
                "rtc_prefix_steps": rtc.get("prefix_steps", 0),
                "rtc_delay_steps": rtc.get("delay_steps", 0),
            },
            observation.images,
        )
