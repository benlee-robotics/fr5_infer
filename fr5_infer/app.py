"""Wires the processes together: cameras, robot loop, policy thread."""

from __future__ import annotations

import dataclasses
import io
import logging
import multiprocessing as mp
import threading
import time
from typing import Any

import numpy as np

from fr5_infer import __version__
from fr5_infer.backends import PolicyBackend
from fr5_infer.camera_process import run_camera_process
from fr5_infer.config import InferConfig
from fr5_infer.conversion import (
    check_cameras,
    check_checkpoint,
    encode_state,
    load_conversion,
    normalize_gripper,
)
from fr5_infer.policy_loop import PolicyLoop
from fr5_infer.robot_process import run_robot_process
from fr5_infer.rollout_log import new_run_dir
from fr5_infer.shm import LatestFrameSlot, RobotStateSlot, TrajectoryMailbox


log = logging.getLogger(__name__)

PREVIEW_SIZE = (320, 240)
PROCESS_JOIN_S = 5.0


def make_backend(config: InferConfig, cameras: tuple[str, ...], fps: int) -> PolicyBackend:
    policy = config.policy
    if policy.backend == "replay":
        from fr5_infer.backends.replay import ReplayBackend

        assert policy.replay.dataset is not None
        return ReplayBackend(policy.replay.dataset, policy.replay.episode, fps)
    from fr5_infer.backends.pi05 import Pi05Backend

    assert policy.checkpoint is not None
    return Pi05Backend(policy.checkpoint, cameras, policy.device)


class InferApp:
    def __init__(self, config: InferConfig) -> None:
        self.config = config
        self.conversion = load_conversion(config.policy.conversion)  # type: ignore[arg-type]
        cameras = {camera.name: camera for camera in config.cameras}
        check_cameras(
            self.conversion, {name: (c.height, c.width) for name, c in cameras.items()}
        )
        self.checkpoint_info: dict[str, Any] = {}
        if config.policy.backend == "pi05":
            assert config.policy.checkpoint is not None
            self.checkpoint_info = check_checkpoint(config.policy.checkpoint, self.conversion)

        context = mp.get_context("spawn")
        self.mailbox = TrajectoryMailbox(context, config.execution.max_points)
        self.robot_slot = RobotStateSlot(context)
        self.camera_stop = context.Event()
        self.frame_slots: dict[str, LatestFrameSlot] = {}
        self.camera_processes: dict[str, Any] = {}
        for name in self.conversion.cameras:
            camera = cameras[name]
            slot = LatestFrameSlot(context, name, camera.height, camera.width)
            process = context.Process(
                target=run_camera_process,
                args=(camera, slot, self.conversion.latency_ns(name), self.camera_stop,
                      config.teleop.log_level),
                name=f"camera-{name}",
                daemon=True,
            )
            self.frame_slots[name] = slot
            self.camera_processes[name] = process

        self.robot_conn, child_conn = context.Pipe()
        self.robot_process = context.Process(
            target=run_robot_process,
            args=(config, child_conn, self.mailbox, self.robot_slot),
            name="robot",
            daemon=True,
        )
        self._conn_lock = threading.Lock()
        self.policy: PolicyLoop | None = None
        self.backend_info: dict[str, Any] = {"name": config.policy.backend, "loading": True}

    # ── lifecycle ────────────────────────────────────────────────────

    def start(self) -> None:
        for process in self.camera_processes.values():
            process.start()
        self.robot_process.start()
        # Load the model while the robot connects and the cameras settle.
        backend = make_backend(self.config, self.conversion.cameras, self.conversion.fps)
        if hasattr(backend, "warmup"):
            first = self.config.cameras[0]
            home_joints = self.config.teleop.home.joints_deg or (0.0,) * 6
            backend.warmup(
                self._home_state(),
                self.config.policy.task,
                (first.height, first.width),
                np.radians(home_joints).astype(np.float32),
            )
        self.backend_info = {"name": backend.info.name, **backend.info.details}
        self.policy = PolicyLoop(
            self.config,
            self.conversion,
            backend,
            self.mailbox,
            self.robot_slot,
            self.frame_slots,
            self._base_meta(),
        )
        self.policy.start()

    def close(self) -> None:
        if self.policy is not None:
            self.policy.stop_event.set()
            self.policy.join(timeout=PROCESS_JOIN_S)
        if self.robot_process.is_alive():
            self.send("SHUTDOWN")
            self.robot_process.join(timeout=PROCESS_JOIN_S)
        self.camera_stop.set()
        for process in self.camera_processes.values():
            process.join(timeout=PROCESS_JOIN_S)
        for process in (self.robot_process, *self.camera_processes.values()):
            if process.is_alive():
                log.warning("terminating %s", process.name)
                process.terminate()

    # ── commands ─────────────────────────────────────────────────────

    def send(self, *command: Any) -> None:
        with self._conn_lock:
            self.robot_conn.send(command)

    def start_episode(self) -> dict[str, Any]:
        robot = self.robot_slot.read()
        if self.policy is None:
            return {"ok": False, "error": "policy is still loading"}
        if robot.state != "IDLE":
            return {"ok": False, "error": f"robot is {robot.state}"}
        run_dir = new_run_dir(self.config.log.dir, self.config.mode)
        self.policy.arm(run_dir)
        self.mailbox.clear()
        self.send("START", str(run_dir))
        return {"ok": True, "run_dir": str(run_dir)}

    def set_task(self, task: str) -> dict[str, Any]:
        task = task.strip()
        if not task or self.policy is None:
            return {"ok": False, "error": "empty task or policy not loaded"}
        self.policy.set_task(task)
        return {"ok": True, "task": task}

    # ── status ───────────────────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        now_ns = time.monotonic_ns()
        robot = self.robot_slot.read()
        gripper = self.config.teleop.gripper
        robot_status = {
            "alive": self.robot_process.is_alive(),
            "state": robot.state,
            "reason": robot.reason,
            "tcp": [round(v, 2) for v in robot.tcp],
            "cmd_tcp": [round(v, 2) for v in robot.cmd_tcp] if robot.cmd_valid else None,
            "gripper": round(
                normalize_gripper(robot.gripper_position, gripper.open_position, gripper.closed_position),
                3,
            ),
            "gripper_closed": robot.gripper_closed,
            "servo_active": robot.servo_active,
            "episode": robot.episode,
            "state_age_ms": round((now_ns - robot.t_ns) / 1e6, 1) if robot.valid else None,
        }
        if robot.state in {"STARTING", "RUNNING", "GRIPPER"}:
            robot_status["episode_s"] = round((now_ns - robot.episode_start_ns) / 1e9, 1)
            if robot.traj_end_ns:
                robot_status["trajectory_left_s"] = round((robot.traj_end_ns - now_ns) / 1e9, 2)
        cameras = {}
        for name, slot in self.frame_slots.items():
            frame_age = None
            frame = slot.read()
            if frame is not None:
                frame_age = round((now_ns - frame.capture_ns) / 1e6, 1)
            cameras[name] = {
                "alive": self.camera_processes[name].is_alive(),
                "age_ms": frame_age,
                "error": slot.error() or None,
            }
        return {
            "mode": self.config.mode,
            "backend": self.backend_info,
            "robot": robot_status,
            "policy": self.policy.snapshot() if self.policy else {"state": "loading"},
            "cameras": cameras,
            "limits": {
                "max_velocity_mm_s": self.config.max_velocity_mm_s,
                "max_angular_velocity_deg_s": self.config.max_angular_velocity_deg_s,
            },
        }

    def camera_jpeg(self, name: str) -> bytes | None:
        slot = self.frame_slots.get(name)
        frame = slot.read() if slot is not None else None
        if frame is None:
            return None
        from PIL import Image

        image = Image.fromarray(frame.rgb).resize(PREVIEW_SIZE)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=70)
        return buffer.getvalue()

    # ── helpers ──────────────────────────────────────────────────────

    def _home_state(self):  # -> np.ndarray
        gripper = self.config.teleop.gripper
        return encode_state(
            self.config.teleop.home.tcp,  # type: ignore[arg-type]
            gripper.open_position,
            gripper.open_position,
            gripper.closed_position,
        )

    def _base_meta(self) -> dict[str, Any]:
        config = self.config
        return {
            "fr5_infer_version": __version__,
            "mode": config.mode,
            "backend": self.backend_info,
            "checkpoint": self.checkpoint_info,
            "conversion": str(self.conversion.path),
            "dataset_repo_id": self.conversion.repo_id,
            "fps": self.conversion.fps,
            "cameras": list(self.conversion.cameras),
            "teleop_config": str(config.teleop_config),
            "execution": dataclasses.asdict(config.execution),
            "safety": dataclasses.asdict(config.safety),
            "gripper": dataclasses.asdict(config.gripper),
            "limits": {
                "max_velocity_mm_s": config.max_velocity_mm_s,
                "max_angular_velocity_deg_s": config.max_angular_velocity_deg_s,
                "max_step_mm": config.teleop.max_step_mm,
                "workspace": dataclasses.asdict(config.teleop.workspace),
            },
        }
