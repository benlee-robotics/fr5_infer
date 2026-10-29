"""fr5_infer settings (config.yaml) plus the fr5_teleop config it points to.

Robot IP, SDK path, workspace, motion limits, gripper, home pose and camera
serials are read from the fr5_teleop config so both tools share one source
of truth. Only policy execution settings live here.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from teleop.config import CameraConfig, TeleopConfig


MODES = ("fake", "shadow", "real")
BACKENDS = ("pi05", "replay")
RTC_SCHEDULES = ("ZEROS", "ONES", "LINEAR", "EXP")


@dataclass(frozen=True)
class ReplayConfig:
    # LeRobot dataset root; its recorded actions stand in for a policy.
    dataset: Path | None = None
    episode: int = 0


@dataclass(frozen=True)
class RtcConfig:
    # Real-Time Chunking, lerobot's inference-time ("guided") mode, so any
    # pi05 checkpoint works: each new chunk is steered towards the part of the
    # previous chunk that is still to be executed, which keeps consecutive
    # chunks on one path instead of independent samples.
    enabled: bool = False
    # Previous-chunk steps the guidance covers. Steps that run during the
    # inference weigh 1, the weight then falls to 0 at this step.
    execution_horizon: int = 10
    max_guidance_weight: float = 10.0
    schedule: str = "LINEAR"  # ZEROS | ONES | LINEAR | EXP


@dataclass(frozen=True)
class PolicyConfig:
    backend: str = "pi05"
    checkpoint: Path | None = None
    # fr5_conversion.json of the training dataset: fps, rotation, cameras.
    conversion: Path | None = None
    task: str = "pick up the cube and place it in the box"
    device: str = "cuda"
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    rtc: RtcConfig = field(default_factory=RtcConfig)


@dataclass(frozen=True)
class ExecutionConfig:
    # Re-infer after this many chunk steps were executed (guide: 15-25).
    replan_steps: int = 20
    # ...or earlier when the remaining trajectory is shorter than the latency
    # EMA plus this margin.
    refill_margin_s: float = 0.15
    # Chunk points due sooner than this are dropped before sending.
    min_lead_s: float = 0.02
    # Overlap with the previous trajectory that is cross-faded.
    blend_s: float = 0.15
    # ACT-style temporal ensembling: every point is the weighted mean of all
    # chunks predicted for that time, the i-th (0 = oldest) weighing
    # exp(-ensemble_decay * i). The gripper output is averaged too.
    ensemble: bool = False
    ensemble_decay: float = 0.01
    # Sample the trajectory this far ahead of now (servo latency).
    lookahead_s: float = 0.0
    # Stop the servo this long after the last trajectory point.
    trajectory_grace_s: float = 0.10
    max_points: int = 64
    # Give up waiting for the first (or post-gripper) trajectory.
    start_timeout_s: float = 15.0


@dataclass(frozen=True)
class SafetyConfig:
    max_start_offset_mm: float = 20.0
    max_start_offset_deg: float = 5.0
    # Applied as min(fr5_teleop motion/orientation limit, this value).
    max_velocity_mm_s: float = 100.0
    max_angular_velocity_deg_s: float = 15.0
    # Absolute range around the home orientation.
    max_orientation_deviation_deg: float = 45.0
    max_episode_s: float = 90.0
    max_state_age_s: float = 0.1
    max_frame_age_s: float = 0.2


@dataclass(frozen=True)
class GripperPolicyConfig:
    close_threshold: float = 0.6
    open_threshold: float = 0.4
    min_dwell_s: float = 0.5


@dataclass(frozen=True)
class WebConfig:
    host: str = "127.0.0.1"
    port: int = 8080


@dataclass(frozen=True)
class LogConfig:
    dir: Path = Path("runs")
    # Also store the policy's input images (JPEG) for every inference.
    save_images: bool = False


@dataclass(frozen=True)
class InferConfig:
    teleop_config: Path
    teleop: TeleopConfig
    mode: str = "fake"
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    gripper: GripperPolicyConfig = field(default_factory=GripperPolicyConfig)
    web: WebConfig = field(default_factory=WebConfig)
    log: LogConfig = field(default_factory=LogConfig)

    # ── derived values ───────────────────────────────────────────────

    @property
    def max_velocity_mm_s(self) -> float:
        return min(self.teleop.max_velocity_mm_s, self.safety.max_velocity_mm_s)

    @property
    def max_angular_velocity_deg_s(self) -> float:
        return min(
            self.teleop.orientation.max_angular_velocity_deg_s,
            self.safety.max_angular_velocity_deg_s,
        )

    @property
    def cameras(self) -> tuple[CameraConfig, ...]:
        """Every camera of the teleop config, enabled, fake in fake mode."""
        cameras = []
        for camera in self.teleop.recording.cameras:
            changes: dict[str, Any] = {"enabled": True}
            if self.mode == "fake":
                changes["source"] = "fake"
            cameras.append(dataclasses.replace(camera, **changes))
        return tuple(cameras)

    def validate(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}")
        if self.policy.backend not in BACKENDS:
            raise ValueError(f"policy.backend must be one of {', '.join(BACKENDS)}")
        if self.policy.backend == "pi05" and self.policy.checkpoint is None:
            raise ValueError("policy.checkpoint is required for the pi05 backend")
        if self.policy.backend == "replay" and self.policy.replay.dataset is None:
            raise ValueError("policy.replay.dataset is required for the replay backend")
        rtc = self.policy.rtc
        if rtc.enabled and self.policy.backend != "pi05":
            raise ValueError("policy.rtc needs the pi05 backend")
        if rtc.execution_horizon < 1 or not rtc.max_guidance_weight > 0:
            raise ValueError("policy.rtc.execution_horizon >= 1 and max_guidance_weight > 0")
        if rtc.schedule not in RTC_SCHEDULES:
            raise ValueError(f"policy.rtc.schedule must be one of {', '.join(RTC_SCHEDULES)}")
        if self.policy.conversion is None or not self.policy.conversion.is_file():
            raise ValueError(f"policy.conversion not found: {self.policy.conversion}")
        if self.teleop.arms:
            raise ValueError("bimanual robot.arms is not supported; use robot.ip")
        if self.mode != "fake" and not self.teleop.robot_ip:
            raise ValueError("robot.ip is required in the teleop config")
        if self.mode == "real" and not self.teleop.gripper.enabled:
            raise ValueError("gripper.enabled must be true in the teleop config")
        if self.teleop.home.tcp is None:
            raise ValueError("home.tcp is required in the teleop config")
        execution = self.execution
        if execution.replan_steps < 1 or execution.max_points < 2:
            raise ValueError("execution.replan_steps >= 1 and max_points >= 2")
        for name, value in (
            *((f"execution.{f.name}", getattr(execution, f.name)) for f in fields(execution)),
            *((f"safety.{f.name}", getattr(self.safety, f.name)) for f in fields(self.safety)),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative number")
        if self.max_velocity_mm_s <= 0 or self.max_angular_velocity_deg_s <= 0:
            raise ValueError("velocity limits must be positive")
        gripper = self.gripper
        if not 0 <= gripper.open_threshold < gripper.close_threshold <= 1:
            raise ValueError("gripper thresholds need 0 <= open < close <= 1")
        if not self.cameras:
            raise ValueError("the teleop config lists no recording.cameras")

    # ── loading ──────────────────────────────────────────────────────

    @classmethod
    def from_yaml(
        cls, path: Path, *, overrides: dict[str, Any] | None = None
    ) -> InferConfig:
        config_path = path.expanduser().resolve()
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"{config_path}: top level must be a mapping")
        base = config_path.parent
        for key, value in (overrides or {}).items():
            section, _, name = key.rpartition(".")
            target = raw
            for part in filter(None, section.split(".")):
                target = target.setdefault(part, {})
            target[name] = value

        _reject_unknown(raw, {f.name for f in fields(cls)} - {"teleop"}, "config")
        teleop_path = _resolve(base, raw.get("teleop_config", "../fr5_teleop/config.yaml"))
        # The TLS files serve the teleop web server only and are not mounted
        # next to fr5_infer; everything else is validated as fr5_teleop does.
        teleop = dataclasses.replace(
            TeleopConfig.from_yaml(teleop_path, validate=False),
            tls_cert_path=None,
            tls_key_path=None,
        )
        teleop.validate()

        policy_raw = dict(raw.get("policy") or {})
        replay = _build(ReplayConfig, policy_raw.pop("replay", None), "policy.replay")
        rtc = _build(RtcConfig, policy_raw.pop("rtc", None), "policy.rtc")
        if replay.dataset is not None:
            replay = dataclasses.replace(replay, dataset=_resolve(base, replay.dataset))
        policy = _build(PolicyConfig, policy_raw, "policy")
        policy = dataclasses.replace(
            policy,
            checkpoint=_resolve(base, policy.checkpoint),
            conversion=_resolve(base, policy.conversion),
            replay=replay,
            rtc=rtc,
        )
        log_config = _build(LogConfig, raw.get("log"), "log")
        config = cls(
            teleop_config=teleop_path,
            teleop=teleop,
            mode=str(raw.get("mode", "fake")),
            policy=policy,
            execution=_build(ExecutionConfig, raw.get("execution"), "execution"),
            safety=_build(SafetyConfig, raw.get("safety"), "safety"),
            gripper=_build(GripperPolicyConfig, raw.get("gripper"), "gripper"),
            web=_build(WebConfig, raw.get("web"), "web"),
            log=dataclasses.replace(log_config, dir=_resolve(base, log_config.dir)),
        )
        config.validate()
        return config


def _resolve(base: Path, value: Any) -> Path | None:
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _reject_unknown(raw: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"{where}: unknown keys {', '.join(sorted(unknown))}")


def _build(cls: type, raw: Any, where: str) -> Any:
    if raw is None:
        return cls()
    if not isinstance(raw, dict):
        raise ValueError(f"{where} must be a mapping")
    _reject_unknown(raw, {f.name for f in fields(cls)}, where)
    values = {}
    for item in fields(cls):
        if item.name not in raw:
            continue
        value = raw[item.name]
        default = getattr(cls(), item.name)
        if isinstance(default, bool):
            value = bool(value)
        elif isinstance(default, int):
            value = int(value)
        elif isinstance(default, float):
            value = float(value)
        values[item.name] = value
    return cls(**values)
