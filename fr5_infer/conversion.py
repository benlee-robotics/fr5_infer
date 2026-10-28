"""Observation and action vectors, identical to the dataset converter.

Encoding reuses ``teleop.recording.lerobot_convert`` (the code that built the
training dataset). Only the inverse direction (rot6d back to a TCP pose) is
new here: Gram-Schmidt on the two columns, then the converter's quaternion
routine and the teleop Euler convention (R = Rz·Ry·Rx).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from teleop.control_math import TcpPose, quat_to_euler_deg
from teleop.recording.lerobot_convert import matrix_to_quat_xyzw, pose_vector


STATE_DIM = 10
ACTION_DIM = 10


@dataclass(frozen=True)
class ConversionInfo:
    """What the training dataset was built with (fr5_conversion.json)."""

    path: Path
    repo_id: str
    fps: int
    cameras: tuple[str, ...]
    camera_latency_ns: dict[str, int]
    # Per camera (height, width) from the dataset meta/info.json, if present.
    image_shapes: dict[str, tuple[int, int]]

    def latency_ns(self, camera: str) -> int:
        return self.camera_latency_ns.get(camera, 0)


def load_conversion(path: Path) -> ConversionInfo:
    data = json.loads(path.read_text(encoding="utf-8"))
    options = data["options"]
    if options.get("rotation") != "rot6d":
        raise ValueError(f"{path}: rotation {options.get('rotation')!r}, only rot6d is supported")
    if options.get("action_mode") != "command":
        raise ValueError(
            f"{path}: action_mode {options.get('action_mode')!r}, only command is supported"
        )
    cameras = tuple(data.get("cameras") or ())
    default_ms = float(options.get("camera_latency_ms", 0.0))
    by_name = options.get("camera_latency_ms_by_name") or {}
    latency = {
        name: int(float(by_name.get(name, default_ms)) * 1e6) for name in cameras
    }
    shapes: dict[str, tuple[int, int]] = {}
    info_path = path.parent / "meta" / "info.json"
    if info_path.is_file():
        features = json.loads(info_path.read_text(encoding="utf-8"))["features"]
        for name in cameras:
            feature = features.get(f"observation.images.{name}")
            if feature is not None:
                height, width = feature["shape"][:2]
                shapes[name] = (int(height), int(width))
    return ConversionInfo(
        path=path,
        repo_id=str(data.get("repo_id", "")),
        fps=int(options["fps"]),
        cameras=cameras,
        camera_latency_ns=latency,
        image_shapes=shapes,
    )


def check_cameras(
    conversion: ConversionInfo, cameras: dict[str, tuple[int, int]]
) -> None:
    """Configured cameras (name -> (height, width)) must match the dataset."""
    missing = [name for name in conversion.cameras if name not in cameras]
    if missing:
        raise ValueError(
            f"cameras {missing} of the dataset are not in the teleop config "
            f"recording.cameras (have {sorted(cameras)})"
        )
    for name, shape in conversion.image_shapes.items():
        if cameras[name] != shape:
            raise ValueError(
                f"camera {name} is {cameras[name][1]}x{cameras[name][0]}, "
                f"the dataset recorded {shape[1]}x{shape[0]}"
            )


def check_checkpoint(checkpoint: Path, conversion: ConversionInfo) -> dict[str, Any]:
    """The policy's input/output features must be what the dataset provides."""
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    inputs = config.get("input_features") or {}
    outputs = config.get("output_features") or {}
    problems = []
    state = inputs.get("observation.state")
    if state is None or list(state["shape"]) != [STATE_DIM]:
        problems.append(f"observation.state shape {state and state['shape']} != [{STATE_DIM}]")
    action = outputs.get("action")
    if action is None or list(action["shape"]) != [ACTION_DIM]:
        problems.append(f"action shape {action and action['shape']} != [{ACTION_DIM}]")
    images = sorted(
        key.removeprefix("observation.images.")
        for key in inputs
        if key.startswith("observation.images.")
    )
    if images != sorted(conversion.cameras):
        problems.append(f"policy cameras {images} != dataset cameras {sorted(conversion.cameras)}")
    for name, (height, width) in conversion.image_shapes.items():
        feature = inputs.get(f"observation.images.{name}")
        if feature is not None and list(feature["shape"]) != [3, height, width]:
            problems.append(
                f"observation.images.{name} shape {feature['shape']} != [3, {height}, {width}]"
            )
    if problems:
        raise ValueError(f"checkpoint {checkpoint} does not match the dataset: " + "; ".join(problems))
    return {
        "type": config.get("type"),
        "chunk_size": config.get("chunk_size"),
        "n_action_steps": config.get("n_action_steps"),
    }


# ── vectors ─────────────────────────────────────────────────────────


def normalize_gripper(position: float, open_position: float, closed_position: float) -> float:
    """Same as the gripper() helper inside lerobot_convert.build_episode."""
    span = closed_position - open_position
    return float(np.clip((position - open_position) / span, 0.0, 1.0))


def encode_state(
    tcp: TcpPose, gripper_position: float, open_position: float, closed_position: float
) -> np.ndarray:
    """[x,y,z m, rot6d..., gripper 0..1] as float32, like observation.state."""
    pose = pose_vector(np.asarray([tcp], dtype=np.float64), "rot6d")[0]
    gripper = normalize_gripper(gripper_position, open_position, closed_position)
    return np.concatenate([pose, [gripper]]).astype(np.float32)


def rot6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    """(..., 6) first two columns -> orthonormal (..., 3, 3) (Gram-Schmidt)."""
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]
    b1 = a1 / np.clip(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-8, None)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.clip(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-8, None)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


@dataclass(frozen=True)
class DecodedChunk:
    position_mm: np.ndarray  # (T, 3)
    quat_xyzw: np.ndarray  # (T, 4), sign-continuous along T
    gripper: np.ndarray  # (T,) raw policy output, ~0 open .. ~1 closed


def decode_chunk(chunk: np.ndarray) -> DecodedChunk:
    chunk = np.asarray(chunk, dtype=np.float64)
    if chunk.ndim != 2 or chunk.shape[1] != ACTION_DIM:
        raise ValueError(f"action chunk must be (T, {ACTION_DIM}), got {chunk.shape}")
    if not np.isfinite(chunk).all():
        raise ValueError("action chunk has non-finite values")
    return DecodedChunk(
        position_mm=chunk[:, 0:3] * 1000.0,
        quat_xyzw=matrix_to_quat_xyzw(rot6d_to_matrix(chunk[:, 3:9])),
        gripper=chunk[:, 9].copy(),
    )


def pose_to_tcp(position_mm: Any, quat_xyzw: Any) -> TcpPose:
    rx, ry, rz = quat_to_euler_deg(tuple(float(v) for v in quat_xyzw))
    return (
        float(position_mm[0]),
        float(position_mm[1]),
        float(position_mm[2]),
        rx,
        ry,
        rz,
    )
