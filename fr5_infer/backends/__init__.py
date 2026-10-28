"""Policy backends: something that turns an observation into an action chunk."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class Observation:
    state: np.ndarray  # (10,) float32, observation.state
    images: dict[str, np.ndarray]  # camera name -> (H, W, 3) uint8 RGB
    task: str
    t_obs_ns: int
    # Seconds since the episode started (the replay backend indexes by it).
    episode_time_s: float = 0.0
    # (6,) float32 radians, like the dataset's observation.joint_pos.
    joint_pos: np.ndarray | None = None


@dataclass
class BackendInfo:
    name: str
    details: dict[str, Any] = field(default_factory=dict)


class PolicyBackend(Protocol):
    info: BackendInfo

    def reset(self) -> None:
        """Called at every episode start."""

    def predict(self, observation: Observation) -> np.ndarray:
        """(T, 10) action chunk in physical units (dataset action space)."""
