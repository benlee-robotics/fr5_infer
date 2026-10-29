"""Recorded dataset actions as a stand-in policy (no model, no GPU).

Returns ``action[k0 : k0 + chunk]`` of one episode, with k0 the dataset frame
at the current episode time, so the whole pipeline (timing, blending,
gripper sequence, limits) can be checked before a checkpoint is available.
The observation is ignored, so start the robot where the episode started.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from fr5_infer.backends import BackendInfo, Observation


class ReplayBackend:
    def __init__(
        self,
        dataset: Path,
        episode: int,
        fps: int,
        chunk_size: int = 50,
        latency_s: float = 0.1,
    ) -> None:
        import pandas as pd

        files = sorted((dataset / "data").glob("chunk-*/file-*.parquet"))
        if not files:
            raise FileNotFoundError(f"no parquet files under {dataset / 'data'}")
        frames = pd.concat(
            [pd.read_parquet(path, columns=["action", "episode_index", "frame_index"]) for path in files]
        )
        rows = frames[frames["episode_index"] == episode].sort_values("frame_index")
        if rows.empty:
            raise ValueError(f"episode {episode} not found in {dataset}")
        self.actions = np.stack(rows["action"].to_numpy()).astype(np.float32)
        self.fps = fps
        self.chunk_size = chunk_size
        # Simulated inference time, so latency handling is exercised too.
        self.latency_s = latency_s
        self.info = BackendInfo(
            name="replay",
            details={
                "dataset": str(dataset),
                "episode": episode,
                "frames": len(self.actions),
                "duration_s": round(len(self.actions) / fps, 2),
            },
        )

    @property
    def first_action(self) -> np.ndarray:
        return self.actions[0]

    def reset(self) -> None:
        pass

    def break_continuation(self) -> None:
        pass

    def predict(self, observation: Observation, inference_delay_s: float = 0.0) -> np.ndarray:
        time.sleep(self.latency_s)
        start = int(round(observation.episode_time_s * self.fps))
        start = min(max(start, 0), len(self.actions) - 1)
        # Shorter chunks near the end: the trajectory runs out and the robot
        # stops with trajectory_exhausted, like a real end of episode.
        return self.actions[start : start + self.chunk_size].copy()
