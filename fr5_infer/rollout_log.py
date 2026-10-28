"""Per-episode rollout record under ``log.dir/<timestamp>_<mode>/``.

    meta.json        mode, backend, checkpoint, conversion, task, settings,
                     end reason, inference latency stats
    inferences.npz   one row per inference: observation time, latency,
                     observation.state, raw action chunk, what was sent
    ticks.npy        every 8 ms robot tick (written by the robot process,
                     dtype robot_process.TICK_DTYPE)
    images/          policy input frames (log.save_images)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np


def new_run_dir(root: Path, mode: str) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = root / f"{stamp}_{mode}"
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = root / f"{stamp}_{mode}_{suffix}"
    run_dir.mkdir(parents=True)
    return run_dir


class EpisodeLog:
    def __init__(self, run_dir: Path, meta: dict[str, Any], save_images: bool) -> None:
        self.run_dir = run_dir
        self.meta = dict(meta)
        self.meta["started_wall"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.save_images = save_images
        self.rows: list[dict[str, Any]] = []
        self._write_meta()

    def add_inference(self, row: dict[str, Any], images: dict[str, np.ndarray] | None) -> None:
        index = len(self.rows)
        self.rows.append(row)
        if self.save_images and images:
            from PIL import Image

            folder = self.run_dir / "images"
            folder.mkdir(exist_ok=True)
            for name, rgb in images.items():
                Image.fromarray(rgb).save(folder / f"{index:05d}_{name}.jpg", quality=90)

    def close(self, end_reason: str, extra: dict[str, Any] | None = None) -> None:
        if self.rows:
            arrays = {key: _stack([row[key] for row in self.rows]) for key in self.rows[0]}
            np.savez_compressed(self.run_dir / "inferences.npz", **arrays)
            latency = arrays["latency_s"]
            self.meta["inference"] = {
                "count": len(self.rows),
                "latency_s_p50": round(float(np.percentile(latency, 50)), 4),
                "latency_s_p90": round(float(np.percentile(latency, 90)), 4),
                "latency_s_max": round(float(latency.max()), 4),
            }
        self.meta["end_reason"] = end_reason
        self.meta["ended_wall"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.meta.update(extra or {})
        self._write_meta()

    def _write_meta(self) -> None:
        (self.run_dir / "meta.json").write_text(
            json.dumps(self.meta, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )


def _stack(values: list[Any]) -> np.ndarray:
    """Stack rows; arrays of different lengths (short chunks) are NaN-padded."""
    arrays = [np.asarray(value) for value in values]
    if len({array.shape for array in arrays}) <= 1:
        return np.stack(arrays)
    length = max(len(array) for array in arrays)
    padded = np.full((len(arrays), length, *arrays[0].shape[1:]), np.nan, dtype=np.float32)
    for i, array in enumerate(arrays):
        padded[i, : len(array)] = array
    return padded
