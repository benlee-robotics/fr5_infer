"""LeRobot Pi0.5 checkpoint, loaded in-process on the GPU.

Based on fr5_vla/fr5_pi05_infer.py (FR5Pi05PolicyRunner). Rotation decoding
is done by fr5_infer.conversion instead of scipy, so this module only maps
observation -> normalized batch -> action chunk -> physical units.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import numpy as np

from fr5_infer.backends import BackendInfo, Observation


# Keep from_pretrained from waiting on the HuggingFace Hub.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

log = logging.getLogger(__name__)


class Pi05Backend:
    def __init__(self, checkpoint: Path, cameras: tuple[str, ...], device: str = "cuda") -> None:
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors
        import lerobot.policies.pi05.configuration_pi05  # noqa: F401  (registers pi05)

        self._torch = torch
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available for the pi05 backend")
        self.device = device
        self.cameras = cameras
        self.use_bf16 = device.startswith("cuda")
        started = time.monotonic()
        config = PreTrainedConfig.from_pretrained(str(checkpoint))
        config.device = device
        policy_class = get_policy_class(config.type)
        self.model = policy_class.from_pretrained(str(checkpoint), config=config).to(device)
        self.model.eval()
        overrides: dict = {"device_processor": {"device": device}}
        # Always use the tokenizer saved with the checkpoint. lerobot 0.6
        # resolves it itself; older checkpoints name the gated
        # google/paligemma-3b-pt-224 hub tokenizer instead, which an offline
        # machine (or container) cannot load without a HuggingFace cache.
        tokenizer_dir = checkpoint / "tokenizer"
        if tokenizer_dir.is_dir():
            overrides["tokenizer_processor"] = {"tokenizer_name": str(tokenizer_dir)}
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.model.config,
            pretrained_path=str(checkpoint),
            preprocessor_overrides=overrides,
        )
        # The dataset also stores observation.joint_pos; feed it when the
        # checkpoint declares it, as in training.
        self.uses_joint_pos = "observation.joint_pos" in (config.input_features or {})
        load_s = time.monotonic() - started
        self.info = BackendInfo(
            name="pi05",
            details={
                "checkpoint": str(checkpoint),
                "type": config.type,
                "chunk_size": getattr(config, "chunk_size", None),
                "device": device,
                "load_s": round(load_s, 1),
            },
        )
        log.info("pi05 loaded from %s in %.1f s", checkpoint, load_s)

    def warmup(
        self,
        state: np.ndarray,
        task: str,
        shape: tuple[int, int],
        joint_pos: np.ndarray | None = None,
    ) -> float:
        images = {name: np.zeros((*shape, 3), dtype=np.uint8) for name in self.cameras}
        started = time.monotonic()
        self.predict(
            Observation(state=state, images=images, task=task, t_obs_ns=0, joint_pos=joint_pos)
        )
        elapsed = time.monotonic() - started
        if self.device.startswith("cuda"):
            memory_gb = self._torch.cuda.max_memory_allocated() / 1e9
            self.info.details["gpu_memory_gb"] = round(memory_gb, 2)
        self.info.details["warmup_s"] = round(elapsed, 2)
        log.info("pi05 warmup %.2f s %s", elapsed, self.info.details)
        return elapsed

    def reset(self) -> None:
        if hasattr(self.model, "reset"):
            self.model.reset()

    def _image(self, rgb: np.ndarray):  # -> torch.Tensor
        tensor = self._torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(dtype=self._torch.float32) / 255.0

    def predict(self, observation: Observation) -> np.ndarray:
        torch = self._torch
        batch = {
            f"observation.images.{name}": self._image(observation.images[name])
            for name in self.cameras
        }
        batch["observation.state"] = torch.from_numpy(
            np.asarray(observation.state, dtype=np.float32)
        ).unsqueeze(0)
        if self.uses_joint_pos:
            if observation.joint_pos is None:
                raise ValueError("the checkpoint needs observation.joint_pos")
            batch["observation.joint_pos"] = torch.from_numpy(
                np.asarray(observation.joint_pos, dtype=np.float32)
            ).unsqueeze(0)
        batch["task"] = observation.task
        with torch.no_grad():
            processed = self.preprocessor(batch)
            with torch.autocast(
                device_type="cuda" if self.device.startswith("cuda") else "cpu",
                dtype=torch.bfloat16,
                enabled=self.use_bf16,
            ):
                actions = self.model.predict_action_chunk(processed)
            decoded = self.postprocessor(actions)
        return decoded[0].detach().cpu().float().numpy()
