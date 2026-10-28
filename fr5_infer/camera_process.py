"""One process per camera: capture with the fr5_teleop frame source and
overwrite the latest frame in shared memory.

librealsense can block for seconds while holding the GIL, so it stays out of
the robot loop and the inference thread. Capture time is host arrival minus
the camera latency the dataset converter used.
"""

from __future__ import annotations

import logging
from multiprocessing.synchronize import Event

from teleop.config import CameraConfig
from teleop.recording.camera import make_frame_source

from fr5_infer.shm import LatestFrameSlot


log = logging.getLogger(__name__)

RESTART_DELAY_S = 2.0


def run_camera_process(
    config: CameraConfig,
    slot: LatestFrameSlot,
    latency_ns: int,
    stop: Event,
    log_level: str = "INFO",
) -> None:
    logging.basicConfig(
        level=log_level,
        format=f"%(asctime)s %(levelname)s [camera {config.name}] %(name)s: %(message)s",
    )
    while not stop.is_set():
        source = make_frame_source(config)
        try:
            info = source.start()
            log.info("camera %s streaming: %s %s", config.name, info.get("model"), info.get("serial"))
            while not stop.is_set():
                frame = source.read(timeout_s=1.0)
                if frame is not None:
                    slot.write(frame.color, frame.arrival_mono_ns - latency_ns)
        except Exception as exc:  # noqa: BLE001
            log.error("camera %s failed: %s", config.name, exc)
            slot.set_error(str(exc))
            stop.wait(RESTART_DELAY_S)
        finally:
            try:
                source.stop()
            except Exception:  # noqa: BLE001
                pass
