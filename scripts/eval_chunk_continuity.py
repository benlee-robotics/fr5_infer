"""Open-loop check of how consecutive chunks join: independent, RTC, ensemble.

Walks recorded episodes like the robot would: infer at frames 0, R, 2R, ...
(R = --replan) from the recorded observation, the frame time standing in for
t_obs. Chunk row k is the action for frame t + k. The first --delay steps of
each chunk are spent waiting for the inference, so the robot switches from
the previous chunk to the new one at frame t + d. Reported per variant:

- jump: position/rotation difference between the previous and the new
  trajectory at the switch (what the 0.15 s cross-fade has to absorb)
- err: mean position error of the executed part [t + d, t + R + d) against
  the recorded actions (does continuity cost accuracy?)
- latency of the inference

Variants: "independent" (every chunk from fresh noise, as v0.1.0) and "rtc"
run their own inference sequences; "+ens" applies temporal ensembling on top
of each. The checkpoint is loaded once with RTC set up; "independent" simply
never passes a previous chunk.

    python scripts/eval_chunk_continuity.py \\
        --checkpoint checkpoints/fr5_pick_cube_pi05_expert/002500/pretrained_model \\
        --dataset ../fr5_teleop/datasets/fr5_pick_cube_20261027 --episodes 0 40 80
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fr5_infer.backends import Observation  # noqa: E402
from fr5_infer.backends.pi05 import Pi05Backend  # noqa: E402
from fr5_infer.config import RtcConfig  # noqa: E402
from fr5_infer.conversion import check_checkpoint, decode_chunk, load_conversion  # noqa: E402
from fr5_infer.trajectory import TemporalEnsemble, Trajectory, align_chunk  # noqa: E402
from teleop.control_math import quat_angle_deg  # noqa: E402


def jump(old: Trajectory, new: Trajectory, t_ns: int) -> tuple[float, float]:
    old_position, old_quat = old.sample(t_ns)
    new_position, new_quat = new.sample(t_ns)
    return (
        float(np.linalg.norm(np.subtract(new_position, old_position))),
        float(quat_angle_deg(old_quat, new_quat)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", default=[0])
    parser.add_argument("--replan", type=int, default=20, help="execution.replan_steps")
    parser.add_argument("--delay-s", type=float, default=0.15, help="inference latency + min_lead_s")
    parser.add_argument("--horizon", type=int, default=10, help="policy.rtc.execution_horizon")
    parser.add_argument("--guidance", type=float, default=10.0, help="policy.rtc.max_guidance_weight")
    parser.add_argument("--schedule", default="LINEAR")
    parser.add_argument("--ensemble-decay", type=float, default=0.01)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    conversion = load_conversion(args.dataset / "fr5_conversion.json")
    print("checkpoint:", check_checkpoint(args.checkpoint, conversion))
    rtc = RtcConfig(True, args.horizon, args.guidance, args.schedule)
    backend = Pi05Backend(args.checkpoint, conversion.cameras, args.device, conversion.fps, rtc)
    print("loaded:", json.dumps(backend.info.details))
    period_ns = round(1e9 / conversion.fps)
    delay = math.ceil(args.delay_s * conversion.fps)

    results: dict[str, dict[str, list[float]]] = {}
    for episode in args.episodes:
        dataset = LeRobotDataset(f"local/{args.dataset.name}", root=args.dataset, episodes=[episode])
        actions = np.stack([dataset.hf_dataset[i]["action"].numpy() for i in range(len(dataset))])
        truth = decode_chunk(actions)
        for name, guided in (("independent", False), ("rtc", True)):
            backend.reset()
            previous: dict[str, Trajectory | None] = {name: None, f"{name}+ens": None}
            ensemble = TemporalEnsemble(args.ensemble_decay)
            for t in range(0, len(dataset) - args.replan, args.replan):
                sample = dataset[t]
                images = {
                    camera: (sample[f"observation.images.{camera}"].permute(1, 2, 0).numpy() * 255.0)
                    .round()
                    .astype(np.uint8)
                    for camera in conversion.cameras
                }
                observation = Observation(
                    state=sample["observation.state"].numpy(),
                    images=images,
                    task=sample["task"],
                    t_obs_ns=t * period_ns,
                    joint_pos=sample["observation.joint_pos"].numpy(),
                )
                if not guided:
                    backend.break_continuation()
                started = time.monotonic()
                chunk = backend.predict(observation, args.delay_s)
                latency_ms = (time.monotonic() - started) * 1000
                decoded = decode_chunk(chunk)
                aligned = align_chunk(decoded, t * period_ns, conversion.fps)
                keep = np.arange(len(aligned)) >= delay
                executed = aligned.select(keep)
                variants = {name: executed}
                variants[f"{name}+ens"], _ = ensemble.add(executed, decoded.gripper[keep])
                switch_ns = (t + delay) * period_ns
                for variant, trajectory in variants.items():
                    stats = results.setdefault(variant, {})
                    end = min(t + args.replan + delay, len(actions))
                    frames = np.arange(t + delay, end)
                    planned = np.array([trajectory.sample(int(f * period_ns))[0] for f in frames])
                    error = np.linalg.norm(planned - truth.position_mm[frames], axis=1)
                    stats.setdefault("err_mm", []).append(float(error.mean()))
                    stats.setdefault("latency_ms", []).append(latency_ms)
                    if previous[variant] is not None:
                        jump_mm, jump_deg = jump(previous[variant], trajectory, switch_ns)
                        stats.setdefault("jump_mm", []).append(jump_mm)
                        stats.setdefault("jump_deg", []).append(jump_deg)
                    previous[variant] = trajectory
            print(f"episode {episode} {name}: {len(results[name]['err_mm'])} chunks")

    print(f"\nreplan {args.replan} steps, delay {delay} steps, RTC horizon {args.horizon}, "
          f"guidance {args.guidance}, {args.schedule}; ensemble decay {args.ensemble_decay}")
    print(f"{'variant':18s} {'jump mm p50/p90/max':>22s} {'jump deg p50/max':>17s} "
          f"{'err mm':>7s} {'latency ms p50/max':>19s}")
    for variant, stats in results.items():
        jump_mm, jump_deg = np.array(stats["jump_mm"]), np.array(stats["jump_deg"])
        # The first chunk of each sequence warms up nothing; skip it for latency.
        latency = np.array(stats["latency_ms"][1:])
        print(
            f"{variant:18s} {np.percentile(jump_mm, 50):6.1f} {np.percentile(jump_mm, 90):6.1f} "
            f"{jump_mm.max():6.1f}   {np.percentile(jump_deg, 50):6.2f} {jump_deg.max():6.2f}   "
            f"{np.mean(stats['err_mm']):6.1f}   {np.percentile(latency, 50):7.0f} {latency.max():7.0f}"
        )


if __name__ == "__main__":
    main()
