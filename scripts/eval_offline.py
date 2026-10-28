"""Open-loop check of a checkpoint on recorded episodes (no robot, no cameras).

For frames of the chosen episodes, feed the recorded observation (both
images, observation.state, observation.joint_pos, task) through the same
Pi05Backend the robot uses and compare the predicted chunk with the recorded
actions of the next ``chunk`` frames. A "hold" baseline (repeat the current
state) shows whether the policy predicts motion at all.

    python scripts/eval_offline.py \\
        --checkpoint checkpoints/fr5_pick_cube_pi05_expert/002500/pretrained_model \\
        --dataset ../fr5_teleop/datasets/fr5_pick_cube_20261027 --episodes 0 10 40
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fr5_infer.backends import Observation  # noqa: E402
from fr5_infer.backends.pi05 import Pi05Backend  # noqa: E402
from fr5_infer.conversion import check_checkpoint, load_conversion, rot6d_to_matrix  # noqa: E402


def rotation_error_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle between rotations given as rot6d rows (..., 6)."""
    ra, rb = rot6d_to_matrix(a), rot6d_to_matrix(b)
    trace = np.einsum("...ij,...ij->...", ra, rb)  # trace(ra^T rb)
    return np.degrees(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))


def errors(pred: np.ndarray, truth: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "pos_mm": np.linalg.norm(pred[:, :3] - truth[:, :3], axis=1) * 1000.0,
        "rot_deg": rotation_error_deg(pred[:, 3:9], truth[:, 3:9]),
        "gripper_agree": ((pred[:, 9] > 0.5) == (truth[:, 9] > 0.5)).astype(float),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", default=[0])
    parser.add_argument("--stride", type=int, default=30, help="evaluate every N-th frame")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, help="save per-sample results (.npz)")
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    conversion = load_conversion(args.dataset / "fr5_conversion.json")
    print("checkpoint:", check_checkpoint(args.checkpoint, conversion))
    backend = Pi05Backend(args.checkpoint, conversion.cameras, args.device)
    print("loaded:", json.dumps(backend.info.details))

    rows = []
    for episode in args.episodes:
        dataset = LeRobotDataset(
            f"local/{args.dataset.name}", root=args.dataset, episodes=[episode]
        )
        actions = np.stack([dataset.hf_dataset[i]["action"].numpy() for i in range(len(dataset))])
        backend.reset()
        for t in range(0, len(dataset), args.stride):
            sample = dataset[t]
            images = {
                name: (sample[f"observation.images.{name}"].permute(1, 2, 0).numpy() * 255.0)
                .round()
                .astype(np.uint8)
                for name in conversion.cameras
            }
            state = sample["observation.state"].numpy()
            observation = Observation(
                state=state,
                images=images,
                task=sample["task"],
                t_obs_ns=0,
                joint_pos=sample["observation.joint_pos"].numpy(),
            )
            started = time.monotonic()
            chunk = backend.predict(observation)
            latency_s = time.monotonic() - started
            truth = actions[t : t + len(chunk)]
            pred = chunk[: len(truth)]
            hold = np.repeat(state[None], len(truth), axis=0)
            model = errors(pred, truth)
            baseline = errors(hold, truth)
            rows.append(
                {
                    "episode": episode,
                    "frame": t,
                    "latency_s": latency_s,
                    "pos_mm_first": model["pos_mm"][0],
                    "pos_mm_mean": model["pos_mm"].mean(),
                    "pos_mm_end": model["pos_mm"][-1],
                    "rot_deg_mean": model["rot_deg"].mean(),
                    "gripper_agree": model["gripper_agree"].mean(),
                    "hold_pos_mm_mean": baseline["pos_mm"].mean(),
                    "hold_rot_deg_mean": baseline["rot_deg"].mean(),
                    "hold_gripper_agree": baseline["gripper_agree"].mean(),
                    "gripper_raw_first": float(chunk[0, 9]),
                    "gripper_truth_first": float(truth[0, 9]),
                }
            )
            r = rows[-1]
            print(
                f"ep {episode:3d} t {t:4d}  pos {r['pos_mm_first']:6.1f}/{r['pos_mm_mean']:6.1f}"
                f"/{r['pos_mm_end']:6.1f} mm (hold {r['hold_pos_mm_mean']:6.1f})  "
                f"rot {r['rot_deg_mean']:5.2f} deg (hold {r['hold_rot_deg_mean']:5.2f})  "
                f"grip {r['gripper_agree']:.2f} raw {r['gripper_raw_first']:.2f}/"
                f"{r['gripper_truth_first']:.0f}  {latency_s * 1000:5.0f} ms"
            )

    keys = rows[0].keys() - {"episode", "frame"}
    summary = {key: float(np.mean([row[key] for row in rows])) for key in sorted(keys)}
    latencies = np.array([row["latency_s"] for row in rows[1:]] or [rows[0]["latency_s"]])
    summary["latency_ms_p50"] = float(np.percentile(latencies, 50) * 1000)
    summary["latency_ms_max"] = float(latencies.max() * 1000)
    print("\nsummary over", len(rows), "samples (first/mean/end = chunk step 0 / all / last):")
    for key, value in summary.items():
        print(f"  {key:22s} {value:9.3f}")
    if args.out:
        np.savez(args.out, **{key: np.array([row[key] for row in rows]) for key in rows[0]})


if __name__ == "__main__":
    main()
