"""Time one gripper action with the arm standing still (fr5_teleop 0.4.2 check).

Sends MoveGripper (non-blocking) and polls GetGripperMotionDone every 20 ms
until it reports done, then 0.3 s more. Prints how long the command call
took, the done/fault sequence and the gripper position. With fr5_teleop
0.4.1 done came ~20 ms after the command although the gripper moved ~1 s.

The robot is connected and enabled as in --mode real (errors reset, robot
enabled, gripper activated); the arm is never commanded.

    python scripts/gripper_check.py close
    python scripts/gripper_check.py open
    python scripts/gripper_check.py close --fake   # without a robot
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fr5_infer.config import InferConfig  # noqa: E402

POLL_S = 0.02
AFTER_DONE_S = 0.3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("close", "open"))
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--fake", action="store_true", help="FakeRobotClient, 0.8 s motion")
    args = parser.parse_args()

    config = InferConfig.from_yaml(args.config, overrides={"mode": "fake" if args.fake else "real"})
    gripper = config.teleop.gripper
    if args.fake:
        from teleop.robot.fake_client import FakeRobotBehavior, FakeRobotClient

        client = FakeRobotClient(FakeRobotBehavior(gripper_motion_s=0.8))
    else:
        from teleop.robot.fairino_client import FairinoRobotClient

        client = FairinoRobotClient(config.teleop)
    client.connect()
    try:
        client.initialize()
        before = client.get_state_snapshot()
        target = gripper.closed_position if args.action == "close" else gripper.open_position
        started = time.monotonic()
        client.move_gripper(target)
        command_s = time.monotonic() - started
        samples = []  # (t_s, done, fault, position)
        done_s = None
        while True:
            now = time.monotonic() - started
            motion = client.get_gripper_motion_state()
            snapshot = client.get_state_snapshot()
            position = None if snapshot is None else round(snapshot.gripper_position)
            samples.append((now, motion.done, motion.fault, position))
            if motion.done and done_s is None:
                done_s = now
            if done_s is not None and now > done_s + AFTER_DONE_S:
                break
            if now > gripper.action_timeout_s:
                break
            time.sleep(POLL_S)
    finally:
        client.close()

    print(f"action {args.action} -> position {target} (velocity {gripper.velocity}, force {gripper.force})")
    print(f"gripper position before: {None if before is None else round(before.gripper_position)}")
    print(f"MoveGripper call returned after {command_s * 1000:.0f} ms")
    print(f"first poll {samples[0][0] * 1000:.0f} ms after the command: done={samples[0][1]} fault={samples[0][2]}")
    if done_s is None:
        print(f"NOT DONE within action_timeout_s {gripper.action_timeout_s} s (would be gripper_timeout)")
    else:
        print(f"done after {done_s:.2f} s")
    # Runs of equal (done, fault): when the reported state changed.
    runs = []
    for t, done, fault, position in samples:
        if not runs or tuple(runs[-1][1:3]) != (done, fault):
            runs.append([t, done, fault, position, position])
        runs[-1][4] = position
    for t, done, fault, first, last in runs:
        print(f"  from {t:5.2f} s  done={int(done)} fault={fault}  position {first} -> {last}")
    print(f"gripper position after: {samples[-1][3]}")


if __name__ == "__main__":
    main()
