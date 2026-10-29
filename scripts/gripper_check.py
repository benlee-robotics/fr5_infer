"""Time one gripper action with the arm standing still (fr5_teleop gripper check).

Sends MoveGripper (non-blocking) and polls the client every 20 ms until it
reports done, then 0.5 s more. Prints how long the command call took, each
change of the client's done/fault verdict next to the raw
GetGripperMotionDone result and the CNDE gripper fields it is based on, and
when the measured position stopped changing.

Expected with fr5_teleop 0.4.3 (field FR5 + Robotiq): raw [1, 0] while
moving, then [1, 3] at the target or [1, 2] holding an object, done from the
first non-zero status, fault 0; the position reads 0 while moving and
settles 0.1-0.3 s after done.

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
AFTER_DONE_S = 0.5
CNDE_FIELDS = ("gripper_motiondone", "gripper_fault", "gripper_fault_id", "gripper_active", "gripper_current")


def raw_state(client) -> tuple:
    """(GetGripperMotionDone result, CNDE gripper fields), None for the fake client."""
    robot = getattr(client, "_robot", None)
    if robot is None:
        return (None, None)
    pkg = getattr(robot, "robot_state_pkg", None)
    fields = tuple(getattr(pkg, name, None) for name in CNDE_FIELDS)
    return (robot.GetGripperMotionDone(), fields)


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
        samples = []  # (t_s, done, fault, position, raw)
        done_s = None
        while True:
            now = time.monotonic() - started
            motion = client.get_gripper_motion_state()
            raw = raw_state(client)
            snapshot = client.get_state_snapshot()
            position = None if snapshot is None else round(snapshot.gripper_position)
            samples.append((now, motion.done, motion.fault, position, raw))
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
    # Runs of equal verdict and raw state: when anything but the position changed.
    print(f"  CNDE fields: {', '.join(CNDE_FIELDS)}")
    runs = []
    for t, done, fault, position, raw in samples:
        if not runs or tuple(runs[-1][1:4]) != (done, fault, raw):
            runs.append([t, done, fault, raw, position, position])
        runs[-1][5] = position
    for t, done, fault, raw, first, last in runs:
        print(f"  from {t:5.2f} s  done={int(done)} fault={fault}  raw {raw[0]} cnde {raw[1]}"
              f"  position {first} -> {last}")
    final = samples[-1][3]
    settled_s = None  # first poll from which the position stayed final
    for t, _, _, position, _ in samples:
        if position != final:
            settled_s = None
        elif settled_s is None:
            settled_s = t
    print(f"gripper position after: {final}, unchanged since {settled_s:.2f} s")


if __name__ == "__main__":
    main()
