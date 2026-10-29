"""Summarize one run folder: gripper actions, chunk switches, RTC, speed.

    python scripts/analyze_run.py runs/20261029_142412_real
    python scripts/analyze_run.py            # newest folder under runs/

- gripper: per action how long GRIPPER took and how long the first
  STARTING -> RUNNING tick took (a ServoMoveStart blocked by a still moving
  gripper shows up as ~1 s), the gripper value of the first observation
  after it (0.0 after a grasp means the gripper still read as open)
- switch jump: distance between the previous and the new chunk where the
  robot changes over (the first sent point), like
  scripts/eval_chunk_continuity.py offline; gripper actions excluded
- rtc: inferences guided by the previous chunk (rtc_prefix_steps > 0)
- cmd: commanded speed and cmd - tcp tracking, tick period
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

STATES = ("IDLE", "HOMING", "STARTING", "RUNNING", "GRIPPER", "FAULT")
PERIOD_S = 1 / 30


def percentiles(values: np.ndarray, label: str, unit: str) -> str:
    if len(values) == 0:
        return f"{label}: none"
    p50, p90 = np.percentile(values, [50, 90])
    return f"{label}: p50 {p50:.1f} p90 {p90:.1f} max {values.max():.1f} {unit} (n={len(values)})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, nargs="?")
    args = parser.parse_args()
    run = args.run or sorted(p for p in Path("runs").iterdir() if p.is_dir())[-1]
    meta = json.loads((run / "meta.json").read_text())
    backend = meta.get("backend", {})
    print(f"{run}  end {meta.get('end_reason')}  velocity {meta['safety']['max_velocity_mm_s']} mm/s  "
          f"rtc {backend.get('rtc', '-')}  ensemble {meta['execution'].get('ensemble', '-')}")

    ticks = np.load(run / "ticks.npy")
    if (run / "inferences.npz").exists():
        inf = np.load(run / "inferences.npz")
    else:
        inf = None
    t0 = ticks["t_ns"][0] if len(ticks) else 0
    state = ticks["state"]

    # Gripper actions: GRIPPER entry -> STARTING -> RUNNING.
    print("gripper actions:")
    k = 1
    while k < len(ticks):
        if state[k] == STATES.index("GRIPPER") and state[k - 1] != state[k]:
            start = k
            while k < len(ticks) and state[k] == STATES.index("GRIPPER"):
                k += 1
            if k >= len(ticks):
                break
            starting = k
            while k < len(ticks) and state[k] == STATES.index("STARTING"):
                k += 1
            gripper_s = (ticks["t_ns"][starting] - ticks["t_ns"][start]) / 1e9
            line = (f"  t {(ticks['t_ns'][start] - t0) / 1e9:6.2f} s  "
                    f"{'close' if ticks['gripper_closed'][starting] else 'open '}  GRIPPER {gripper_s:.2f} s")
            if k < len(ticks) and state[k] == STATES.index("RUNNING"):
                waiting_s = (ticks["t_ns"][k] - ticks["t_ns"][starting]) / 1e9
                stall_ms = (ticks["t_ns"][k + 1] - ticks["t_ns"][k]) / 1e6 if k + 1 < len(ticks) else float("nan")
                line += f"  STARTING {waiting_s:.2f} s  first RUNNING tick {stall_ms:.0f} ms"
            else:
                line += f"  then {STATES[state[k]] if k < len(ticks) else 'end'}"
            if inf is not None:
                after = np.nonzero(inf["t_obs_ns"] >= ticks["t_ns"][starting])[0]
                if len(after):
                    line += f"  next obs gripper {inf['state'][after[0], -1]:.2f}"
            print(line)
        k += 1

    if inf is not None and len(inf["t_obs_ns"]) > 1:
        jumps = []
        for i in range(1, len(inf["t_obs_ns"])):
            if inf["gripper_change_ns"][i - 1] >= 0:
                continue  # a gripper action sits between the two chunks
            dropped = int(inf["dropped_points"][i])
            offset = round((inf["t_obs_ns"][i] - inf["t_obs_ns"][i - 1]) * 1e-9 / PERIOD_S) + dropped
            previous, new = inf["chunk"][i - 1], inf["chunk"][i]
            if offset >= len(previous) or np.isnan(previous[offset, 0]) or dropped >= len(new):
                continue
            jumps.append(np.linalg.norm(new[dropped, :3] - previous[offset, :3]) * 1000)
        print(percentiles(np.array(jumps), "switch jump", "mm"))
        latency = inf["latency_s"][1:] * 1000
        print(percentiles(latency, "latency", "ms"))
        if "rtc_prefix_steps" in inf:
            guided = int((inf["rtc_prefix_steps"] > 0).sum())
            print(f"rtc guided: {guided}/{len(inf['t_obs_ns'])} inferences, "
                  f"delay steps {sorted(set(inf['rtc_delay_steps'].tolist()))}")

    commanded = ticks[ticks["cmd_valid"] == 1]
    if len(commanded) > 1:
        step = np.linalg.norm(np.diff(commanded["cmd"][:, :3], axis=0), axis=1)
        dt = np.diff(commanded["t_ns"]) / 1e9
        moving = dt < 0.05  # skip the servo pauses
        print(percentiles(step[moving] / dt[moving], "cmd speed", "mm/s"))
        error = np.linalg.norm(commanded["cmd"][:, :3] - commanded["tcp"][:, :3], axis=1)
        print(percentiles(error, "cmd - tcp", "mm"))
    tick_ms = np.diff(ticks["t_ns"]) / 1e6
    if len(tick_ms):
        print(f"tick: p99 {np.percentile(tick_ms, 99):.2f} ms max {tick_ms.max():.0f} ms")


if __name__ == "__main__":
    main()
