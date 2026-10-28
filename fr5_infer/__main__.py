"""python -m fr5_infer [--mode fake|shadow|real] [--backend pi05|replay] ..."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from aiohttp import web

from fr5_infer.app import InferApp
from fr5_infer.config import BACKENDS, MODES, InferConfig
from fr5_infer.web import make_web_app


log = logging.getLogger("fr5_infer")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="fr5_infer", description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--mode", choices=MODES, help="override config mode")
    parser.add_argument("--backend", choices=BACKENDS, help="override policy.backend")
    parser.add_argument("--checkpoint", type=Path, help="override policy.checkpoint")
    parser.add_argument("--episode", type=int, help="replay backend: dataset episode index")
    parser.add_argument("--task", help="override policy.task")
    parser.add_argument("--port", type=int, help="override web.port")
    parser.add_argument(
        "--confirm-hardware",
        action="store_true",
        help="required for --mode real: the robot will move",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    overrides = {
        key: value
        for key, value in {
            "mode": args.mode,
            "policy.backend": args.backend,
            "policy.checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
            "policy.replay.episode": args.episode,
            "policy.task": args.task,
            "web.port": args.port,
        }.items()
        if value is not None
    }
    config = InferConfig.from_yaml(args.config, overrides=overrides)
    logging.basicConfig(
        level=config.teleop.log_level,
        format="%(asctime)s %(levelname)s [main] %(name)s: %(message)s",
    )
    if config.mode == "real" and not args.confirm_hardware:
        sys.exit("--mode real moves the robot: add --confirm-hardware")
    if config.mode != "fake":
        log.warning(
            "connecting to the FR5 at %s: make sure the fr5_teleop container is stopped "
            "(only one program may command the robot)",
            config.teleop.robot_ip,
        )
    log.info(
        "mode=%s backend=%s limits: %.0f mm/s %.0f deg/s, orientation within %.0f deg of home",
        config.mode,
        config.policy.backend,
        config.max_velocity_mm_s,
        config.max_angular_velocity_deg_s,
        config.safety.max_orientation_deviation_deg,
    )

    infer = InferApp(config)
    try:
        infer.start()
        log.info("operator page: http://%s:%d/", config.web.host, config.web.port)
        web.run_app(
            make_web_app(infer),
            host=config.web.host,
            port=config.web.port,
            print=None,
        )
    finally:
        infer.close()


if __name__ == "__main__":
    main()
