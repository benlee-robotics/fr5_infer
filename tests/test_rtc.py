"""RTC config checks and the continuation the pi05 backend hands lerobot."""

from pathlib import Path

import pytest

from fr5_infer.config import InferConfig, RtcConfig


CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"


@pytest.fixture
def conversion(tmp_path):
    path = tmp_path / "fr5_conversion.json"
    path.write_text(
        '{"options": {"fps": 30, "rotation": "rot6d", "action_mode": "command"},'
        ' "cameras": ["top", "wrist"]}'
    )
    return path


def load(conversion, **overrides):
    return InferConfig.from_yaml(
        CONFIG, overrides={"mode": "fake", "policy.conversion": str(conversion), **overrides}
    )


def test_rtc_and_ensemble_are_read_from_the_config(conversion):
    config = load(
        conversion,
        **{
            "policy.rtc.enabled": True,
            "policy.rtc.execution_horizon": 15,
            "policy.rtc.schedule": "EXP",
            "execution.ensemble": True,
        },
    )
    assert config.policy.rtc == RtcConfig(True, 15, 10.0, "EXP")
    assert config.execution.ensemble


def test_rtc_is_rejected_for_the_replay_backend(conversion):
    with pytest.raises(ValueError, match="pi05"):
        load(conversion, **{"policy.backend": "replay", "policy.rtc.enabled": True})


def test_unknown_rtc_schedule_is_rejected(conversion):
    with pytest.raises(ValueError, match="schedule"):
        load(conversion, **{"policy.rtc.schedule": "COSINE"})


def backend_with_previous(t_obs_ns, steps=50):
    torch = pytest.importorskip("torch")
    from fr5_infer.backends.pi05 import Pi05Backend

    # Only the continuation bookkeeping, no model.
    backend = object.__new__(Pi05Backend)
    backend.fps = 30
    backend.rtc = RtcConfig(enabled=True, execution_horizon=10)
    backend._previous = (t_obs_ns, torch.arange(steps, dtype=torch.float32)[:, None].repeat(1, 10))
    return backend


def test_rtc_prefix_starts_at_the_new_observation_time():
    backend = backend_with_previous(0)
    # 20 steps (0.667 s) later, 130 ms inference + 20 ms lead.
    inputs = backend._rtc_inputs(20 * 33_333_333, 0.15)
    leftover = inputs["prev_chunk_left_over"]
    assert leftover.shape == (1, 30, 10)
    assert float(leftover[0, 0, 0]) == 20.0
    assert inputs["inference_delay"] == 5  # ceil(0.15 * 30)
    assert inputs["execution_horizon"] == 10
    assert backend.last_rtc == {"prefix_steps": 30, "delay_steps": 5}


def test_rtc_without_overlap_or_after_a_break_is_unguided():
    backend = backend_with_previous(0)
    assert backend._rtc_inputs(2 * 10**9, 0.15) == {}  # 60 steps later
    assert backend.last_rtc == {"prefix_steps": 0, "delay_steps": 0}
    backend = backend_with_previous(0)
    backend.break_continuation()
    assert backend._rtc_inputs(10**8, 0.15) == {}
