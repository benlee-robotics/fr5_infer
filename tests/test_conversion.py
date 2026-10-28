from pathlib import Path

import numpy as np
import pytest

from fr5_infer.conversion import (
    decode_chunk,
    encode_state,
    load_conversion,
    normalize_gripper,
    pose_to_tcp,
    rot6d_to_matrix,
)
from teleop.recording.lerobot_convert import euler_deg_to_matrix


TELEOP = Path(__file__).resolve().parents[2] / "fr5_teleop"
DATASET = TELEOP / "datasets" / "fr5_pick_cube_20261027"
HOME_TCP = (393.416, -122.637, 503.735, -179.441, 1.038, -3.243)
# FR5_PI05_DEPLOY_GUIDE.md section 2-2
GUIDE_HOME_XYZ_M = [0.393416, -0.122636, 0.503736]
GUIDE_HOME_ROT6D = [0.998235, -0.056555, -0.018110, -0.056738, -0.998342, -0.009754]


def test_home_state_matches_the_deploy_guide():
    state = encode_state(HOME_TCP, 0, 0, 100)
    np.testing.assert_allclose(state[:3], GUIDE_HOME_XYZ_M, atol=2e-6)
    np.testing.assert_allclose(state[3:9], GUIDE_HOME_ROT6D, atol=2e-5)
    assert state[9] == 0.0


@pytest.mark.skipif(not DATASET.is_dir(), reason="dataset not available")
def test_home_state_matches_the_dataset_first_frame():
    import pandas as pd

    frames = pd.read_parquet(
        DATASET / "data" / "chunk-000" / "file-000.parquet",
        columns=["observation.state", "episode_index", "frame_index"],
    )
    first = frames[(frames.episode_index == 0) & (frames.frame_index == 0)]
    recorded = first["observation.state"].iloc[0]
    state = encode_state(HOME_TCP, 1, 0, 100)
    np.testing.assert_allclose(state, recorded, atol=1e-4)


def test_gripper_normalization_matches_the_converter():
    assert normalize_gripper(0, 0, 100) == 0.0
    assert normalize_gripper(58, 0, 100) == pytest.approx(0.58)
    assert normalize_gripper(120, 0, 100) == 1.0
    assert normalize_gripper(-5, 0, 100) == 0.0
    # Inverted ranges work like (value - open) / (closed - open).
    assert normalize_gripper(25, 100, 0) == pytest.approx(0.75)


@pytest.mark.parametrize(
    "euler",
    [HOME_TCP[3:], (179.9, -0.5, 90.0), (-170.0, 20.0, -45.0), (10.0, 30.0, 160.0)],
)
def test_decode_round_trip(euler):
    tcp = (400.0, -100.0, 300.0, *euler)
    state = encode_state(tcp, 50, 0, 100)
    decoded = decode_chunk(np.stack([state, state]))
    back = pose_to_tcp(decoded.position_mm[0], decoded.quat_xyzw[0])
    np.testing.assert_allclose(back[:3], tcp[:3], atol=1e-3)
    np.testing.assert_allclose(
        euler_deg_to_matrix(np.asarray([back[3:]])),
        euler_deg_to_matrix(np.asarray([tcp[3:]])),
        atol=1e-6,
    )
    assert decoded.gripper[0] == pytest.approx(0.5)


def test_rot6d_gram_schmidt_repairs_non_orthonormal_columns():
    d6 = np.array([1.1, 0.05, 0.0, 0.1, 0.9, 0.02])
    matrix = rot6d_to_matrix(d6)
    np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-9)
    assert np.linalg.det(matrix) == pytest.approx(1.0)


def test_decode_rejects_bad_chunks():
    with pytest.raises(ValueError):
        decode_chunk(np.zeros((5, 9)))
    chunk = np.zeros((5, 10))
    chunk[2, 0] = np.nan
    with pytest.raises(ValueError):
        decode_chunk(chunk)


@pytest.mark.skipif(not DATASET.is_dir(), reason="dataset not available")
def test_load_conversion():
    info = load_conversion(DATASET / "fr5_conversion.json")
    assert info.fps == 30
    assert set(info.cameras) == {"top", "wrist"}
    assert info.image_shapes["top"] == (480, 640)
