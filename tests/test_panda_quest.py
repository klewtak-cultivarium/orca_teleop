import numpy as np
import pytest

from orca_teleop.panda_quest.dataset_replay import (
    RETARGETER_HAND_LANDMARK_NAMES,
    WEBXR_TO_RETARGETER_LANDMARK_INDICES,
    retargeter_landmarks_from_webxr,
)
from orca_teleop.panda_quest.transforms import (
    XR_TO_MUJOCO_BASIS,
    webxr_palm_matrix_to_mujoco_matrix,
)


def _indexed_webxr_landmarks() -> np.ndarray:
    indices = np.arange(25, dtype=np.float64)
    return np.column_stack([indices, indices + 100.0, -indices])


def _webxr_landmarks_from_mujoco_points(points: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) @ XR_TO_MUJOCO_BASIS


def test_webxr_landmarks_reduce_to_retargeter_layout_order():
    landmarks = retargeter_landmarks_from_webxr(_indexed_webxr_landmarks(), "left")

    assert landmarks.shape == (21, 3)
    assert len(RETARGETER_HAND_LANDMARK_NAMES) == 21
    assert landmarks[:, 0].tolist() == list(WEBXR_TO_RETARGETER_LANDMARK_INDICES)


def test_webxr_landmarks_preserve_tips_after_dropping_metacarpals():
    landmarks = retargeter_landmarks_from_webxr(_indexed_webxr_landmarks(), "left")
    by_name = dict(zip(RETARGETER_HAND_LANDMARK_NAMES, landmarks[:, 0], strict=True))

    assert by_name["thumb_tip"] == 4
    assert by_name["index_tip"] == 9
    assert by_name["middle_tip"] == 14
    assert by_name["ring_tip"] == 19
    assert by_name["pinky_tip"] == 24


def test_webxr_landmarks_map_mcp_to_phalanx_proximal():
    """The MCP knuckle for each non-thumb finger must come from WebXR's
    `*-phalanx-proximal` joint (idx 6/11/16/21 in the 25-joint layout), not
    the `*-metacarpal` joint (idx 5/10/15/20). The latter is, per the WebXR
    Hand Input spec and confirmed empirically on Quest, the wrist-side end
    of the metacarpal bone — using it as the finger base collapses the
    retargeter's palm position onto the wrist."""
    landmarks = retargeter_landmarks_from_webxr(_indexed_webxr_landmarks(), "left")
    by_name = dict(zip(RETARGETER_HAND_LANDMARK_NAMES, landmarks[:, 0], strict=True))

    assert by_name["index_mcp"] == 6
    assert by_name["middle_mcp"] == 11
    assert by_name["ring_mcp"] == 16
    assert by_name["pinky_mcp"] == 21


def test_webxr_landmarks_map_pip_dip_to_phalanx_intermediate_distal():
    landmarks = retargeter_landmarks_from_webxr(_indexed_webxr_landmarks(), "left")
    by_name = dict(zip(RETARGETER_HAND_LANDMARK_NAMES, landmarks[:, 0], strict=True))

    assert by_name["index_pip"] == 7
    assert by_name["index_dip"] == 8
    assert by_name["middle_pip"] == 12
    assert by_name["middle_dip"] == 13
    assert by_name["ring_pip"] == 17
    assert by_name["ring_dip"] == 18
    assert by_name["pinky_pip"] == 22
    assert by_name["pinky_dip"] == 23


def test_webxr_landmarks_preserve_right_hand_chirality():
    """WebXR already delivers right-handed data, so the right-hand mapping
    must not mirror any axis — that would flip chirality and break the
    right-OrcaHand IK target."""
    source = _indexed_webxr_landmarks()
    landmarks = retargeter_landmarks_from_webxr(source, "right")
    expected = source[list(WEBXR_TO_RETARGETER_LANDMARK_INDICES)].copy()

    np.testing.assert_allclose(landmarks, expected)


def test_webxr_landmarks_match_for_both_sides():
    source = _indexed_webxr_landmarks()
    left = retargeter_landmarks_from_webxr(source, "left")
    right = retargeter_landmarks_from_webxr(source, "right")

    np.testing.assert_allclose(left, right)


def test_webxr_landmarks_reject_wrong_shape():
    with pytest.raises(ValueError, match="shape"):
        retargeter_landmarks_from_webxr(np.zeros((21, 3)), "left")


def test_webxr_landmarks_reject_unknown_side():
    with pytest.raises(ValueError, match="Unsupported side"):
        retargeter_landmarks_from_webxr(np.zeros((25, 3)), "center")


def test_webxr_palm_matrix_uses_landmark_palm_axes():
    mujoco_landmarks = np.zeros((25, 3), dtype=np.float64)
    mujoco_landmarks[0] = [1.0, 2.0, 3.0]
    mujoco_landmarks[5] = [2.0, 2.0, 3.0]
    mujoco_landmarks[10] = [1.0, 3.0, 3.0]
    mujoco_landmarks[15] = [0.0, 2.0, 3.0]

    matrix = webxr_palm_matrix_to_mujoco_matrix(
        _webxr_landmarks_from_mujoco_points(mujoco_landmarks),
        wrist_translation=np.array([9.0, 8.0, 7.0]),
    )

    np.testing.assert_allclose(matrix[:3, :3], np.eye(3), atol=1e-8)
    np.testing.assert_allclose(matrix[:3, 3], [9.0, 8.0, 7.0])


def test_webxr_palm_matrix_defaults_to_landmark_wrist_translation():
    mujoco_landmarks = np.zeros((25, 3), dtype=np.float64)
    mujoco_landmarks[0] = [1.0, 2.0, 3.0]
    mujoco_landmarks[5] = [2.0, 2.0, 3.0]
    mujoco_landmarks[10] = [1.0, 3.0, 3.0]
    mujoco_landmarks[15] = [0.0, 2.0, 3.0]

    matrix = webxr_palm_matrix_to_mujoco_matrix(
        _webxr_landmarks_from_mujoco_points(mujoco_landmarks)
    )

    np.testing.assert_allclose(matrix[:3, 3], mujoco_landmarks[0])
