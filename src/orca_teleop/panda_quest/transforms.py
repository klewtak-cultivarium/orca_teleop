from __future__ import annotations

from collections.abc import Sequence

import numpy as np

XR_TO_MUJOCO_BASIS = np.array(
    [
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


def xr_flat_matrix_to_numpy(raw_matrix: Sequence[float]) -> np.ndarray:
    matrix = np.asarray(raw_matrix, dtype=np.float64)
    if matrix.size != 16:
        raise ValueError(f"Expected 16 values for a pose matrix, received {matrix.size}.")
    return matrix.reshape(4, 4).T


def xr_matrix_to_mujoco_matrix(raw_matrix: Sequence[float]) -> np.ndarray:
    xr_matrix = xr_flat_matrix_to_numpy(raw_matrix)
    mujoco_matrix = np.eye(4, dtype=np.float64)
    mujoco_matrix[:3, :3] = XR_TO_MUJOCO_BASIS @ xr_matrix[:3, :3] @ XR_TO_MUJOCO_BASIS.T
    mujoco_matrix[:3, 3] = XR_TO_MUJOCO_BASIS @ xr_matrix[:3, 3]
    return mujoco_matrix


def xr_points_to_mujoco_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected XR points with shape (N, 3), got {points.shape}.")
    return points @ XR_TO_MUJOCO_BASIS.T


def _normalized(vector: np.ndarray, *, name: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        raise ValueError(f"Cannot build palm frame; {name} axis is degenerate.")
    return vector / norm


def webxr_palm_matrix_to_mujoco_matrix(
    landmarks: np.ndarray,
    *,
    wrist_translation: np.ndarray | None = None,
) -> np.ndarray:
    """Build a MuJoCo-frame palm pose from WebXR hand landmarks.

    WebXR wrist joint orientation is not a palm-plane convention. For arm IK we
    instead derive a frame from hand geometry: x is ring-to-index, y is
    wrist-to-middle, and z is the palm normal.
    """
    landmarks_mujoco = xr_points_to_mujoco_points(np.asarray(landmarks, dtype=np.float64))
    if landmarks_mujoco.shape != (25, 3):
        raise ValueError(
            f"Expected WebXR hand landmarks with shape (25, 3), got {landmarks_mujoco.shape}."
        )

    wrist = landmarks_mujoco[0]
    index_base = landmarks_mujoco[5]
    middle_base = landmarks_mujoco[10]
    ring_base = landmarks_mujoco[15]

    y_axis = _normalized(middle_base - wrist, name="wrist-to-middle")
    x_axis = index_base - ring_base
    x_axis = _normalized(x_axis - np.dot(x_axis, y_axis) * y_axis, name="ring-to-index")
    z_axis = _normalized(np.cross(x_axis, y_axis), name="palm-normal")
    y_axis = _normalized(np.cross(z_axis, x_axis), name="orthogonalized wrist-to-middle")

    translation = (
        wrist if wrist_translation is None else np.asarray(wrist_translation, dtype=np.float64)
    )
    return make_transform(
        np.column_stack([x_axis, y_axis, z_axis]),
        translation,
    )


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def rotation_vector_from_matrix(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    cos_angle = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    angle = float(np.arccos(cos_angle))
    if angle < 1e-8:
        return np.zeros(3, dtype=np.float64)

    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    )
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-8:
        return np.zeros(3, dtype=np.float64)
    return axis * (angle / axis_norm)
