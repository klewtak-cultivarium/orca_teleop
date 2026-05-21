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
