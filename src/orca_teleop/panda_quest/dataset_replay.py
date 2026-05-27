from __future__ import annotations

import itertools
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from orca_teleop.panda_quest.transforms import make_transform

UNITY_LEFT_TO_FLU = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)
WEBXR_HAND_JOINT_NAMES = (
    "wrist",
    "thumb-metacarpal",
    "thumb-phalanx-proximal",
    "thumb-phalanx-distal",
    "thumb-tip",
    "index-finger-metacarpal",
    "index-finger-phalanx-proximal",
    "index-finger-phalanx-intermediate",
    "index-finger-phalanx-distal",
    "index-finger-tip",
    "middle-finger-metacarpal",
    "middle-finger-phalanx-proximal",
    "middle-finger-phalanx-intermediate",
    "middle-finger-phalanx-distal",
    "middle-finger-tip",
    "ring-finger-metacarpal",
    "ring-finger-phalanx-proximal",
    "ring-finger-phalanx-intermediate",
    "ring-finger-phalanx-distal",
    "ring-finger-tip",
    "pinky-finger-metacarpal",
    "pinky-finger-phalanx-proximal",
    "pinky-finger-phalanx-intermediate",
    "pinky-finger-phalanx-distal",
    "pinky-finger-tip",
)
RETARGETER_HAND_LANDMARK_NAMES = (
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
)
# WebXR exposes 25 joints per hand; the retargeter wants the 21-joint
# MediaPipe/MANO layout. For the thumb, WebXR's four post-wrist joints
# (thumb-metacarpal=CMC, phalanx-proximal=MCP, phalanx-distal=IP, tip) line
# up one-for-one with MediaPipe's (thumb_cmc, thumb_mcp, thumb_ip, thumb_tip).
# For the other four fingers, WebXR ships an extra `*-finger-metacarpal`
# joint that the WebXR Hand Input spec places at the *wrist-side end* of the
# metacarpal bone — i.e. at the carpometacarpal junction, not at the MCP
# knuckle (verified on Quest with `scripts/quest_telemetry_probe.py`:
# wrist→metacarpal ≈ 4 cm, wrist→phalanx_proximal ≈ 9 cm). MediaPipe's
# `{finger}_mcp` is the MCP knuckle, which is WebXR's `phalanx-proximal`, so
# we *drop the metacarpal* (not the phalanx-distal!) and keep
# (phalanx-proximal, phalanx-intermediate, phalanx-distal, tip).
WEBXR_TO_RETARGETER_LANDMARK_INDICES = (
    0,  # wrist
    1,
    2,
    3,
    4,  # thumb: WebXR thumb-metacarpal=CMC, phalanx-proximal=MCP, phalanx-distal=IP, tip
    6,
    7,
    8,
    9,  # index: drop WebXR metacarpal; phalanx-proximal=MCP, intermediate=PIP, distal=DIP, tip
    11,
    12,
    13,
    14,  # middle
    16,
    17,
    18,
    19,  # ring
    21,
    22,
    23,
    24,  # pinky
)


@dataclass(frozen=True)
class WristPoseSample:
    timestamp_ns: int
    side: str
    matrix: np.ndarray
    landmarks: np.ndarray | None = None


def quat_xyzw_to_rotmat(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    norm = qx * qx + qy * qy + qz * qz + qw * qw
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    scale = 2.0 / norm
    xx, yy, zz = qx * qx * scale, qy * qy * scale, qz * qz * scale
    xy, xz, yz = qx * qy * scale, qx * qz * scale, qy * qz * scale
    wx, wy, wz = qw * qx * scale, qw * qy * scale, qw * qz * scale
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )


def unity_wrist_to_mujoco_matrix(row: dict, side: str) -> np.ndarray:
    p_unity = np.array(
        [
            row[f"{side}_wrist_x"],
            row[f"{side}_wrist_y"],
            row[f"{side}_wrist_z"],
        ],
        dtype=np.float64,
    )
    r_unity = quat_xyzw_to_rotmat(
        row[f"{side}_wrist_qx"],
        row[f"{side}_wrist_qy"],
        row[f"{side}_wrist_qz"],
        row[f"{side}_wrist_qw"],
    )
    p_flu = UNITY_LEFT_TO_FLU @ p_unity
    r_flu = UNITY_LEFT_TO_FLU @ r_unity @ UNITY_LEFT_TO_FLU.T
    return make_transform(r_flu, p_flu)


def load_hf_wrist_pose_samples(
    repo_id: str,
    *,
    filename: str = "data.parquet",
    side: str = "right",
    refresh: bool = False,
) -> list[WristPoseSample]:
    if side not in ("left", "right"):
        raise ValueError(f"Unsupported side: {side!r}")

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=filename,
        force_download=refresh,
    )
    rows = pq.read_table(path).to_pylist()
    samples = [
        WristPoseSample(
            timestamp_ns=int(row["t_ns"]),
            side=side,
            matrix=unity_wrist_to_mujoco_matrix(row, side),
            landmarks=_landmarks_from_row(row, side),
        )
        for row in rows
        if row.get(f"{side}_visible")
    ]
    if not samples:
        raise RuntimeError(f"No visible {side} wrist poses found in {repo_id}/{filename}.")
    return samples


def _landmarks_from_row(row: dict, side: str) -> np.ndarray | None:
    raw_landmarks = row.get(f"{side}_landmarks")
    if raw_landmarks is None:
        return None
    landmarks = np.asarray(raw_landmarks, dtype=np.float64)
    if landmarks.size == 0:
        return None
    if landmarks.size % 3 != 0:
        raise ValueError(
            f"Expected {side}_landmarks to contain a multiple of 3 values, got {landmarks.size}."
        )
    return landmarks.reshape(-1, 3)


def retargeter_landmarks_from_quest(points: np.ndarray, side: str) -> np.ndarray:
    """Adapt recorded Quest landmarks to the retargeter's 21-point convention."""
    if side not in ("left", "right"):
        raise ValueError(f"Unsupported side: {side!r}")
    landmarks = np.asarray(points, dtype=np.float64).copy()
    if landmarks.ndim != 2 or landmarks.shape != (21, 3):
        raise ValueError(f"Expected Quest landmarks with shape (21, 3), got {landmarks.shape}.")
    if side == "right":
        landmarks[:, 1] *= -1.0
    return landmarks


def retargeter_landmarks_from_webxr(points: np.ndarray, side: str) -> np.ndarray:
    """Adapt live WebXR hand joints to the retargeter's 21-point convention.

    WebXR exposes 25 joints: wrist plus four thumb joints and five joints for
    each non-thumb finger, ordered by ``WEBXR_HAND_JOINT_NAMES``. The retargeter
    expects the 21-point MediaPipe/MANO-like layout in
    ``RETARGETER_HAND_LANDMARK_NAMES`` order. For each non-thumb finger we
    drop the WebXR ``*-finger-metacarpal`` joint (which the spec places at the
    *wrist-side* end of the metacarpal bone, not at the MCP knuckle) and keep
    phalanx-proximal/intermediate/distal/tip — those map one-for-one to
    MediaPipe's mcp/pip/dip/tip. The thumb already has only four post-wrist
    joints, which line up with MediaPipe's cmc/mcp/ip/tip directly.

    WebXR already reports hand joints in a right-handed reference space, so no
    chirality flip is needed here — the retargeter's canonical hand frame
    (``z = cross(x, y)``, "out of the palm for a right hand") matches the
    incoming right-hand geometry directly. Mirroring an axis would flip the
    right hand into a left-hand shape and break IK against the right OrcaHand
    URDF. The recorded HF replays go through ``retargeter_landmarks_from_quest``
    instead, which still mirrors because that data is stored in Unity's
    left-handed frame.
    """
    if side not in ("left", "right"):
        raise ValueError(f"Unsupported side: {side!r}")
    landmarks_25 = np.asarray(points, dtype=np.float64)
    if landmarks_25.ndim != 2 or landmarks_25.shape != (25, 3):
        raise ValueError(f"Expected WebXR landmarks with shape (25, 3), got {landmarks_25.shape}.")

    return landmarks_25[list(WEBXR_TO_RETARGETER_LANDMARK_INDICES)].copy()


def iter_wrist_pose_samples(
    samples: list[WristPoseSample],
    *,
    loop: bool,
) -> Iterator[WristPoseSample]:
    if loop:
        yield from itertools.cycle(samples)
    else:
        yield from samples
