"""Manus glove coordinate conversions for the teleop pipeline.

All Manus-specific coordinate handling lives here so the rest of the pipeline
(retargeter, sim) stays source-agnostic.
"""

from __future__ import annotations

import numpy as np

MANO_LANDMARK_NAMES: tuple[str, ...] = (
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

# Manus has one additional non-thumb CMC position per finger. The teleop
# pipeline consumes the MediaPipe/MANO-style 21-point surface, so those extra
# CMCs are intentionally skipped.
MANUS_POSITION_JOINTS_FOR_MANO: tuple[str, ...] = (
    "Hand",
    "Thumb_CMC",
    "Thumb_MCP",
    "Thumb_IP",
    "Thumb_TIP",
    "Index_MCP",
    "Index_PIP",
    "Index_DIP",
    "Index_TIP",
    "Middle_MCP",
    "Middle_PIP",
    "Middle_DIP",
    "Middle_TIP",
    "Ring_MCP",
    "Ring_PIP",
    "Ring_DIP",
    "Ring_TIP",
    "Pinky_MCP",
    "Pinky_PIP",
    "Pinky_DIP",
    "Pinky_TIP",
)

# Manus 25-joint → MANO 21-joint index mapping.
# Manus SDK native joint order (verified against the SDK visualizer):
#   0       Hand root (wrist)
#   1-4     Thumb  (CMC, MCP, IP, TIP)           — 4 joints
#   5-9     Index  (CMC, MCP, PIP, DIP, TIP)     — 5 joints
#   10-14   Middle (CMC, MCP, PIP, DIP, TIP)     — 5 joints
#   15-19   Ring   (CMC, MCP, PIP, DIP, TIP)     — 5 joints
#   20-24   Pinky  (CMC, MCP, PIP, DIP, TIP)     — 5 joints
# MANO expects: wrist, thumb(4), index(4), middle(4), ring(4), pinky(4) = 21.
# We skip the non-thumb CMC joints at indices 5, 10, 15, 20.
MANUS_TO_MANO = [0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24]


def manus_zmq_to_mano_keypoints(positions: np.ndarray) -> np.ndarray:
    """Convert 21-joint ZMQ positions to wrist-relative MANO convention.

    The C++ client publishes global-frame positions in a right-handed Z-up
    frame (X-from-viewer, Y-left, Z-up).  This function makes them
    wrist-relative and converts to the same convention used by
    ``manus_unity_positions_to_mano_keypoints``.

    The SDK and Unity coordinate systems relate as:
        X_unity = -Y_sdk,  Y_unity = Z_sdk,  Z_unity = X_sdk

    Applying the Unity MANO formula ``(-X_u, Z_u, -Y_u)`` in SDK terms gives:
        MANO = (Y_sdk, X_sdk, -Z_sdk)

    Args:
        positions: ``(21, 3)`` array already indexed to the 21 MANO joints.

    Returns:
        ``(21, 3)`` float32 array in MANO coordinates, wrist-relative, meters.
    """
    kp = np.array(positions, dtype=np.float32)
    kp = kp - kp[0:1, :]  # make wrist-relative
    return np.stack([kp[:, 1], kp[:, 0], -kp[:, 2]], axis=-1)


def manus_unity_positions_to_mano_keypoints(positions_cm: np.ndarray) -> np.ndarray:
    """Convert selected Manus Unity-style positions to teleop hand landmarks.

    Args:
        positions_cm: ``(21, 3)`` or ``(N, 21, 3)`` array in centimeters,
            ordered as ``MANUS_POSITION_JOINTS_FOR_MANO``.

    Returns:
        ``(21, 3)`` or ``(N, 21, 3)`` float32 array in meters, wrist-relative,
        ordered as ``MANO_LANDMARK_NAMES``. The coordinates are ready to be
        flattened row-major into the teleop gRPC ``HandFrame.keypoints`` field.

    Coordinate contract:
        keypoints_m = [-X_cm, Z_cm, -Y_cm] / 100
        after subtracting ``Hand_Position`` from every selected joint.
    """
    positions = np.asarray(positions_cm, dtype=np.float32)
    single_frame = positions.ndim == 2
    if single_frame:
        positions = positions[None, ...]

    if positions.ndim != 3 or positions.shape[1:] != (21, 3):
        raise ValueError(
            f"positions_cm must have shape (21, 3) or (N, 21, 3); got {positions.shape}"
        )

    relative_cm = positions - positions[:, [0], :]
    keypoints = np.empty_like(relative_cm, dtype=np.float32)
    keypoints[..., 0] = -relative_cm[..., 0] / 100.0
    keypoints[..., 1] = relative_cm[..., 2] / 100.0
    keypoints[..., 2] = -relative_cm[..., 1] / 100.0
    return keypoints[0] if single_frame else keypoints
