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
# pipeline consumes the MediaPipe/MANO-style 21-point surface, so extra
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
# Manus SDK native raw skeleton order (thumb is LAST, not first):
#   0       Hand root (wrist)
#   1-5     Index  (CMC, MCP, PIP, DIP, TIP)     — 5 joints
#   6-10    Middle (CMC, MCP, PIP, DIP, TIP)     — 5 joints
#   11-15   Ring   (CMC, MCP, PIP, DIP, TIP)     — 5 joints
#   16-20   Pinky  (CMC, MCP, PIP, DIP, TIP)     — 5 joints
#   21-24   Thumb  (CMC, MCP, IP, TIP)           — 4 joints
# MANO expects: wrist, thumb(4), index(4), middle(4), ring(4), pinky(4) = 21.
# Also, we entirely skip the non-thumb CMC joints at indices 1, 6, 11, 16.
MANUS_TO_MANO = [0, 21, 22, 23, 24, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20]


def manus_zmq_to_mano_keypoints(positions: np.ndarray) -> np.ndarray:
    """Convert 21-joint ZMQ positions to wrist-relative keypoints.

    The native ManusClient C++ client publishes global-frame positions in a
    right-handed Z-up frame (X-from-viewer, Y-left, Z-up) with no additional
    transforms.  This function makes them wrist-relative and passes them
    straight through — the retargeter's normalization step
    (``get_normalized_local_manohand_joint_pos``) derives its own coordinate
    frame from hand geometry, so no axis remapping is needed.

    The previous Sharpa-based client applied its own coordinate transforms in
    C++, which were compensated here with a ``[Y, X, -Z]`` axis swap.  That
    swap is commented out now that the native client forwards raw SDK data.

    Args:
        positions: ``(21, 3)`` array already indexed to the 21 MANO joints.

    Returns:
        ``(21, 3)`` float32 array, wrist-relative, meters.
    """
    kp = np.array(positions, dtype=np.float32)
    kp = kp - kp[0:1, :]  # make wrist-relative

    return kp


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
