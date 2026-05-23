"""Live 3D hand-landmark visualizer.

Runs in a separate subprocess (via ``multiprocessing.Process``) to avoid
matplotlib threading issues.  The retargeter worker pushes ``(21, 3)``
keypoints into a ``multiprocessing.Queue``; the visualizer process reads
and animates them with ``FuncAnimation``.

Usage::

    viz = HandLandmarkVisualizer()
    viz.start()
    viz.put(keypoints, "right")  # called from retargeter thread
    ...
    viz.stop()
"""

from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np

# MANO 21-keypoint finger chains (each starts at wrist=0).
_FINGER_CHAINS: dict[str, list[int]] = {
    "thumb": [0, 1, 2, 3, 4],
    "index": [0, 5, 6, 7, 8],
    "middle": [0, 9, 10, 11, 12],
    "ring": [0, 13, 14, 15, 16],
    "pinky": [0, 17, 18, 19, 20],
}

# 20 bone connections derived from the chains above.
MANO_CONNECTIONS: list[tuple[int, int]] = []
for _chain in _FINGER_CHAINS.values():
    for _i in range(len(_chain) - 1):
        MANO_CONNECTIONS.append((_chain[_i], _chain[_i + 1]))

FINGER_COLORS: dict[str, str] = {
    "thumb": "red",
    "index": "green",
    "middle": "blue",
    "ring": "orange",
    "pinky": "purple",
}


@dataclass(frozen=True, slots=True)
class _VizFrame:
    keypoints: np.ndarray  # (21, 3)
    handedness: str


def _run_visualizer(q: mp.Queue) -> None:  # noqa: C901
    """Subprocess entry point: read keypoints from *q* and animate a 3D plot."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d.art3d import Line3D

    fig = plt.figure("Hand Landmarks", figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # One Line3D per finger (bones + joint dots via 'o-' style).
    finger_lines: dict[str, Line3D] = {}
    for name, color in FINGER_COLORS.items():
        (line,) = ax.plot([], [], [], "o-", color=color, markersize=4, linewidth=2, label=name)
        finger_lines[name] = line

    # Wrist marker (square, black).
    (wrist_marker,) = ax.plot([], [], [], "s", color="black", markersize=6)

    ax.legend(loc="upper left", fontsize="small")

    frame_count = 0
    t_start = time.perf_counter()
    last_frame: _VizFrame | None = None

    def _update(_frame_number: int) -> list:
        nonlocal last_frame, frame_count, t_start

        # Drain queue — keep only the newest frame.
        newest: _VizFrame | None = None
        while True:
            try:
                newest = q.get_nowait()
            except Exception:
                break
        if newest is not None:
            last_frame = newest
            frame_count += 1

        if last_frame is None:
            return []

        kp = last_frame.keypoints  # (21, 3)

        # Update per-finger lines.
        for name, chain in _FINGER_CHAINS.items():
            xs = kp[chain, 0]
            ys = kp[chain, 1]
            zs = kp[chain, 2]
            finger_lines[name].set_data_3d(xs, ys, zs)

        # Wrist marker.
        wrist_marker.set_data_3d([kp[0, 0]], [kp[0, 1]], [kp[0, 2]])

        # Auto-scale axes to hand bounding box with equal aspect.
        mins = kp.min(axis=0)
        maxs = kp.max(axis=0)
        center = (mins + maxs) / 2.0
        half_range = max((maxs - mins).max() / 2.0, 0.01) * 1.2
        ax.set_xlim(center[0] - half_range, center[0] + half_range)
        ax.set_ylim(center[1] - half_range, center[1] + half_range)
        ax.set_zlim(center[2] - half_range, center[2] + half_range)

        # Title with handedness, frame count, fps.
        elapsed = time.perf_counter() - t_start
        fps = frame_count / elapsed if elapsed > 0 else 0.0
        ax.set_title(f"{last_frame.handedness.upper()} hand | frame {frame_count} | {fps:.1f} fps")

        return list(finger_lines.values()) + [wrist_marker]

    _anim = FuncAnimation(fig, _update, interval=33, blit=False, cache_frame_data=False)  # noqa: F841
    plt.show()


class HandLandmarkVisualizer:
    """Push ``(21, 3)`` keypoints to a daemon subprocess that renders them live."""

    def __init__(self, maxsize: int = 4) -> None:
        self._q: mp.Queue = mp.Queue(maxsize=maxsize)
        self._process: mp.Process | None = None

    def start(self) -> None:
        self._process = mp.Process(
            target=_run_visualizer,
            args=(self._q,),
            name="hand-landmark-viz",
            daemon=True,
        )
        self._process.start()

    def put(self, keypoints: np.ndarray, handedness: Literal["left", "right"]) -> None:
        frame = _VizFrame(keypoints=np.asarray(keypoints, dtype=np.float64), handedness=handedness)
        # Non-blocking push; drop oldest on full.
        try:
            self._q.put_nowait(frame)
        except Exception:
            try:
                self._q.get_nowait()
            except Exception:
                pass
            try:
                self._q.put_nowait(frame)
            except Exception:
                pass

    def stop(self) -> None:
        if self._process is not None and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=3.0)
        self._process = None
