"""Standalone Quest telemetry ingress probe.

Run this before starting MuJoCo teleop to confirm the Quest Browser client is
sending controller and WebXR hand-tracking payloads.
"""

from __future__ import annotations

import argparse
import json
import logging
import ssl
import time
from collections.abc import Sequence

import numpy as np

from orca_teleop.panda_quest.quest_bridge import HAND_SIDES, QuestTelemetryBridge

logger = logging.getLogger("quest_telemetry_probe")


def _ssl_context(certfile: str | None, keyfile: str | None) -> ssl.SSLContext | None:
    if certfile is None and keyfile is None:
        return None
    if certfile is None or keyfile is None:
        raise ValueError("--ssl-cert and --ssl-key must be passed together.")
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.load_cert_chain(certfile=certfile, keyfile=keyfile)
    return context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0", help="Quest bridge bind host.")
    parser.add_argument("--port", type=int, default=8765, help="Quest bridge HTTP port.")
    parser.add_argument("--ssl-cert", default=None, help="Optional HTTPS certificate for WebXR.")
    parser.add_argument("--ssl-key", default=None, help="Optional HTTPS key for WebXR.")
    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Seconds between parsed telemetry summaries.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Also print the latest raw telemetry JSON payload.",
    )
    parser.add_argument(
        "--raw-every",
        type=int,
        default=1,
        help="When --raw is set, print raw JSON every N summaries.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=3,
        help="Decimal precision for printed numeric arrays.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser.parse_args()


def _fmt_array(values: Sequence[float] | np.ndarray, precision: int) -> str:
    return np.array2string(np.asarray(values), precision=precision, suppress_small=True)


def _pose_summary(matrix: np.ndarray | None, precision: int) -> str:
    if matrix is None:
        return "missing"
    xyz = matrix[:3, 3]
    forward = matrix[:3, 0]
    return f"xyz={_fmt_array(xyz, precision)} x_axis={_fmt_array(forward, precision)}"


def _landmark_summary(landmarks: np.ndarray | None, precision: int) -> str:
    if landmarks is None:
        return "missing"
    sample_indices = {
        "wrist": 0,
        "thumb_tip": 4,
        "index_tip": 9,
        "middle_tip": 14,
        "ring_tip": 19,
        "pinky_tip": 24,
    }
    samples = " ".join(
        f"{name}={_fmt_array(landmarks[index], precision)}"
        for name, index in sample_indices.items()
    )
    return f"shape={landmarks.shape} {samples}"


def _print_summary(bridge: QuestTelemetryBridge, *, precision: int, sequence: int) -> None:
    now = time.monotonic()
    telemetry_age = now - bridge.state.last_update_monotonic
    print(f"\n[{sequence:05d}] telemetry_age={telemetry_age:.3f}s", flush=True)
    print(f"  head: {_pose_summary(bridge.state.get_head_matrix(), precision)}", flush=True)

    for side in HAND_SIDES:
        controller_matrix = bridge.state.get_controller_matrix(side)
        axes = bridge.state.get_controller_axes(side)
        buttons = bridge.state.get_controller_buttons(side)
        hand_wrist = bridge.state.get_hand_wrist_matrix(side)
        hand_landmarks = bridge.state.get_hand_landmarks(side)

        print(
            f"  {side} controller: {_pose_summary(controller_matrix, precision)} "
            f"axes={_fmt_array(axes, precision)} buttons={_fmt_array(buttons, precision)}",
            flush=True,
        )
        print(f"  {side} hand wrist: {_pose_summary(hand_wrist, precision)}", flush=True)
        print(
            f"  {side} hand landmarks: {_landmark_summary(hand_landmarks, precision)}",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    bridge = QuestTelemetryBridge(
        host=args.host,
        port=args.port,
        ssl_context=_ssl_context(args.ssl_cert, args.ssl_key),
    )
    bridge.start()
    logger.info("Quest telemetry probe listening at %s", bridge.url)
    if bridge.ssl_context is None:
        logger.info(
            "Quest Browser needs a secure context. Use an HTTPS tunnel, for example: "
            "ngrok http %d",
            args.port,
        )
    logger.info("Open the page in Quest Browser, tap Start, then show your hands.")

    sequence = 0
    try:
        logger.info("Waiting for the first Quest telemetry packet before printing summaries...")
        while not bridge.wait_until_connected(timeout=0.5):
            pass
        logger.info("Quest telemetry connected; printing summaries every %.2fs.", args.interval)
        while True:
            sequence += 1
            _print_summary(bridge, precision=args.precision, sequence=sequence)
            if args.raw and sequence % max(1, args.raw_every) == 0:
                payload = bridge.get_last_telemetry_payload()
                print("  raw telemetry:", flush=True)
                print(json.dumps(payload, indent=2), flush=True)
            time.sleep(max(0.05, args.interval))
    except KeyboardInterrupt:
        logger.info("Stopping Quest telemetry probe.")
    finally:
        bridge.stop()


if __name__ == "__main__":
    main()
