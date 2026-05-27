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

from orca_teleop.panda_quest.dataset_replay import (
    RETARGETER_HAND_LANDMARK_NAMES,
    WEBXR_HAND_JOINT_NAMES,
    retargeter_landmarks_from_webxr,
)
from orca_teleop.panda_quest.quest_bridge import HAND_SIDES, QuestTelemetryBridge

logger = logging.getLogger("quest_telemetry_probe")

# Per the WebXR Hand Input spec, `*-finger-metacarpal` sits at the wrist-side
# end of the metacarpal bone (carpometacarpal joint), and `*-phalanx-proximal`
# sits at the MCP knuckle. The retargeter consumes the "MCP" landmark as the
# finger base, so we need to know which WebXR joint Quest actually reports
# there: anatomically realistic MCP-knuckle distances from the wrist on an
# adult hand are ~5–9 cm; carpometacarpal distances are ~1–3 cm.
_FINGER_METACARPAL_INDICES = {
    "index": WEBXR_HAND_JOINT_NAMES.index("index-finger-metacarpal"),
    "middle": WEBXR_HAND_JOINT_NAMES.index("middle-finger-metacarpal"),
    "ring": WEBXR_HAND_JOINT_NAMES.index("ring-finger-metacarpal"),
    "pinky": WEBXR_HAND_JOINT_NAMES.index("pinky-finger-metacarpal"),
}
_FINGER_PHALANX_PROXIMAL_INDICES = {
    "index": WEBXR_HAND_JOINT_NAMES.index("index-finger-phalanx-proximal"),
    "middle": WEBXR_HAND_JOINT_NAMES.index("middle-finger-phalanx-proximal"),
    "ring": WEBXR_HAND_JOINT_NAMES.index("ring-finger-phalanx-proximal"),
    "pinky": WEBXR_HAND_JOINT_NAMES.index("pinky-finger-phalanx-proximal"),
}
_KNUCKLE_DISTANCE_THRESHOLD_M = 0.045


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
        "--anatomy-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Print per-frame wrist→metacarpal and wrist→phalanx-proximal "
            "distances per finger, with a verdict about which WebXR joint "
            "Quest is reporting at the MCP knuckle."
        ),
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


def _retargeter_landmark_summary(
    landmarks: np.ndarray | None,
    side: str,
    precision: int,
) -> str:
    if landmarks is None:
        return "missing"
    try:
        retargeter_landmarks = retargeter_landmarks_from_webxr(landmarks, side)
    except ValueError as exc:
        return f"unavailable ({exc})"

    sample_names = (
        "wrist",
        "thumb_tip",
        "index_tip",
        "middle_tip",
        "ring_tip",
        "pinky_tip",
    )
    samples = " ".join(
        f"{name}={_fmt_array(retargeter_landmarks[RETARGETER_HAND_LANDMARK_NAMES.index(name)], precision)}"
        for name in sample_names
    )
    return f"shape={retargeter_landmarks.shape} {samples}"


def _anatomy_check_lines(
    landmarks: np.ndarray | None,
    precision: int,
) -> list[str]:
    """Probe whether Quest reports `*-finger-metacarpal` near the wrist (spec)
    or at the MCP knuckle (what the retargeter currently assumes)."""
    if landmarks is None or landmarks.shape != (25, 3):
        return ["    (no 25-joint hand landmarks)"]

    wrist = landmarks[0]
    metacarpal_dists: list[float] = []
    phalanx_dists: list[float] = []
    rows: list[str] = []
    for finger in ("index", "middle", "ring", "pinky"):
        m_idx = _FINGER_METACARPAL_INDICES[finger]
        p_idx = _FINGER_PHALANX_PROXIMAL_INDICES[finger]
        d_m = float(np.linalg.norm(landmarks[m_idx] - wrist))
        d_p = float(np.linalg.norm(landmarks[p_idx] - wrist))
        metacarpal_dists.append(d_m)
        phalanx_dists.append(d_p)
        rows.append(
            f"    {finger:<6} wrist→metacarpal={d_m:.{precision}f}m  "
            f"wrist→phalanx_proximal={d_p:.{precision}f}m  "
            f"Δ={d_p - d_m:+.{precision}f}m"
        )

    median_metacarpal = float(np.median(metacarpal_dists))
    median_phalanx = float(np.median(phalanx_dists))

    if median_metacarpal >= _KNUCKLE_DISTANCE_THRESHOLD_M:
        verdict = (
            f"verdict: metacarpal is at the MCP knuckle "
            f"(median wrist→metacarpal={median_metacarpal:.3f}m ≥ "
            f"{_KNUCKLE_DISTANCE_THRESHOLD_M:.3f}m). "
            "Current WEBXR_TO_RETARGETER_LANDMARK_INDICES is anatomically OK."
        )
    else:
        verdict = (
            f"verdict: metacarpal is near the wrist "
            f"(median wrist→metacarpal={median_metacarpal:.3f}m < "
            f"{_KNUCKLE_DISTANCE_THRESHOLD_M:.3f}m); "
            f"phalanx-proximal is at the knuckle "
            f"(median={median_phalanx:.3f}m). "
            "Swap the index/middle/ring/pinky base indices in "
            "WEBXR_TO_RETARGETER_LANDMARK_INDICES from {5,10,15,20} to {6,11,16,21}."
        )

    palm_width_metacarpal = float(
        np.linalg.norm(
            landmarks[_FINGER_METACARPAL_INDICES["index"]]
            - landmarks[_FINGER_METACARPAL_INDICES["pinky"]]
        )
    )
    palm_width_phalanx = float(
        np.linalg.norm(
            landmarks[_FINGER_PHALANX_PROXIMAL_INDICES["index"]]
            - landmarks[_FINGER_PHALANX_PROXIMAL_INDICES["pinky"]]
        )
    )

    rows.append(
        f"    palm width (index↔pinky): metacarpal={palm_width_metacarpal:.{precision}f}m  "
        f"phalanx_proximal={palm_width_phalanx:.{precision}f}m"
    )
    rows.append(f"    {verdict}")
    return rows


def _print_summary(
    bridge: QuestTelemetryBridge,
    *,
    precision: int,
    sequence: int,
    anatomy_check: bool,
) -> None:
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
        print(
            f"  {side} retargeter landmarks: "
            f"{_retargeter_landmark_summary(hand_landmarks, side, precision)}",
            flush=True,
        )
        if anatomy_check:
            print(f"  {side} anatomy check:", flush=True)
            for line in _anatomy_check_lines(hand_landmarks, precision):
                print(line, flush=True)


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
            _print_summary(
                bridge,
                precision=args.precision,
                sequence=sequence,
                anatomy_check=args.anatomy_check,
            )
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
