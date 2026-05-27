"""Minimal Meta Quest -> OrcaHand sim teleop for evaluating hand-pose ingress.

Wires the same WebXR/WebSocket bridge used by ``teleop_panda_quest.py`` into
the ``orca_sim`` MuJoCo hand-only env used by ``teleop_sim.py``. No arm, no
scene, no recording. Use this to look at how the Quest delivers hand
landmarks (vs. just wrist poses) by watching the retargeted OrcaHand mirror
your real hand in the viewer.

On macOS, the MuJoCo human viewer must run on the main thread launched via
``mjpython`` (not plain ``python``).

Examples:
    mjpython scripts/teleop_quest_hand_only.py --side right
    mjpython scripts/teleop_quest_hand_only.py --side right \\
        --ssl-cert cert.pem --ssl-key key.pem
"""

from __future__ import annotations

import argparse
import logging
import ssl
import time
from collections import deque
from pathlib import Path
from statistics import mean

from orca_teleop.panda_quest.dataset_replay import retargeter_landmarks_from_webxr
from orca_teleop.panda_quest.quest_bridge import QuestTelemetryBridge
from orca_teleop.retargeting.retargeter import Retargeter, TargetPose
from orca_teleop.sim import OrcaHandSimSink

logger = logging.getLogger("teleop_quest_hand_only")


_STAGE_NAMES = ("read", "adapt", "retarget", "dispatch", "total")


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((pct / 100.0) * (len(s) - 1)))))
    return s[k]


class _Profiler:
    def __init__(self, window: int) -> None:
        self.stages: dict[str, deque[float]] = {
            name: deque(maxlen=window) for name in _STAGE_NAMES
        }
        self.pose_age_ms: deque[float] = deque(maxlen=window)
        self.ingress_ms: deque[float] = deque(maxlen=window)
        self.end_to_end_ms: deque[float] = deque(maxlen=window)
        self.frame_count = 0
        self.window_start = time.monotonic()
        self.last_report = self.window_start

    def record_stage(self, name: str, ms: float) -> None:
        self.stages[name].append(ms)

    def record_freshness(
        self,
        pose_age_ms: float,
        ingress_ms: float | None,
        end_to_end_ms: float | None,
    ) -> None:
        self.pose_age_ms.append(pose_age_ms)
        if ingress_ms is not None:
            self.ingress_ms.append(ingress_ms)
        if end_to_end_ms is not None:
            self.end_to_end_ms.append(end_to_end_ms)

    def maybe_report(self, interval_s: float) -> None:
        now = time.monotonic()
        if now - self.last_report < interval_s:
            return
        elapsed = now - self.window_start
        if elapsed <= 0 or self.frame_count == 0:
            self.last_report = now
            return
        fps = self.frame_count / elapsed
        parts = [f"loop_fps={fps:5.1f}"]
        for name in _STAGE_NAMES:
            samples = list(self.stages[name])
            if not samples:
                continue
            parts.append(
                f"{name}_ms(mean/p50/p95)="
                f"{mean(samples):5.2f}/"
                f"{_percentile(samples, 50):5.2f}/"
                f"{_percentile(samples, 95):5.2f}"
            )
        if self.pose_age_ms:
            samples = list(self.pose_age_ms)
            parts.append(
                "pose_age_ms(mean/p95)="
                f"{mean(samples):5.1f}/{_percentile(samples, 95):5.1f}"
            )
        if self.ingress_ms:
            samples = list(self.ingress_ms)
            parts.append(
                "quest_to_host_ms(mean/p50/p95)="
                f"{mean(samples):6.1f}/"
                f"{_percentile(samples, 50):6.1f}/"
                f"{_percentile(samples, 95):6.1f}"
            )
        if self.end_to_end_ms:
            samples = list(self.end_to_end_ms)
            parts.append(
                "end_to_end_ms(mean/p50/p95)="
                f"{mean(samples):6.1f}/"
                f"{_percentile(samples, 50):6.1f}/"
                f"{_percentile(samples, 95):6.1f}"
            )
        logger.info("PROFILE %s", " ".join(parts))
        self.last_report = now
        self.window_start = now
        self.frame_count = 0
        for q in self.stages.values():
            q.clear()
        self.pose_age_ms.clear()
        self.ingress_ms.clear()
        self.end_to_end_ms.clear()


def _ssl_context(certfile: str | None, keyfile: str | None) -> ssl.SSLContext | None:
    if certfile is None and keyfile is None:
        return None
    if certfile is None or keyfile is None:
        raise ValueError("--ssl-cert and --ssl-key must be passed together.")
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.load_cert_chain(certfile=certfile, keyfile=keyfile)
    return context


def _default_orcahand_model_path(side: str) -> str | None:
    try:
        import orca_core
    except Exception:
        return None
    path = (
        Path(orca_core.__file__).resolve().parent
        / "models"
        / "v2"
        / f"orcahand_{side}"
        / "config.yaml"
    )
    return str(path) if path.exists() else None


def _default_orcahand_urdf_path(side: str) -> str | None:
    path = (
        Path.home()
        / "Documents"
        / "orcahand_description"
        / "v2"
        / "models"
        / "urdf"
        / f"orcahand_{side}.urdf"
    )
    return str(path) if path.exists() else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument("--host", default="0.0.0.0", help="Quest bridge bind host.")
    parser.add_argument("--port", type=int, default=8765, help="Quest bridge HTTP port.")
    parser.add_argument("--ssl-cert", default=None, help="Optional HTTPS certificate for WebXR.")
    parser.add_argument("--ssl-key", default=None, help="Optional HTTPS key for WebXR.")
    parser.add_argument(
        "--version",
        default=None,
        help="orca_sim hand embodiment version, e.g. 'v1' or 'v2'.",
    )
    parser.add_argument(
        "--render-mode",
        default="human",
        choices=("human", "rgb_array"),
        help="MuJoCo render mode. 'human' opens a viewer; 'rgb_array' is headless.",
    )
    parser.add_argument("--hand-model-path", default=None)
    parser.add_argument("--hand-urdf-path", default=None)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--profile",
        action="store_true",
        help=(
            "Log rolling per-stage latency and pipeline FPS. quest_to_host_ms "
            "compares Quest Date.now() to host time.time(); accurate only if "
            "both clocks are NTP-synced."
        ),
    )
    parser.add_argument(
        "--profile-every",
        type=float,
        default=2.0,
        help="Seconds between PROFILE log lines (when --profile is set).",
    )
    parser.add_argument(
        "--profile-window",
        type=int,
        default=300,
        help="Max samples retained per stage for the rolling profile.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    hand_model = args.hand_model_path or _default_orcahand_model_path(args.side)
    hand_urdf = args.hand_urdf_path or _default_orcahand_urdf_path(args.side)
    retargeter = Retargeter.from_paths(hand_model, hand_urdf)
    logger.info("Retargeter loaded (model=%s urdf=%s).", hand_model, hand_urdf)

    sink = OrcaHandSimSink(
        env_name=args.side,
        version=args.version,
        render_mode=args.render_mode,
    )
    sink.connect()

    bridge = QuestTelemetryBridge(
        host=args.host,
        port=args.port,
        ssl_context=_ssl_context(args.ssl_cert, args.ssl_key),
    )
    bridge.start()
    logger.info("Quest bridge listening at %s", bridge.url)
    if bridge.ssl_context is None:
        logger.info(
            "Quest Browser requires a secure context. Use an HTTPS tunnel, e.g.: "
            "ngrok http %d",
            args.port,
        )

    # Seed the viewer with the sink's neutral pose so it pops open before any
    # hand telemetry arrives — otherwise mj_step is never called and nothing
    # is shown.
    last_action = sink._last_action
    sink.dispatch_action(last_action)

    period = 1.0 / max(args.fps, 1.0)
    landmarks_missing = True
    calibrating_logged = False
    streaming_logged = False
    profiler = _Profiler(window=args.profile_window) if args.profile else None

    try:
        next_tick = time.monotonic()
        while True:
            frame_start_ns = time.perf_counter_ns()

            t0 = time.perf_counter_ns()
            landmarks = bridge.state.get_hand_landmarks(args.side)
            recv_monotonic = bridge.state.last_update_monotonic
            recv_wall = bridge.state.last_update_wall
            client_wall_ms = bridge.state.last_client_wall_ms
            read_ns = time.perf_counter_ns() - t0

            adapt_ns = 0
            retarget_ns = 0

            if landmarks is None:
                if not landmarks_missing:
                    landmarks_missing = True
                    logger.info(
                        "Quest %s hand landmarks lost; holding last action.",
                        args.side,
                    )
            else:
                if landmarks_missing:
                    landmarks_missing = False
                    logger.info(
                        "Quest %s hand landmarks received; retargeting.",
                        args.side,
                    )
                try:
                    t0 = time.perf_counter_ns()
                    joint_positions = retargeter_landmarks_from_webxr(
                        landmarks, args.side
                    )
                    target = TargetPose(
                        joint_positions=joint_positions,
                        source="mediapipe",
                    )
                    adapt_ns = time.perf_counter_ns() - t0

                    t0 = time.perf_counter_ns()
                    action = retargeter.retarget(target)
                    retarget_ns = time.perf_counter_ns() - t0
                except Exception:
                    logger.exception("Retargeting failed; holding last action.")
                    action = None

                if action is None:
                    if not calibrating_logged:
                        calibrating_logged = True
                        logger.info(
                            "Calibrating retargeter scale (first frames return no action)."
                        )
                else:
                    if not streaming_logged:
                        streaming_logged = True
                        logger.info("Retargeter calibrated; streaming OrcaHand actions.")
                    last_action = action

            t0 = time.perf_counter_ns()
            sink.dispatch_action(last_action)
            dispatch_ns = time.perf_counter_ns() - t0

            total_ns = time.perf_counter_ns() - frame_start_ns

            if profiler is not None:
                profiler.frame_count += 1
                profiler.record_stage("read", read_ns / 1e6)
                if landmarks is not None:
                    profiler.record_stage("adapt", adapt_ns / 1e6)
                    profiler.record_stage("retarget", retarget_ns / 1e6)
                profiler.record_stage("dispatch", dispatch_ns / 1e6)
                profiler.record_stage("total", total_ns / 1e6)
                if recv_monotonic > 0:
                    pose_age_ms = (time.monotonic() - recv_monotonic) * 1000.0
                    if client_wall_ms is not None and recv_wall > 0:
                        ingress_ms = recv_wall * 1000.0 - client_wall_ms
                        end_to_end_ms = time.time() * 1000.0 - client_wall_ms
                    else:
                        ingress_ms = None
                        end_to_end_ms = None
                    profiler.record_freshness(pose_age_ms, ingress_ms, end_to_end_ms)
                profiler.maybe_report(args.profile_every)

            next_tick += period
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_tick = time.monotonic()
    except KeyboardInterrupt:
        logger.info("Interrupted; shutting down.")
    finally:
        bridge.stop()
        sink.close()


if __name__ == "__main__":
    main()
