"""Quest hand-tracking teleop for in-hand cube manipulation + LeRobotDataset recording.

Streams Quest hand landmarks into the ``OrcaHandRightCubeOrientation`` env from
``orca_sim`` (OrcaHand palm-up with a small cube in the palm). Each frame:
landmarks -> retargeter -> hand joint targets -> sim step -> dataset frame.

Episodes are written to a ``LeRobotDataset`` at ``--output``. Ctrl+C ends the
current episode and finalizes the dataset.

No panda arm, no wrist-pose streaming — just hand landmarks, same ingress as
``teleop_quest_hand_only.py``.

macOS note: human render must run on the main thread; launch with ``mjpython``.

Examples:
    mjpython scripts/teleop_inhand_manipulation.py \\
        --output ./datasets/orca-inhand --task "rotate the cube"

    mjpython scripts/teleop_inhand_manipulation.py \\
        --output ./datasets/orca-inhand --task "rotate the cube" \\
        --num-episodes 5 --episode-seconds 20 --push-to-hub --repo-id user/orca-inhand
"""

from __future__ import annotations

import argparse
import logging
import select
import shutil
import signal
import ssl
import sys
import time
from pathlib import Path

try:  # POSIX-only; Windows users will get keyboard controls disabled.
    import termios
    import tty
    _HAS_TERMIOS = True
except ImportError:
    _HAS_TERMIOS = False

import mujoco
import numpy as np
from orca_core import OrcaJointPositions
from orca_sim.task_envs import OrcaHandRightCubeOrientation

from orca_teleop.panda_quest.dataset_replay import retargeter_landmarks_from_webxr
from orca_teleop.panda_quest.quest_bridge import QuestTelemetryBridge
from orca_teleop.retargeting.retargeter import Retargeter, TargetPose

logger = logging.getLogger("teleop_inhand")


class KeyboardController:
    """Non-blocking single-char stdin reader for live teleop controls.

    Bindings:
      SPACE  → toggle pause (sim is not stepped, frames not recorded)
      e      → terminate current episode now (saves + advances to next)
      q      → quit the whole run cleanly (saves current episode first)

    Reads in cbreak mode so chars arrive without pressing Enter. Skipped if
    stdin isn't a TTY (e.g., redirected) or on platforms without termios.
    Keep the terminal window focused to send keystrokes — the MuJoCo viewer
    won't intercept them.
    """

    PAUSE = " "
    RESET = "e"
    QUIT = "q"

    def __init__(self) -> None:
        self.paused = False
        self.quit_requested = False
        self._reset_pending = False
        self._enabled = _HAS_TERMIOS and sys.stdin.isatty()
        self._old_attrs = None

    def start(self) -> None:
        if not self._enabled:
            logger.warning("Stdin is not a TTY / termios unavailable; keyboard controls disabled.")
            return
        try:
            fd = sys.stdin.fileno()
            self._old_attrs = termios.tcgetattr(fd)
            tty.setcbreak(fd)
            logger.info("Keyboard controls active — SPACE=pause, e=reset episode, q=quit.")
        except Exception:
            logger.exception("Failed to put stdin in cbreak mode; keyboard controls disabled.")
            self._enabled = False
            self._old_attrs = None

    def stop(self) -> None:
        if self._old_attrs is None:
            return
        try:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_attrs)
        except Exception:
            logger.exception("Failed to restore terminal mode.")
        self._old_attrs = None

    def update(self) -> None:
        """Drain pending keystrokes and update internal state."""
        if not self._enabled:
            return
        while True:
            try:
                r, _, _ = select.select([sys.stdin], [], [], 0)
            except Exception:
                return
            if not r:
                return
            ch = sys.stdin.read(1)
            if ch == self.PAUSE:
                self.paused = not self.paused
                logger.info("[KB] %s", "PAUSED" if self.paused else "RESUMED")
            elif ch == self.RESET:
                self._reset_pending = True
                logger.info("[KB] reset requested — terminating current episode.")
            elif ch == self.QUIT:
                self.quit_requested = True
                logger.info("[KB] quit requested — finalizing after current episode.")

    def consume_reset(self) -> bool:
        if self._reset_pending:
            self._reset_pending = False
            return True
        return False


class WristAngleEstimator:
    """Map the Quest hand-tracking wrist pose to an OrcaHand wrist-motor angle.

    The Quest pose matrix is derived by the runtime from hand landmarks, so this
    *is* a landmark-driven mapping — just packaged as a 4x4 frame. On the first
    valid frame we capture the pitch of the hand's local +X axis (toward the
    fingers) above the horizontal plane as the calibration zero, then on every
    subsequent frame return the delta in degrees.

    Sign convention matches the OrcaHand wrist motor: positive = flexion
    (palm tilts toward the forearm, i.e. user curls their wrist closing the
    fingers downward); negative = extension. The retargeter clamps the final
    angle to the joint's physical limits.
    """

    def __init__(self) -> None:
        self._zero_pitch_rad: float | None = None

    @staticmethod
    def _hand_x_pitch_rad(wrist_matrix: np.ndarray) -> float:
        # Hand-local +X axis (toward the fingers) expressed in world frame.
        x_in_world = np.asarray(wrist_matrix[:3, 0], dtype=float)
        z = float(np.clip(x_in_world[2], -1.0, 1.0))
        return float(np.arcsin(z))

    def reset(self) -> None:
        self._zero_pitch_rad = None

    def update(self, wrist_matrix: np.ndarray | None) -> float:
        if wrist_matrix is None:
            return 0.0
        pitch = self._hand_x_pitch_rad(wrist_matrix)
        if self._zero_pitch_rad is None:
            self._zero_pitch_rad = pitch
            return 0.0
        # OrcaHand flexion is positive; Quest pitch goes negative when the
        # user flexes (hand points downward), so delta = zero - pitch.
        return float(np.degrees(self._zero_pitch_rad - pitch))


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
        / "models" / "v2" / f"orcahand_{side}" / "config.yaml"
    )
    return str(path) if path.exists() else None


def _default_orcahand_urdf_path(side: str) -> str | None:
    path = (
        Path.home() / "Documents" / "orcahand_description"
        / "v2" / "models" / "urdf" / f"orcahand_{side}.urdf"
    )
    return str(path) if path.exists() else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path,
                        help="Local dataset root directory to write the LeRobotDataset into.")
    parser.add_argument("--task", required=True,
                        help="Task description recorded with every frame.")
    parser.add_argument("--repo-id", default=None,
                        help="LeRobotDataset repo-id (org/name). Defaults to local/<output dirname>.")
    parser.add_argument("--num-episodes", type=int, default=1,
                        help="Number of episodes to record (default: 1).")
    parser.add_argument("--episode-seconds", type=float, default=None,
                        help="Length of each episode in seconds; default: open-ended (Ctrl+C to end).")
    parser.add_argument("--rest-seconds", type=float, default=2.0,
                        help="Pause between episodes (default: 2.0).")
    parser.add_argument("--fps", type=int, default=30,
                        help="Dataset fps metadata + outer-loop target frequency (default: 30).")
    parser.add_argument("--push-to-hub", action="store_true",
                        help="After recording, push the dataset to the Hugging Face Hub.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Wipe any existing dataset at --output before creating.")
    parser.add_argument("--side", choices=("right",), default="right",
                        help="OrcaHand side. Only right is supported by the cube-orientation env.")
    parser.add_argument("--no-wrist", action="store_true",
                        help="Disable landmark-derived wrist control (wrist stays at neutral).")
    parser.add_argument("--wrist-scale", type=float, default=1.0,
                        help="Scale factor on the computed wrist angle (1.0 = 1:1 mapping).")
    parser.add_argument("--host", default="0.0.0.0", help="Quest bridge bind host.")
    parser.add_argument("--port", type=int, default=8765, help="Quest bridge HTTP port.")
    parser.add_argument("--ssl-cert", default=None, help="Optional HTTPS certificate for WebXR.")
    parser.add_argument("--ssl-key", default=None, help="Optional HTTPS key for WebXR.")
    parser.add_argument("--render-mode", default="human", choices=("human", "rgb_array", "none"),
                        help="MuJoCo render mode. 'none' is fully headless.")
    parser.add_argument("--camera", action="append", default=None,
                        help="Camera name(s) to record into the dataset (repeatable). "
                             "Default: wrist_camera and topdown. Pass an empty --camera '' to disable.")
    parser.add_argument("--image-width", type=int, default=320,
                        help="Width of recorded camera frames (default: 320).")
    parser.add_argument("--image-height", type=int, default=240,
                        help="Height of recorded camera frames (default: 240).")
    parser.add_argument("--hand-model-path", default=None,
                        help="OrcaHand model/config path for the retargeter.")
    parser.add_argument("--hand-urdf-path", default=None,
                        help="OrcaHand URDF path for the retargeter.")
    parser.add_argument("--no-keyboard", action="store_true",
                        help="Disable terminal keyboard controls (space=pause, e=reset, q=quit).")
    parser.add_argument("--log-level", default="INFO",
                        choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Lazy import LeRobot — slow + heavy, only needed when this script is actually run.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # --- Retargeter setup ----------------------------------------------------
    hand_model = args.hand_model_path or _default_orcahand_model_path(args.side)
    hand_urdf = args.hand_urdf_path or _default_orcahand_urdf_path(args.side)
    retargeter = Retargeter.from_paths(hand_model, hand_urdf)
    logger.info("Retargeter loaded (model=%s urdf=%s).", hand_model, hand_urdf)

    # --- Env: palm-up OrcaHand + cube ---------------------------------------
    render_mode = None if args.render_mode == "none" else args.render_mode
    env = OrcaHandRightCubeOrientation(render_mode=render_mode, version="v2")
    env.reset()
    actuator_joint_names = list(env.hand.config.joint_ids)
    n_act = len(actuator_joint_names)
    neutral_positions = OrcaJointPositions(env.hand.config.neutral_position)
    neutral_rad = np.deg2rad(neutral_positions.as_array(actuator_joint_names)).astype(np.float32)
    last_action_rad = neutral_rad.copy()
    logger.info(
        "Env ready: scene=%s, %d actuators, cube body=%s",
        Path(env.scene_path).name, n_act, env.cube_body_name,
    )

    # --- Cameras (in-scene named cameras rendered into the dataset) ---------
    # Default to both wrist_camera and topdown if --camera wasn't passed.
    if args.camera is None:
        camera_names = ["wrist_camera", "topdown"]
    else:
        camera_names = [c for c in args.camera if c]  # filter empty -> disables
    available = {
        mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_CAMERA, i)
        for i in range(env.model.ncam)
    }
    missing = [c for c in camera_names if c not in available]
    if missing:
        raise ValueError(
            f"Cameras not found in the loaded scene: {missing}. "
            f"Available: {sorted(n for n in available if n)}"
        )
    renderer = mujoco.Renderer(env.model, height=args.image_height, width=args.image_width) if camera_names else None
    if camera_names:
        logger.info("Recording cameras: %s @ %dx%d", camera_names, args.image_width, args.image_height)

    # --- LeRobotDataset ------------------------------------------------------
    root = args.output.expanduser().resolve()
    repo_id = args.repo_id or f"local/{root.name}"
    if args.overwrite and root.exists():
        logger.info("--overwrite: removing existing dataset at %s", root)
        shutil.rmtree(root)

    features = {
        "observation.state": {"dtype": "float32", "shape": (n_act,), "names": actuator_joint_names},
        "observation.cube_pos": {"dtype": "float32", "shape": (3,), "names": ["x", "y", "z"]},
        "observation.cube_quat": {"dtype": "float32", "shape": (4,), "names": ["w", "x", "y", "z"]},
        "action": {"dtype": "float32", "shape": (n_act,), "names": actuator_joint_names},
    }
    for cam_name in camera_names:
        features[f"observation.images.{cam_name}"] = {
            "dtype": "video",
            "shape": (args.image_height, args.image_width, 3),
            "names": ["height", "width", "channels"],
        }
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=args.fps,
        features=features,
        root=root,
        use_videos=bool(camera_names),
    )
    logger.info("Dataset created at %s (repo_id=%s, fps=%d)", dataset.root, repo_id, args.fps)

    # --- Quest bridge --------------------------------------------------------
    bridge = QuestTelemetryBridge(
        host=args.host,
        port=args.port,
        ssl_context=_ssl_context(args.ssl_cert, args.ssl_key),
    )
    bridge.start()
    logger.info("Quest bridge listening at %s", bridge.url)
    if bridge.ssl_context is None:
        logger.info("WebXR requires HTTPS — front the bridge with an HTTPS tunnel (e.g. ngrok).")

    # Ctrl+C ends the current episode (first), and ends the whole run (second).
    interrupt = {"count": 0}
    def _on_sigint(*_):
        interrupt["count"] += 1
    signal.signal(signal.SIGINT, _on_sigint)

    period = 1.0 / max(args.fps, 1)
    landmarks_missing = True
    streaming_logged = False
    wrist_estimator = WristAngleEstimator()
    keyboard = KeyboardController()
    if not args.no_keyboard:
        keyboard.start()

    try:
        for ep_idx in range(args.num_episodes):
            if interrupt["count"] >= 2 or keyboard.quit_requested:
                break
            logger.info("=== Episode %d / %d ===", ep_idx + 1, args.num_episodes)
            # Reset env state (puts the cube back in the palm at its keyframe pose).
            env.reset()
            last_action_rad = neutral_rad.copy()
            wrist_estimator.reset()  # re-calibrate zero per episode
            interrupt_at_episode_start = interrupt["count"]

            episode_deadline = (
                time.monotonic() + args.episode_seconds
                if args.episode_seconds is not None else float("inf")
            )
            n_frames = 0
            next_tick = time.monotonic()
            ep_t0 = time.monotonic()

            while time.monotonic() < episode_deadline:
                if interrupt["count"] > interrupt_at_episode_start:
                    logger.info("Episode interrupted; finalizing %d frames.", n_frames)
                    break

                # 0) Drain keyboard input
                keyboard.update()
                if keyboard.consume_reset():
                    logger.info("Episode %d terminated by user; finalizing %d frames.",
                                ep_idx + 1, n_frames)
                    break
                if keyboard.quit_requested:
                    logger.info("Quit requested; finalizing %d frames.", n_frames)
                    break
                if keyboard.paused:
                    # Sim halted; keep the viewer responsive by ticking the
                    # passive viewer (env.step is bypassed when paused).
                    try:
                        env.render()
                    except Exception:
                        pass
                    time.sleep(0.05)
                    next_tick = time.monotonic()
                    continue

                # 1) Pull latest hand landmarks + wrist pose
                landmarks = bridge.state.get_hand_landmarks(args.side)
                wrist_matrix = bridge.state.get_hand_wrist_matrix(args.side)
                if args.no_wrist:
                    wrist_deg = 0.0
                else:
                    wrist_deg = wrist_estimator.update(wrist_matrix) * args.wrist_scale

                if landmarks is None:
                    if not landmarks_missing:
                        logger.info("Quest %s hand landmarks lost; holding last action.", args.side)
                        landmarks_missing = True
                else:
                    if landmarks_missing:
                        logger.info("Quest %s hand landmarks received; retargeting.", args.side)
                        landmarks_missing = False
                    try:
                        joint_positions = retargeter_landmarks_from_webxr(landmarks, args.side)
                        target = TargetPose(
                            joint_positions=joint_positions,
                            source="mediapipe",
                            wrist_angle_degrees=wrist_deg,
                        )
                        action = retargeter.retarget(target)
                    except Exception:
                        logger.exception("Retargeting failed; holding last action.")
                        action = None
                    if action is not None:
                        if not streaming_logged:
                            streaming_logged = True
                            logger.info("Retargeter calibrated; streaming actions to sim.")
                        last_action_rad = np.deg2rad(action.as_array(actuator_joint_names)).astype(np.float32)

                # 2) Step the sim
                try:
                    env.step(last_action_rad)
                except Exception:
                    logger.exception("env.step() failed; aborting episode.")
                    break

                # 3) Record this frame
                hand_qpos_rad = np.asarray(
                    env.hand.get_joint_position().as_array(actuator_joint_names),
                    dtype=np.float32,
                )
                cube_pos = env.data.xpos[env._cube_body_id].astype(np.float32)
                cube_quat = env.data.xquat[env._cube_body_id].astype(np.float32)
                frame: dict = {
                    "observation.state": hand_qpos_rad,
                    "observation.cube_pos": cube_pos,
                    "observation.cube_quat": cube_quat,
                    "action": last_action_rad,
                    "task": args.task,
                }
                if renderer is not None:
                    for cam_name in camera_names:
                        renderer.update_scene(env.data, camera=cam_name)
                        frame[f"observation.images.{cam_name}"] = np.asarray(
                            renderer.render(), dtype=np.uint8
                        )
                try:
                    dataset.add_frame(frame)
                    n_frames += 1
                except Exception:
                    logger.exception("dataset.add_frame failed; aborting episode.")
                    break

                # 4) Pace the outer loop
                next_tick += period
                sleep_s = next_tick - time.monotonic()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    next_tick = time.monotonic()

            ep_dur = time.monotonic() - ep_t0
            logger.info("Saving episode %d (%d frames, %.1fs)", ep_idx + 1, n_frames, ep_dur)
            try:
                dataset.save_episode()
            except Exception:
                logger.exception("save_episode() failed")

            if ep_idx + 1 < args.num_episodes and interrupt["count"] < 2 and args.rest_seconds > 0:
                logger.info("Resting %.1fs before next episode...", args.rest_seconds)
                # Sleep in small chunks so a second Ctrl+C breaks out promptly.
                rest_end = time.monotonic() + args.rest_seconds
                while time.monotonic() < rest_end and interrupt["count"] < 2:
                    time.sleep(0.05)
    finally:
        try:
            num_eps = dataset.num_episodes
        except Exception:
            num_eps = "?"
        logger.info("Dataset now contains %s episode(s) at %s", num_eps, dataset.root)

        if args.push_to_hub:
            try:
                logger.info("Pushing %s to the Hugging Face Hub...", repo_id)
                dataset.push_to_hub()
                logger.info("Push complete.")
            except Exception:
                logger.exception("push_to_hub failed")

        try:
            keyboard.stop()
        except Exception:
            logger.exception("keyboard.stop() failed")
        try:
            bridge.stop()
        finally:
            if renderer is not None:
                try:
                    renderer.close()
                except Exception:
                    logger.exception("renderer.close() failed")
            try:
                env.close()
            except Exception:
                logger.exception("env.close() failed")


if __name__ == "__main__":
    main()
