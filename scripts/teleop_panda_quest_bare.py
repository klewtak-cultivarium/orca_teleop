"""Bare-scene Meta Quest teleop for behaviour-cloning data collection.

Loads the OrcaPanda robot in an empty MuJoCo world (no table, no cubes, just
a floor and a single ``frontal`` camera). The right Quest controller drives a
relative end-effector target; optional Quest hand-tracking retargets onto the
right OrcaHand. Pair with ``--record-lerobot`` to dump (frontal RGB +
proprio + action) episodes into a LeRobotDataset that can be used to train a
BC policy and deploy it back into this same scene.
"""

from __future__ import annotations

import argparse
import logging
import ssl
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from orca_core import OrcaJointPositions

from orca_teleop.panda_quest.dataset_replay import (
    iter_wrist_pose_samples,
    load_hf_wrist_pose_samples,
    retargeter_landmarks_from_quest,
    retargeter_landmarks_from_webxr,
)
from orca_teleop.panda_quest.mujoco_panda import (
    DEFAULT_BARE_KEYFRAME,
    MujocoPandaArm,
    RelativeControllerMapper,
)
from orca_teleop.panda_quest.quest_bridge import QuestTelemetryBridge

logger = logging.getLogger("teleop_panda_quest_bare")

# orca_sim hand env advances 5 mj_steps per control frame at 30 Hz.
_ORCA_SIM_HAND_FRAME_SKIP = 5
_DEFAULT_FRONTAL_CAMERA = "frontal"


@dataclass(frozen=True)
class LiveControlFrame:
    arm_target_qpos: np.ndarray
    hand_action: OrcaJointPositions | None
    teleop_active: bool


_KEY_SPACE = 32
_KEY_R = 82
_AXIS_COLORS = (
    (0.95, 0.20, 0.20),
    (0.20, 0.85, 0.20),
    (0.20, 0.40, 0.95),
)


def _make_key_callback(bridge: QuestTelemetryBridge):
    def key_callback(keycode: int) -> None:
        if keycode == _KEY_SPACE:
            bridge.state.push_event("recenter")
            logger.info("Recenter requested (space).")
        elif keycode == _KEY_R:
            bridge.state.push_event("reset")
            logger.info("Reset requested (r).")

    return key_callback


def _add_frame_geoms(
    scene,
    matrix: np.ndarray | None,
    start_idx: int,
    *,
    scale: float = 0.10,
    width: float = 0.005,
    alpha: float = 1.0,
) -> int:
    import mujoco

    if matrix is None:
        return start_idx
    if start_idx + 3 > scene.maxgeom:
        return start_idx
    origin = np.ascontiguousarray(np.asarray(matrix[:3, 3], dtype=np.float64))
    rotation = np.asarray(matrix[:3, :3], dtype=np.float64)
    identity_mat = np.eye(3, dtype=np.float64).flatten()
    zero_pos = np.zeros(3, dtype=np.float64)
    zero_size = np.zeros(3, dtype=np.float64)
    for axis in range(3):
        idx = start_idx + axis
        geom = scene.geoms[idx]
        rgba = np.array([*_AXIS_COLORS[axis], alpha], dtype=np.float32)
        mujoco.mjv_initGeom(
            geom,
            type=mujoco.mjtGeom.mjGEOM_ARROW,
            size=zero_size,
            pos=zero_pos,
            mat=identity_mat,
            rgba=rgba,
        )
        end = np.ascontiguousarray(origin + rotation[:, axis] * scale)
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_ARROW,
            width,
            origin,
            end,
        )
    return start_idx + 3


def _ssl_context(certfile: str | None, keyfile: str | None) -> ssl.SSLContext | None:
    if certfile is None and keyfile is None:
        return None
    if certfile is None or keyfile is None:
        raise ValueError("--ssl-cert and --ssl-key must be passed together.")
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.load_cert_chain(certfile=certfile, keyfile=keyfile)
    return context


def _button_value(buttons: list[float], index: int) -> float:
    if index < 0:
        return 1.0
    return buttons[index] if index < len(buttons) else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-path",
        default=None,
        help="Optional override for the bare-scene MJCF (defaults to the vendored snapshot).",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Quest bridge bind host.")
    parser.add_argument("--port", type=int, default=8765, help="Quest bridge HTTP port.")
    parser.add_argument("--ssl-cert", default=None, help="Optional HTTPS certificate for WebXR.")
    parser.add_argument("--ssl-key", default=None, help="Optional HTTPS key for WebXR.")
    parser.add_argument(
        "--pose",
        default=DEFAULT_BARE_KEYFRAME,
        help=f"Reset keyframe. Defaults to {DEFAULT_BARE_KEYFRAME!r}.",
    )
    parser.add_argument(
        "--viewer-camera",
        default=_DEFAULT_FRONTAL_CAMERA,
        help=(
            f"Fixed MuJoCo camera name for the passive viewer. Defaults to "
            f"{_DEFAULT_FRONTAL_CAMERA!r} so the operator sees what the policy "
            "will see at inference time."
        ),
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run the control loop without opening the MuJoCo viewer.",
    )
    parser.add_argument(
        "--headless-max-frames",
        type=int,
        default=None,
        help="Optional maximum number of headless loop frames before exiting.",
    )
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument(
        "--input-mode",
        choices=("hand", "controller"),
        default="controller",
        help=(
            "Live Quest input source. 'controller' uses grip pose; "
            "'hand' uses experimental WebXR hand tracking."
        ),
    )
    parser.add_argument(
        "--hold-button-index",
        type=int,
        default=1,
        help="Quest gamepad button that enables arm motion; -1 means always active.",
    )
    parser.add_argument("--translation-scale", type=float, default=1.0)
    parser.add_argument(
        "--replay-hf-dataset",
        default=None,
        help="Replay recorded Quest wrist poses from a Hugging Face dataset.",
    )
    parser.add_argument("--replay-filename", default="data.parquet")
    parser.add_argument("--replay-fps", type=float, default=30.0)
    parser.add_argument("--replay-refresh", action="store_true")
    parser.add_argument("--replay-once", action="store_true")
    parser.add_argument(
        "--hand-retargeting",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Retarget Quest hand landmarks into the right OrcaHand "
            "when hand landmarks are available."
        ),
    )
    parser.add_argument(
        "--hand-model-path",
        default=None,
        help="Optional OrcaHand model/config path for the retargeter.",
    )
    parser.add_argument(
        "--hand-urdf-path",
        default=None,
        help="Optional OrcaHand URDF path for the retargeter.",
    )
    parser.add_argument(
        "--no-rotation",
        action="store_true",
        help="Only map controller translation.",
    )
    parser.add_argument(
        "--debug-overlay",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Draw RGB triads for the carpals body, IK target, and source wrist in the viewer. "
            "Disabled by default so recorded frames stay clean for BC training."
        ),
    )
    parser.add_argument(
        "--overlay-scale",
        type=float,
        default=0.20,
        help="Length of the carpals/target RGB triads in meters.",
    )
    parser.add_argument(
        "--overlay-width",
        type=float,
        default=0.012,
        help="Shaft width of the overlay arrows in meters.",
    )
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=0,
        help="Number of mj_step calls to run after reset before the viewer opens.",
    )
    parser.add_argument(
        "--gravity-compensation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cancel gravity on the robot DOFs so the arm holds q_home steadily.",
    )
    parser.add_argument(
        "--sim-steps-per-frame",
        type=int,
        default=_ORCA_SIM_HAND_FRAME_SKIP,
        help="MuJoCo substeps per teleop frame while the clutch is engaged.",
    )
    parser.add_argument(
        "--live-fps",
        type=float,
        default=None,
        help="Optional cap on the live viewer loop rate in Hz.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    parser.add_argument(
        "--record-lerobot",
        action="store_true",
        help="Record stepped MuJoCo observations/actions into a LeRobotDataset.",
    )
    parser.add_argument(
        "--record-repo-id",
        default="fracapuano/orca-panda-bare",
        help="LeRobotDataset repo id used for local metadata and optional Hub upload.",
    )
    parser.add_argument(
        "--record-task",
        default="Teleoperate the OrcaPanda in free space.",
        help="Task string stored with each LeRobot frame.",
    )
    parser.add_argument(
        "--record-root",
        type=Path,
        default=None,
        help="Optional local LeRobotDataset root.",
    )
    parser.add_argument("--record-overwrite", action="store_true")
    parser.add_argument("--record-push-to-hub", action="store_true")
    parser.add_argument("--record-private", action="store_true")
    parser.add_argument("--record-fps", type=int, default=30)
    parser.add_argument(
        "--record-camera",
        action="append",
        default=[],
        help=(
            "MuJoCo camera to record. Repeatable. Defaults to all scene cameras "
            f"(currently just {_DEFAULT_FRONTAL_CAMERA!r})."
        ),
    )
    parser.add_argument("--record-width", type=int, default=640)
    parser.add_argument("--record-height", type=int, default=480)
    return parser.parse_args()


def _default_right_orcahand_v2_model_path() -> str | None:
    try:
        import orca_core
    except Exception:
        return None

    path = (
        Path(orca_core.__file__).resolve().parent
        / "models"
        / "v2"
        / "orcahand_right"
        / "config.yaml"
    )
    return str(path) if path.exists() else None


def _default_right_orcahand_v2_urdf_path() -> str | None:
    path = (
        Path.home()
        / "Documents"
        / "orcahand_description"
        / "v2"
        / "models"
        / "urdf"
        / "orcahand_right.urdf"
    )
    return str(path) if path.exists() else None


def _make_right_hand_retargeter(args: argparse.Namespace, arm: MujocoPandaArm):
    if not args.hand_retargeting:
        return None
    if args.side != "right":
        logger.warning(
            "Hand retargeting is disabled for --side=%s; the bare scene ships the right OrcaHand.",
            args.side,
        )
        return None
    if not arm.hand_actuator_id_by_joint:
        logger.warning(
            "Hand retargeting is disabled; this MuJoCo scene has no mapped OrcaHand actuators."
        )
        return None

    from orca_teleop.retargeting.retargeter import Retargeter

    hand_model_path = args.hand_model_path or _default_right_orcahand_v2_model_path()
    hand_urdf_path = args.hand_urdf_path or _default_right_orcahand_v2_urdf_path()
    retargeter = Retargeter.from_paths(hand_model_path, hand_urdf_path)
    logger.info(
        "Hand retargeting enabled for %d right OrcaHand actuators; first frames calibrate scale.",
        len(arm.hand_actuator_id_by_joint),
    )
    logger.info("Using hand model=%s urdf=%s", hand_model_path, hand_urdf_path)
    return retargeter


def _make_hf_replay_retargeter(args: argparse.Namespace, arm: MujocoPandaArm, samples):
    if not any(sample.landmarks is not None for sample in samples):
        logger.warning(
            "Hand retargeting is disabled; the HF replay rows do not contain hand landmarks."
        )
        return None
    return _make_right_hand_retargeter(args, arm)


def _apply_control_frame(
    arm: MujocoPandaArm,
    frame: LiveControlFrame | np.ndarray,
    *,
    sim_steps_per_frame: int,
) -> np.ndarray:
    if isinstance(frame, LiveControlFrame):
        if frame.teleop_active:
            if frame.hand_action is not None:
                arm.set_hand_ctrl(frame.hand_action)
            arm.step(frame.arm_target_qpos, nstep=max(1, sim_steps_per_frame))
        else:
            arm.sync_hold()
        return frame.arm_target_qpos

    arm.step(frame, nstep=max(1, sim_steps_per_frame))
    return frame


def _run_with_viewer(
    arm: MujocoPandaArm,
    control_step,
    *,
    sim_steps_per_frame: int,
    viewer_camera: str | None,
    recorder=None,
    key_callback=None,
    overlay_step=None,
    fps: float | None = None,
) -> None:
    import mujoco.viewer

    target_qpos = arm.arm_qpos()
    period = (1.0 / float(fps)) if fps is not None and fps > 0 else None
    next_tick = time.monotonic()
    with mujoco.viewer.launch_passive(
        arm.model, arm.data, key_callback=key_callback
    ) as viewer:
        if viewer_camera is None:
            viewer.cam.lookat[:] = [0.05, 0.0, 0.55]
            viewer.cam.distance = 2.35
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -22
        else:
            camera_id = arm.camera_id(viewer_camera)
            viewer.cam.type = arm.mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = camera_id

        while viewer.is_running():
            frame = control_step(target_qpos)
            target_qpos = _apply_control_frame(
                arm,
                frame,
                sim_steps_per_frame=sim_steps_per_frame,
            )
            if recorder is not None:
                recorder.maybe_record_step()
            if overlay_step is not None:
                overlay_step(viewer)
            viewer.sync()
            if period is not None:
                next_tick += period
                sleep_s = next_tick - time.monotonic()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    next_tick = time.monotonic()


def _run_headless(
    arm: MujocoPandaArm,
    control_step,
    *,
    sim_steps_per_frame: int,
    fps: float | None,
    recorder=None,
    should_stop=None,
    max_frames: int | None = None,
) -> None:
    target_qpos = arm.arm_qpos()
    period = (1.0 / float(fps)) if fps is not None and fps > 0 else None
    next_tick = time.monotonic()
    frame_idx = 0
    while True:
        if max_frames is not None and frame_idx >= max_frames:
            break
        if should_stop is not None and should_stop():
            break

        frame = control_step(target_qpos)
        target_qpos = _apply_control_frame(
            arm,
            frame,
            sim_steps_per_frame=sim_steps_per_frame,
        )
        if recorder is not None:
            recorder.maybe_record_step()

        frame_idx += 1
        if period is None:
            continue
        next_tick += period
        sleep_s = next_tick - time.monotonic()
        if sleep_s > 0:
            time.sleep(sleep_s)
        else:
            next_tick = time.monotonic()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    arm = MujocoPandaArm(
        scene="bare",
        scene_path=args.scene_path,
        pose=args.pose,
        settle_steps=args.settle_steps,
        gravity_compensation=args.gravity_compensation,
    )
    bridge = QuestTelemetryBridge(
        host=args.host,
        port=args.port,
        ssl_context=_ssl_context(args.ssl_cert, args.ssl_key),
    )
    mapper = RelativeControllerMapper(
        translation_scale=args.translation_scale,
        rotation_enabled=not args.no_rotation,
    )
    recorder = None
    if args.record_lerobot:
        from orca_teleop.panda_quest.lerobot_recorder import (
            LeRobotPandaRecorder,
            LeRobotRecordingConfig,
        )

        recorder = LeRobotPandaRecorder(
            arm,
            LeRobotRecordingConfig(
                repo_id=args.record_repo_id,
                task=args.record_task,
                fps=args.record_fps,
                root=args.record_root,
                overwrite=args.record_overwrite,
                push_to_hub=args.record_push_to_hub,
                private=args.record_private,
                camera_names=tuple(args.record_camera),
                image_width=args.record_width,
                image_height=args.record_height,
            ),
        )
        recorder.start()

    try:
        if args.replay_hf_dataset is not None:
            samples = load_hf_wrist_pose_samples(
                args.replay_hf_dataset,
                filename=args.replay_filename,
                side=args.side,
                refresh=args.replay_refresh,
            )
            retargeter = _make_hf_replay_retargeter(args, arm, samples)
            sample_iter = iter_wrist_pose_samples(samples, loop=not args.replay_once)
            period = 1.0 / args.replay_fps
            next_tick = time.monotonic()
            current_sample = next(sample_iter)
            hand_ready_logged = False
            replay_done = False
            logger.info(
                "Replaying %d %s wrist poses from %s/%s at %.1f fps.",
                len(samples),
                args.side,
                args.replay_hf_dataset,
                args.replay_filename,
                args.replay_fps,
            )

            def replay_step(target_qpos: np.ndarray) -> np.ndarray:
                nonlocal current_sample, hand_ready_logged, next_tick, replay_done, retargeter
                now = time.monotonic()
                if now >= next_tick:
                    try:
                        current_sample = next(sample_iter)
                    except StopIteration:
                        replay_done = True
                        return target_qpos
                    next_tick = now + period

                if not mapper.calibrated:
                    mapper.calibrate(current_sample.matrix, arm.end_effector_matrix())
                    logger.info("Calibrated dataset wrist-to-end-effector relative pose.")
                if retargeter is not None and current_sample.landmarks is not None:
                    from orca_teleop.retargeting.retargeter import TargetPose

                    try:
                        hand_target = TargetPose(
                            joint_positions=retargeter_landmarks_from_quest(
                                current_sample.landmarks,
                                current_sample.side,
                            ),
                            source="mediapipe",
                        )
                        hand_action = retargeter.retarget(hand_target)
                        if hand_action is not None:
                            arm.set_hand_ctrl(hand_action)
                            if not hand_ready_logged:
                                hand_ready_logged = True
                                logger.info(
                                    "Hand retargeting calibrated; applying OrcaHand controls."
                                )
                    except Exception:
                        logger.exception(
                            "Hand retargeting failed; continuing with arm-only replay."
                        )
                        retargeter = None
                target_matrix = mapper.target_matrix(current_sample.matrix)
                return arm.solve_ik(target_matrix, initial_qpos=target_qpos)

            if args.headless:
                _run_headless(
                    arm,
                    replay_step,
                    sim_steps_per_frame=args.sim_steps_per_frame,
                    fps=args.replay_fps,
                    recorder=recorder,
                    should_stop=lambda: replay_done,
                    max_frames=args.headless_max_frames,
                )
            else:
                _run_with_viewer(
                    arm,
                    replay_step,
                    sim_steps_per_frame=args.sim_steps_per_frame,
                    viewer_camera=args.viewer_camera,
                    recorder=recorder,
                )
            return

        bridge.start()
        logger.info("Quest bridge listening at %s", bridge.url)
        if bridge.ssl_context is None:
            logger.info(
                "Quest Browser needs a secure context. Use an HTTPS tunnel, for example: "
                "ngrok http %d",
                args.port,
            )

        logger.info("Loading MuJoCo model: %s", arm.model_path)
        live_retargeter = None
        hand_ready_logged = False
        hand_calibration_logged = False
        hand_landmarks_missing_logged = False
        if args.input_mode == "hand":
            live_retargeter = _make_right_hand_retargeter(args, arm)
            if live_retargeter is None:
                logger.info(
                    "Hand mode: show your Quest %s hand and press SPACE in the viewer to "
                    "engage the clutch (auto-recenters at the current wrist pose). "
                    "Press SPACE again to disengage; press R to reset. "
                    "Hand retargeting is disabled.",
                    args.side,
                )
            else:
                logger.info(
                    "Hand mode: show your Quest %s hand and press SPACE in the viewer to "
                    "engage the clutch (auto-recenters at the current wrist pose). "
                    "Press SPACE again to disengage; press R to reset. "
                    "Arm and OrcaHand stay frozen until the clutch is engaged.",
                    args.side,
                )
        else:
            logger.info(
                "Controller mode: hold Quest %s controller button %d to engage the clutch "
                "(auto-recenters at the current grip pose on each engage). Release to "
                "disengage; press R to reset.",
                args.side,
                args.hold_button_index,
            )

        last_status_log = 0.0
        last_source_matrix: np.ndarray | None = None
        hand_clutch_state = False
        prev_clutch_engaged = False
        last_hand_action: OrcaJointPositions | None = None
        last_retarget_recv_mono = 0.0

        try:

            def live_step(target_qpos: np.ndarray) -> LiveControlFrame:
                nonlocal hand_calibration_logged, hand_landmarks_missing_logged
                nonlocal hand_ready_logged, last_status_log, live_retargeter
                nonlocal last_source_matrix, hand_clutch_state, prev_clutch_engaged
                nonlocal last_hand_action, last_retarget_recv_mono

                if args.input_mode == "hand":
                    source_matrix = bridge.state.get_hand_wrist_matrix(args.side)
                    hand_landmarks = bridge.state.get_hand_landmarks(args.side)
                    if bridge.state.pop_event("recenter"):
                        hand_clutch_state = not hand_clutch_state
                        logger.info(
                            "Hand clutch %s.",
                            "engaged" if hand_clutch_state else "disengaged",
                        )
                    clutch_engaged = hand_clutch_state
                else:
                    source_matrix = bridge.state.get_controller_matrix(args.side)
                    hand_landmarks = None
                    buttons = bridge.state.get_controller_buttons(args.side)
                    clutch_engaged = (
                        _button_value(buttons, args.hold_button_index) > 0.5
                    )
                    if bridge.state.pop_event("recenter") and clutch_engaged:
                        if source_matrix is not None:
                            mapper.calibrate(source_matrix, arm.end_effector_matrix())
                            logger.info(
                                "Manual recenter on %s controller grip pose.",
                                args.side,
                            )

                last_source_matrix = source_matrix

                if bridge.state.pop_event("reset"):
                    logger.info("Reset event received.")
                    arm.reset(args.pose)
                    mapper.reset()
                    target_qpos = arm.arm_qpos()
                    hand_clutch_state = False
                    clutch_engaged = False
                    last_hand_action = None
                    last_retarget_recv_mono = 0.0

                clutch_just_engaged = clutch_engaged and not prev_clutch_engaged
                clutch_just_disengaged = (not clutch_engaged) and prev_clutch_engaged
                if clutch_just_disengaged:
                    target_qpos = arm.arm_qpos()
                    logger.info("Clutch disengaged; holding current pose.")

                if clutch_just_engaged:
                    if source_matrix is None:
                        logger.warning(
                            "Clutch engaged but no %s pose is available yet; "
                            "calibration will retry once telemetry arrives.",
                            args.input_mode,
                        )
                    else:
                        mapper.calibrate(source_matrix, arm.end_effector_matrix())
                        logger.info(
                            "Calibrated %s wrist to right-carpals frame on clutch engage.",
                            args.input_mode,
                        )
                elif (
                    clutch_engaged
                    and not mapper.calibrated
                    and source_matrix is not None
                ):
                    mapper.calibrate(source_matrix, arm.end_effector_matrix())
                    logger.info(
                        "Calibrated %s wrist to right-carpals frame (delayed engage).",
                        args.input_mode,
                    )

                prev_clutch_engaged = clutch_engaged

                recv_mono = bridge.state.last_update_monotonic
                fresh_telemetry = recv_mono > last_retarget_recv_mono

                if (
                    live_retargeter is not None
                    and args.input_mode == "hand"
                    and fresh_telemetry
                ):
                    if hand_landmarks is None:
                        if not hand_landmarks_missing_logged:
                            hand_landmarks_missing_logged = True
                            logger.info(
                                "Quest %s hand landmarks lost; "
                                "holding last OrcaHand action.",
                                args.side,
                            )
                    else:
                        if hand_landmarks_missing_logged:
                            hand_landmarks_missing_logged = False
                            logger.info(
                                "Quest %s hand landmarks received; retargeting.",
                                args.side,
                            )
                        from orca_teleop.retargeting.retargeter import TargetPose

                        try:
                            hand_target = TargetPose(
                                joint_positions=retargeter_landmarks_from_webxr(
                                    hand_landmarks,
                                    args.side,
                                ),
                                source="mediapipe",
                            )
                            hand_action = live_retargeter.retarget(hand_target)
                        except Exception:
                            logger.exception(
                                "Live hand retargeting failed; "
                                "continuing with arm-only teleop."
                            )
                            live_retargeter = None
                            hand_action = None

                        if hand_action is None:
                            if not hand_calibration_logged:
                                hand_calibration_logged = True
                                logger.info(
                                    "Calibrating retargeter scale "
                                    "(first frames return no action)."
                                )
                        else:
                            if not hand_ready_logged:
                                hand_ready_logged = True
                                logger.info(
                                    "Retargeter calibrated; "
                                    "OrcaHand controls ready."
                                )
                            last_hand_action = hand_action

                    last_retarget_recv_mono = recv_mono

                if (
                    clutch_engaged
                    and mapper.calibrated
                    and source_matrix is not None
                ):
                    target_matrix = mapper.target_matrix(source_matrix)
                    target_qpos = arm.solve_ik(target_matrix, initial_qpos=target_qpos)

                now = time.monotonic()
                if now - last_status_log > 2.0:
                    last_status_log = now
                    telemetry_age = now - bridge.state.last_update_monotonic
                    logger.debug(
                        "mode=%s calibrated=%s clutch=%s source=%s landmarks=%s "
                        "telemetry_age=%.2fs q=%s",
                        args.input_mode,
                        mapper.calibrated,
                        clutch_engaged,
                        source_matrix is not None,
                        hand_landmarks is not None,
                        telemetry_age,
                        np.array2string(arm.arm_qpos(), precision=3),
                    )
                return LiveControlFrame(
                    arm_target_qpos=target_qpos,
                    hand_action=last_hand_action,
                    teleop_active=clutch_engaged,
                )

            overlay_logged_once = False
            big_scale = float(args.overlay_scale)
            big_width = float(args.overlay_width)
            small_scale = big_scale * 0.6
            small_width = big_width * 0.6

            def overlay_step(viewer) -> None:
                nonlocal overlay_logged_once
                if not args.debug_overlay:
                    return
                scene = viewer.user_scn
                if scene is None:
                    if not overlay_logged_once:
                        overlay_logged_once = True
                        logger.warning(
                            "Debug overlay disabled: viewer.user_scn is None."
                        )
                    return
                with viewer.lock():
                    idx = 0
                    idx = _add_frame_geoms(
                        scene,
                        arm.end_effector_matrix(),
                        idx,
                        scale=big_scale,
                        width=big_width,
                    )
                    if (
                        mapper.calibrated
                        and last_source_matrix is not None
                        and prev_clutch_engaged
                    ):
                        target_matrix = mapper.target_matrix(last_source_matrix)
                        idx = _add_frame_geoms(
                            scene,
                            target_matrix,
                            idx,
                            scale=big_scale,
                            width=big_width * 0.85,
                            alpha=0.45,
                        )
                    if last_source_matrix is not None:
                        idx = _add_frame_geoms(
                            scene,
                            last_source_matrix,
                            idx,
                            scale=small_scale,
                            width=small_width,
                        )
                    scene.ngeom = idx
                if not overlay_logged_once and idx > 0:
                    overlay_logged_once = True
                    logger.info(
                        "Debug overlay drawing %d geoms (scale=%.2fm width=%.3fm, maxgeom=%d).",
                        idx,
                        big_scale,
                        big_width,
                        scene.maxgeom,
                    )

            key_callback = _make_key_callback(bridge)

            if args.headless:
                _run_headless(
                    arm,
                    live_step,
                    sim_steps_per_frame=args.sim_steps_per_frame,
                    fps=args.live_fps,
                    recorder=recorder,
                    max_frames=args.headless_max_frames,
                )
            else:
                _run_with_viewer(
                    arm,
                    live_step,
                    sim_steps_per_frame=args.sim_steps_per_frame,
                    viewer_camera=args.viewer_camera,
                    recorder=recorder,
                    key_callback=key_callback,
                    overlay_step=overlay_step,
                    fps=args.live_fps,
                )
        finally:
            bridge.stop()
    finally:
        if recorder is not None:
            recorder.close()


if __name__ == "__main__":
    main()
