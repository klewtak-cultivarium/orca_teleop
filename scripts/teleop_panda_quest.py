"""Barebones Meta Quest teleop for OrcaPanda MuJoCo scenes.

By default it loads the OrcaPanda cube-stacking scene, opens a MuJoCo viewer on
the host, and uses the right Quest controller as a relative end-effector target.
HF replay can additionally retarget recorded Quest hand landmarks into the
right OrcaHand actuators in the scene.
"""

from __future__ import annotations

import argparse
import logging
import ssl
import time
from pathlib import Path

import numpy as np

from orca_teleop.panda_quest.dataset_replay import (
    iter_wrist_pose_samples,
    load_hf_wrist_pose_samples,
    retargeter_landmarks_from_quest,
    retargeter_landmarks_from_webxr,
)
from orca_teleop.panda_quest.mujoco_panda import (
    DEFAULT_CUBE_STACKING_KEYFRAME,
    MujocoPandaArm,
    RelativeControllerMapper,
)
from orca_teleop.panda_quest.quest_bridge import QuestTelemetryBridge

logger = logging.getLogger("teleop_panda_quest")


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
        "--scene",
        choices=("cube-stacking", "legacy"),
        default="cube-stacking",
        help="MuJoCo scene to load. Defaults to OrcaPanda cube stacking.",
    )
    parser.add_argument(
        "--scene-path",
        default=None,
        help="Path to an OrcaPanda cube-stacking MJCF scene.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to legacy orcapanda.xml. Kept for old direct-model probes.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Quest bridge bind host.")
    parser.add_argument("--port", type=int, default=8765, help="Quest bridge HTTP port.")
    parser.add_argument("--ssl-cert", default=None, help="Optional HTTPS certificate for WebXR.")
    parser.add_argument("--ssl-key", default=None, help="Optional HTTPS key for WebXR.")
    parser.add_argument(
        "--pose",
        default=None,
        help="Reset pose/keyframe. Defaults to orcapanda_home for cube-stacking, ready for legacy.",
    )
    parser.add_argument(
        "--viewer-camera",
        default=None,
        help="Optional fixed MuJoCo camera name for the passive viewer, e.g. orcapanda_overview.",
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
        help="Replay recorded Quest wrist poses from a Hugging Face dataset, e.g. "
        "fracapuano/quest-calibration or fracapuano/quest-poses.",
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
    parser.add_argument("--sim-steps-per-frame", type=int, default=2)
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
        default="fracapuano/orca-panda-test",
        help="LeRobotDataset repo id used for local metadata and optional Hub upload.",
    )
    parser.add_argument(
        "--record-task",
        default="Teleoperate the OrcaPanda to manipulate cubes.",
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
        help="MuJoCo camera to record. Repeatable. Defaults to all scene cameras.",
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
            "Hand retargeting is disabled for --side=%s; "
            "the cube-stacking scene has a right OrcaHand.",
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


def _run_with_viewer(
    arm: MujocoPandaArm,
    control_step,
    *,
    sim_steps_per_frame: int,
    viewer_camera: str | None,
    recorder=None,
) -> None:
    import mujoco.viewer

    target_qpos = arm.arm_qpos()
    with mujoco.viewer.launch_passive(arm.model, arm.data) as viewer:
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
            target_qpos = control_step(target_qpos)
            arm.step(target_qpos, nstep=max(1, sim_steps_per_frame))
            if recorder is not None:
                recorder.maybe_record_step()
            viewer.sync()


def _run_headless(
    arm: MujocoPandaArm,
    control_step,
    *,
    sim_steps_per_frame: int,
    fps: float,
    recorder=None,
    should_stop=None,
    max_frames: int | None = None,
) -> None:
    target_qpos = arm.arm_qpos()
    period = 1.0 / float(fps)
    next_tick = time.monotonic()
    frame_idx = 0
    while True:
        if max_frames is not None and frame_idx >= max_frames:
            break
        if should_stop is not None and should_stop():
            break

        target_qpos = control_step(target_qpos)
        arm.step(target_qpos, nstep=max(1, sim_steps_per_frame))
        if recorder is not None:
            recorder.maybe_record_step()

        frame_idx += 1
        next_tick += period
        sleep_s = next_tick - time.monotonic()
        if sleep_s > 0:
            time.sleep(sleep_s)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    pose = args.pose or (
        DEFAULT_CUBE_STACKING_KEYFRAME if args.scene == "cube-stacking" else "ready"
    )
    arm = MujocoPandaArm(
        model_path=args.model_path,
        scene=args.scene,
        scene_path=args.scene_path,
        pose=pose,
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
        if args.input_mode == "hand":
            live_retargeter = _make_right_hand_retargeter(args, arm)
            logger.info("Show your Quest %s hand to drive the Panda and OrcaHand.", args.side)
        else:
            logger.info(
                "Hold Quest %s controller button %d to drive the Panda.",
                args.side,
                args.hold_button_index,
            )

        last_status_log = 0.0
        was_active = False

        try:

            def live_step(target_qpos: np.ndarray) -> np.ndarray:
                nonlocal hand_ready_logged, last_status_log, live_retargeter, was_active
                if args.input_mode == "hand":
                    source_matrix = bridge.state.get_hand_wrist_matrix(args.side)
                    hand_landmarks = bridge.state.get_hand_landmarks(args.side)
                    active = source_matrix is not None and hand_landmarks is not None
                else:
                    source_matrix = bridge.state.get_controller_matrix(args.side)
                    hand_landmarks = None
                    buttons = bridge.state.get_controller_buttons(args.side)
                    active = (
                        source_matrix is not None
                        and _button_value(buttons, args.hold_button_index) > 0.5
                    )

                if bridge.state.pop_event("reset"):
                    logger.info("Reset event received from Quest.")
                    arm.reset(pose)
                    mapper.reset()
                    target_qpos = arm.arm_qpos()

                if source_matrix is None:
                    mapper.reset()
                elif active:
                    if not mapper.calibrated or not was_active:
                        mapper.calibrate(source_matrix, arm.end_effector_matrix())
                        logger.info(
                            "Calibrated %s-to-end-effector relative pose.",
                            args.input_mode,
                        )
                    if (
                        live_retargeter is not None
                        and args.input_mode == "hand"
                        and hand_landmarks is not None
                    ):
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
                            if hand_action is not None:
                                arm.set_hand_ctrl(hand_action)
                                if not hand_ready_logged:
                                    hand_ready_logged = True
                                    logger.info(
                                        "Live hand retargeting calibrated; "
                                        "applying OrcaHand controls."
                                    )
                        except Exception:
                            logger.exception(
                                "Live hand retargeting failed; continuing with arm-only teleop."
                            )
                            live_retargeter = None
                    target_matrix = mapper.target_matrix(source_matrix)
                    target_qpos = arm.solve_ik(target_matrix, initial_qpos=target_qpos)
                elif was_active:
                    mapper.reset()
                    target_qpos = arm.arm_qpos()
                    logger.info(
                        "Lost active %s input; holding current Panda pose.",
                        args.input_mode,
                    )

                was_active = active

                now = time.monotonic()
                if now - last_status_log > 2.0:
                    last_status_log = now
                    telemetry_age = now - bridge.state.last_update_monotonic
                    logger.debug(
                        "active=%s input=%s telemetry_age=%.2fs q=%s",
                        active,
                        source_matrix is not None,
                        telemetry_age,
                        np.array2string(arm.arm_qpos(), precision=3),
                    )
                return target_qpos

            if args.headless:
                _run_headless(
                    arm,
                    live_step,
                    sim_steps_per_frame=args.sim_steps_per_frame,
                    fps=args.record_fps,
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
                )
        finally:
            bridge.stop()
    finally:
        if recorder is not None:
            recorder.close()


if __name__ == "__main__":
    main()
