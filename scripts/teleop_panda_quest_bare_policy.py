"""Autonomously drive the bare OrcaPanda MuJoCo scene with a trained LeRobot policy.

Mirror of ``teleop_panda_quest_bare.py`` but with Quest input replaced by
policy inference. Expects a checkpoint produced by ``lerobot_train`` (e.g.
``outputs/train/<run>/checkpoints/last/pretrained_model/``) whose input
features match what the bare-scene recorder logged: ``observation.state``
(24-d) plus one or more ``observation.images.<camera>`` keys.

Typical end-to-end flow
-----------------------

1. Record demos with ``scripts/teleop_panda_quest_bare.py --record-lerobot``.
2. Optionally inflate the dataset:
   ``python scripts/duplicate_lerobot_dataset.py --repeats 50 --push-to-hub``
3. Train, e.g. ACT:
   ``lerobot-train --policy.type=act \\
       --dataset.repo_id=fracapuano/orcapanda-base-50x \\
       --batch_size=32 --steps=50000 \\
       --output_dir=outputs/train/act_orcapanda``
4. Deploy:
   ``python scripts/teleop_panda_quest_bare_policy.py \\
       outputs/train/act_orcapanda/checkpoints/last/pretrained_model``

Press SPACE or R in the viewer to reset the scene and the policy action queue.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from lerobot.policies.factory import get_policy_class

from orca_teleop.panda_quest.mujoco_panda import (
    DEFAULT_BARE_KEYFRAME,
    MujocoPandaArm,
)

logger = logging.getLogger("teleop_panda_quest_bare_policy")

_ORCA_SIM_HAND_FRAME_SKIP = 5
_DEFAULT_FRONTAL_CAMERA = "frontal"
_KEY_SPACE = 32
_KEY_R = 82


def _read_policy_type(policy_path: str) -> str:
    local_config = Path(policy_path) / "config.json"
    if local_config.exists():
        config_path = local_config
    else:
        from huggingface_hub import hf_hub_download

        config_path = Path(hf_hub_download(repo_id=policy_path, filename="config.json"))
    with open(config_path) as f:
        cfg = json.load(f)
    policy_type = cfg.get("type")
    if not policy_type:
        raise ValueError(
            f"config.json in {policy_path} has no 'type' field; cannot dispatch policy class."
        )
    return policy_type


def _load_policy(policy_path: str, *, device: str | None = None):
    policy_type = _read_policy_type(policy_path)
    policy_cls = get_policy_class(policy_type)
    policy = policy_cls.from_pretrained(policy_path)
    if device is not None:
        policy.to(device)
    policy.eval()
    return policy, policy_type


def _resolve_policy_io(policy) -> tuple[dict[str, str], bool]:
    """Return ({camera_name: feature_key}, needs_state) inferred from the policy config."""
    input_features = getattr(policy.config, "input_features", None) or {}
    camera_map: dict[str, str] = {}
    for key in input_features:
        if key.startswith("observation.images."):
            cam_name = key.removeprefix("observation.images.")
            camera_map[cam_name] = key
    needs_state = "observation.state" in input_features
    return camera_map, needs_state


def _build_observation(
    arm: MujocoPandaArm,
    renderer,
    image_keys: dict[str, str],
    needs_state: bool,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    obs: dict[str, torch.Tensor] = {}
    if needs_state:
        state = arm.record_joint_qpos().astype(np.float32)
        obs["observation.state"] = torch.from_numpy(state).unsqueeze(0).to(device)
    for camera_name, feature_key in image_keys.items():
        renderer.update_scene(arm.data, camera=camera_name)
        img = renderer.render()
        img_chw = np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0
        obs[feature_key] = torch.from_numpy(img_chw).unsqueeze(0).contiguous().to(device)
    return obs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "policy_path",
        help="Local checkpoint directory (containing config.json + model.safetensors) "
        "or Hugging Face Hub repo id.",
    )
    parser.add_argument("--scene-path", default=None)
    parser.add_argument(
        "--pose",
        default=DEFAULT_BARE_KEYFRAME,
        help=f"Reset keyframe. Defaults to {DEFAULT_BARE_KEYFRAME!r}.",
    )
    parser.add_argument(
        "--viewer-camera",
        default=_DEFAULT_FRONTAL_CAMERA,
        help="Camera name shown in the MuJoCo passive viewer.",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop the run after this many inference steps.",
    )
    parser.add_argument(
        "--reset-every",
        type=int,
        default=None,
        help="Auto-reset arm + policy action queue every N inference steps.",
    )
    parser.add_argument(
        "--sim-steps-per-frame",
        type=int,
        default=_ORCA_SIM_HAND_FRAME_SKIP,
        help="MuJoCo substeps per inference frame (5 matches the recorder default).",
    )
    parser.add_argument(
        "--inference-fps",
        type=float,
        default=30.0,
        help="Cap policy inference cadence in Hz; should match the dataset fps. "
        "Set to 0 to disable the cap.",
    )
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device override (e.g. 'cuda', 'mps', 'cpu'). "
        "Default: whatever the policy config stores.",
    )
    parser.add_argument(
        "--gravity-compensation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--settle-steps", type=int, default=0)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    parser.add_argument(
        "--record-lerobot",
        action="store_true",
        help="Log policy rollouts to a new LeRobotDataset for later inspection.",
    )
    parser.add_argument(
        "--record-repo-id",
        default="fracapuano/orcapanda-base-policy-rollouts",
    )
    parser.add_argument(
        "--record-task",
        default="Policy rollout in the bare OrcaPanda scene.",
    )
    parser.add_argument("--record-root", type=Path, default=None)
    parser.add_argument("--record-overwrite", action="store_true")
    parser.add_argument("--record-push-to-hub", action="store_true")
    parser.add_argument("--record-private", action="store_true")
    parser.add_argument("--record-fps", type=int, default=30)
    parser.add_argument(
        "--record-camera",
        action="append",
        default=[],
        help="Camera to record; repeatable. Default: all scene cameras.",
    )
    parser.add_argument("--record-width", type=int, default=640)
    parser.add_argument("--record-height", type=int, default=480)
    parser.add_argument(
        "--video-out",
        type=Path,
        default=None,
        help="Render --video-camera into an mp4 at this path while the policy runs.",
    )
    parser.add_argument(
        "--video-camera",
        default=None,
        help="Camera to render into --video-out. Defaults to --viewer-camera.",
    )
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=480)
    parser.add_argument(
        "--video-fps",
        type=float,
        default=None,
        help="Playback fps for --video-out. Defaults to --inference-fps (clamped to 1 if 0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    import mujoco

    arm = MujocoPandaArm(
        scene="bare",
        scene_path=args.scene_path,
        pose=args.pose,
        settle_steps=args.settle_steps,
        gravity_compensation=args.gravity_compensation,
    )

    logger.info("Loading policy from %s ...", args.policy_path)
    policy, policy_type = _load_policy(args.policy_path, device=args.device)
    device = next(policy.parameters()).device
    logger.info(
        "Loaded %s policy (type=%s) on %s.",
        type(policy).__name__,
        policy_type,
        device,
    )

    camera_map, needs_state = _resolve_policy_io(policy)
    if not camera_map and not needs_state:
        raise SystemExit(
            "Policy declares no observation.state or observation.images.* inputs; "
            "nothing to feed it."
        )
    for cam_name in camera_map:
        arm.camera_id(cam_name)
    logger.info(
        "Policy inputs: cameras=%s, needs_state=%s",
        list(camera_map),
        needs_state,
    )

    expected_action_dim = len(arm.actuator_ids) + len(arm.hand_actuator_id_by_v2_joint)
    output_features = getattr(policy.config, "output_features", None) or {}
    action_feature = output_features.get("action")
    if action_feature is not None and tuple(action_feature.shape) != (expected_action_dim,):
        logger.warning(
            "Policy action dim %s differs from arm action dim (%d); "
            "apply_action will fail if these don't match.",
            tuple(action_feature.shape),
            expected_action_dim,
        )

    renderer = mujoco.Renderer(arm.model, height=args.image_height, width=args.image_width)

    video_writer = None
    video_renderer = None
    video_camera = None
    video_frames_written = 0
    if args.video_out is not None:
        import imageio.v2 as imageio

        video_camera = args.video_camera or args.viewer_camera
        if not video_camera:
            raise SystemExit("--video-out requires --video-camera or --viewer-camera to be set.")
        arm.camera_id(video_camera)
        video_renderer = mujoco.Renderer(
            arm.model, height=args.video_height, width=args.video_width
        )
        video_fps = args.video_fps
        if video_fps is None:
            video_fps = (
                args.inference_fps if args.inference_fps and args.inference_fps > 0 else 30.0
            )
        args.video_out.parent.mkdir(parents=True, exist_ok=True)
        video_writer = imageio.get_writer(
            str(args.video_out),
            fps=video_fps,
            codec="libx264",
            quality=8,
            macro_block_size=1,
        )
        logger.info(
            "Rendering camera %r to %s at %.1f fps.",
            video_camera,
            args.video_out,
            video_fps,
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

    reset_pending = {"flag": False}

    def reset_scene() -> None:
        arm.reset(args.pose)
        policy.reset()
        logger.info("Reset arm and policy action queue.")

    def key_callback(keycode: int) -> None:
        if keycode in (_KEY_SPACE, _KEY_R):
            reset_pending["flag"] = True
            logger.info("Reset requested via viewer key.")

    frame_idx = 0
    period = 1.0 / args.inference_fps if args.inference_fps and args.inference_fps > 0 else None

    def step_once() -> None:
        nonlocal frame_idx, video_frames_written
        with torch.inference_mode():
            obs = _build_observation(arm, renderer, camera_map, needs_state, device)
            action = policy.select_action(obs).squeeze(0).cpu().numpy()
        arm.step_action(action, nstep=max(1, args.sim_steps_per_frame))
        if recorder is not None:
            recorder.maybe_record_step()
        if video_writer is not None and video_renderer is not None:
            video_renderer.update_scene(arm.data, camera=video_camera)
            video_writer.append_data(video_renderer.render())
            video_frames_written += 1
        frame_idx += 1
        if args.reset_every is not None and frame_idx % args.reset_every == 0:
            reset_pending["flag"] = True

    try:
        if args.headless:
            next_tick = time.monotonic()
            while args.max_frames is None or frame_idx < args.max_frames:
                if reset_pending["flag"]:
                    reset_scene()
                    reset_pending["flag"] = False
                step_once()
                if period is not None:
                    next_tick += period
                    sleep_s = next_tick - time.monotonic()
                    if sleep_s > 0:
                        time.sleep(sleep_s)
                    else:
                        next_tick = time.monotonic()
        else:
            import mujoco.viewer

            with mujoco.viewer.launch_passive(
                arm.model, arm.data, key_callback=key_callback
            ) as viewer:
                if args.viewer_camera:
                    cam_id = arm.camera_id(args.viewer_camera)
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    viewer.cam.fixedcamid = cam_id
                next_tick = time.monotonic()
                while viewer.is_running():
                    if args.max_frames is not None and frame_idx >= args.max_frames:
                        break
                    if reset_pending["flag"]:
                        reset_scene()
                        reset_pending["flag"] = False
                    step_once()
                    viewer.sync()
                    if period is not None:
                        next_tick += period
                        sleep_s = next_tick - time.monotonic()
                        if sleep_s > 0:
                            time.sleep(sleep_s)
                        else:
                            next_tick = time.monotonic()
    finally:
        renderer.close()
        if video_writer is not None:
            video_writer.close()
            logger.info(
                "Wrote %d frame(s) of camera %r to %s.",
                video_frames_written,
                video_camera,
                args.video_out,
            )
        if video_renderer is not None:
            video_renderer.close()
        if recorder is not None:
            recorder.close()

    logger.info("Done. Ran %d inference steps.", frame_idx)


if __name__ == "__main__":
    main()
