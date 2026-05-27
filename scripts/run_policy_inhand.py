"""Autonomous in-hand manipulation rollout with a LeRobot ACT policy.

Mirrors the env, cameras, dataset schema, and keyboard controls of
``teleop_inhand_manipulation.py``, but replaces Quest + retargeter with a
policy in the inner loop. Useful as a wiring smoke-test before any training
(``--from-scratch`` gives a random-init ACT; the policy output will be
gibberish but the pipeline runs), then later for evaluating a trained
checkpoint (``--policy-path PATH`` or ``--policy-repo-id REPO``).

The schema is locked to what ``teleop_inhand_manipulation.py`` records:
    observation.state               (17,) float32  hand joint positions, rad
    observation.cube_pos            (3,)  float32  cube world xyz
    observation.cube_quat           (4,)  float32  cube world wxyz
    observation.images.wrist_camera (H,W,3) uint8  forearm-mounted view
    observation.images.topdown      (H,W,3) uint8  angled top-down view
    action                          (17,) float32  hand joint targets, rad

macOS: launch with ``mjpython`` for the human viewer.

Examples:
    mjpython scripts/run_policy_inhand.py --from-scratch \\
        --output ./datasets/orca-inhand-rollout --task "rotate the cube"

    mjpython scripts/run_policy_inhand.py \\
        --policy-repo-id fracapuano/orca-inhand-act-v1 \\
        --output ./datasets/orca-inhand-rollout --task "rotate the cube"
"""

from __future__ import annotations

import argparse
import logging
import shutil
import signal
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
from orca_core import OrcaJointPositions
from orca_sim.task_envs import OrcaHandRightCubeOrientation

# Reuse the keyboard controller from the teleop script so behaviour stays in sync.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from teleop_inhand_manipulation import KeyboardController  # noqa: E402

logger = logging.getLogger("run_policy_inhand")


def _build_scratch_policy(
    n_actuators: int,
    image_height: int,
    image_width: int,
    *,
    chunk_size: int,
    device: str,
):
    """Build a randomly initialised ACT policy matching our observation schema."""
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy

    cfg = ACTConfig(
        n_obs_steps=1,
        n_action_steps=chunk_size,
        chunk_size=chunk_size,
        vision_backbone="resnet18",
        input_features={
            # Schema matches what teleop_inhand_manipulation.py records and what
            # the trained checkpoint expects, so observation construction is
            # identical across scratch and trained runs.
            "observation.state": PolicyFeature(FeatureType.STATE, (n_actuators,)),
            "observation.cube_pos": PolicyFeature(FeatureType.STATE, (3,)),
            "observation.cube_quat": PolicyFeature(FeatureType.STATE, (4,)),
            "observation.images.wrist_camera": PolicyFeature(
                FeatureType.VISUAL, (3, image_height, image_width)
            ),
            "observation.images.topdown": PolicyFeature(
                FeatureType.VISUAL, (3, image_height, image_width)
            ),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (n_actuators,))},
    )
    # Scratch policies have no calibration stats — use identity normalisation so
    # the model doesn't try to divide by uninitialised std tensors.
    cfg.normalization_mapping = {
        "VISUAL": NormalizationMode.IDENTITY,
        "STATE": NormalizationMode.IDENTITY,
        "ACTION": NormalizationMode.IDENTITY,
    }
    policy = ACTPolicy(cfg)
    policy.to(device)
    policy.eval()
    return policy


def _resolve_checkpoint_dir(source: str) -> str:
    """Accept a checkpoint dir and point at its `pretrained_model/` subdir.

    LeRobot saves checkpoints as <ckpt>/pretrained_model/{config.json,...}. Users
    naturally pass the checkpoint dir (e.g. .../checkpoints/last); auto-descend
    into pretrained_model when that's where config.json actually lives. HF Hub
    repo ids (no local dir) are returned unchanged.
    """
    p = Path(source)
    if not p.exists():
        return source  # likely a hub repo id
    if (p / "config.json").exists():
        return str(p)
    nested = p / "pretrained_model"
    if (nested / "config.json").exists():
        logger.info("Resolved checkpoint to %s", nested)
        return str(nested)
    return str(p)


def _load_pretrained_policy(source: str, device: str):
    """Returns (policy, preprocessor, postprocessor).

    The processors carry the dataset normalization stats baked in at train time;
    without them the policy receives unnormalized inputs and emits garbage.
    """
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    resolved = _resolve_checkpoint_dir(source)
    policy = ACTPolicy.from_pretrained(resolved)
    policy.to(device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=resolved)
    return policy, preprocessor, postprocessor


def _hwc_uint8_to_chw_float(img: np.ndarray, device: str) -> torch.Tensor:
    """(H, W, 3) uint8 -> (1, 3, H, W) float32 in [0, 1]."""
    t = torch.from_numpy(img).to(device=device, non_blocking=True)
    return t.permute(2, 0, 1).float().div_(255.0).unsqueeze(0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group()
    src.add_argument(
        "--from-scratch",
        action="store_true",
        help="Build a randomly-initialised ACT policy (default).",
    )
    src.add_argument(
        "--policy-path",
        default=None,
        help="Local directory holding a trained ACT checkpoint (LeRobot format).",
    )
    src.add_argument(
        "--policy-repo-id",
        default=None,
        help="HF Hub repo id of a trained ACT checkpoint, e.g. user/orca-inhand-act.",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=100,
        help="Action chunk size for the (scratch) ACT config (default: 100).",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Torch device: 'cpu', 'cuda', 'mps'. Default: cuda > mps > cpu.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional LeRobotDataset root to record the rollout into.",
    )
    p.add_argument(
        "--task",
        default="autonomous in-hand rollout",
        help="Task description recorded with every frame.",
    )
    p.add_argument(
        "--repo-id",
        default=None,
        help="LeRobotDataset repo-id (org/name). Default: local/<output dirname>.",
    )
    p.add_argument("--num-episodes", type=int, default=1)
    p.add_argument(
        "--episode-seconds",
        type=float,
        default=None,
        help="Length of each episode; default: open-ended (e/q to end).",
    )
    p.add_argument("--rest-seconds", type=float, default=2.0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Wipe any existing dataset at --output before creating.",
    )
    p.add_argument("--render-mode", default="human", choices=("human", "rgb_array", "none"))
    p.add_argument("--image-width", type=int, default=320)
    p.add_argument("--image-height", type=int, default=240)
    p.add_argument(
        "--camera",
        action="append",
        default=None,
        help="Camera names to render (repeatable). Default: wrist_camera + topdown.",
    )
    p.add_argument(
        "--no-keyboard",
        action="store_true",
        help="Disable terminal keyboard controls (SPACE / e / q).",
    )
    p.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return p.parse_args()


def _pick_device(arg: str | None) -> str:
    if arg is not None:
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    device = _pick_device(args.device)
    logger.info("Torch device: %s", device)

    # --- Env: palm-up OrcaHand + cube ---------------------------------------
    render_mode = None if args.render_mode == "none" else args.render_mode
    env = OrcaHandRightCubeOrientation(render_mode=render_mode, version="v2")
    env.reset()
    actuator_joint_names = list(env.hand.config.joint_ids)
    n_act = len(actuator_joint_names)
    neutral_rad = np.deg2rad(
        OrcaJointPositions(env.hand.config.neutral_position).as_array(actuator_joint_names)
    ).astype(np.float32)
    logger.info(
        "Env ready: scene=%s, %d actuators, cube body=%s",
        Path(env.scene_path).name,
        n_act,
        env.cube_body_name,
    )

    # --- Cameras ------------------------------------------------------------
    if args.camera is None:
        camera_names = ["wrist_camera", "topdown"]
    else:
        camera_names = [c for c in args.camera if c]
    available = {
        mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(env.model.ncam)
    }
    missing = [c for c in camera_names if c not in available]
    if missing:
        raise ValueError(
            f"Cameras not found in the loaded scene: {missing}. "
            f"Available: {sorted(n for n in available if n)}"
        )
    renderer = mujoco.Renderer(env.model, height=args.image_height, width=args.image_width)
    logger.info("Cameras: %s @ %dx%d", camera_names, args.image_width, args.image_height)

    # --- Policy -------------------------------------------------------------
    preprocessor = None
    postprocessor = None
    if args.policy_path:
        logger.info("Loading policy from local checkpoint: %s", args.policy_path)
        policy, preprocessor, postprocessor = _load_pretrained_policy(args.policy_path, device)
        policy_source = f"path:{args.policy_path}"
    elif args.policy_repo_id:
        logger.info("Loading policy from HF Hub: %s", args.policy_repo_id)
        policy, preprocessor, postprocessor = _load_pretrained_policy(args.policy_repo_id, device)
        policy_source = f"hub:{args.policy_repo_id}"
    else:
        logger.info("Building scratch ACT policy (chunk_size=%d, vision=resnet18)", args.chunk_size)
        policy = _build_scratch_policy(
            n_act,
            args.image_height,
            args.image_width,
            chunk_size=args.chunk_size,
            device=device,
        )
        policy_source = "scratch"  # identity normalization; no processors needed
    n_params = sum(p.numel() for p in policy.parameters())
    logger.info(
        "Policy ready: source=%s, params=%.1fM, processors=%s",
        policy_source,
        n_params / 1e6,
        preprocessor is not None,
    )

    # --- Optional dataset to record the rollout into ------------------------
    dataset = None
    if args.output is not None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        root = args.output.expanduser().resolve()
        repo_id = args.repo_id or f"local/{root.name}"
        if args.overwrite and root.exists():
            logger.info("--overwrite: removing existing dataset at %s", root)
            shutil.rmtree(root)
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (n_act,),
                "names": actuator_joint_names,
            },
            "observation.cube_pos": {"dtype": "float32", "shape": (3,), "names": ["x", "y", "z"]},
            "observation.cube_quat": {
                "dtype": "float32",
                "shape": (4,),
                "names": ["w", "x", "y", "z"],
            },
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
            use_videos=True,
        )
        logger.info("Recording rollout to %s (repo_id=%s)", dataset.root, repo_id)

    # --- Ctrl+C + keyboard --------------------------------------------------
    interrupt = {"count": 0}

    def _on_sigint(*_):
        interrupt["count"] += 1

    signal.signal(signal.SIGINT, _on_sigint)

    keyboard = KeyboardController()
    if not args.no_keyboard:
        keyboard.start()

    period = 1.0 / max(args.fps, 1)

    try:
        for ep_idx in range(args.num_episodes):
            if interrupt["count"] >= 2 or keyboard.quit_requested:
                break
            logger.info(
                "=== Episode %d / %d (policy=%s) ===", ep_idx + 1, args.num_episodes, policy_source
            )
            env.reset()
            policy.reset()  # clear ACT's action queue between episodes
            last_action_rad = neutral_rad.copy()
            interrupt_at_episode_start = interrupt["count"]
            episode_deadline = (
                time.monotonic() + args.episode_seconds
                if args.episode_seconds is not None
                else float("inf")
            )
            n_frames = 0
            next_tick = time.monotonic()
            ep_t0 = time.monotonic()

            while time.monotonic() < episode_deadline:
                if interrupt["count"] > interrupt_at_episode_start:
                    logger.info("Episode interrupted; finalizing %d frames.", n_frames)
                    break

                keyboard.update()
                if keyboard.consume_reset():
                    logger.info(
                        "Episode %d terminated by user; finalizing %d frames.", ep_idx + 1, n_frames
                    )
                    break
                if keyboard.quit_requested:
                    logger.info("Quit requested; finalizing %d frames.", n_frames)
                    break
                if keyboard.paused:
                    try:
                        env.render()
                    except Exception:
                        pass
                    time.sleep(0.05)
                    next_tick = time.monotonic()
                    continue

                # 1) Build observation (state + camera frames in the policy's expected layout)
                hand_qpos_rad = np.asarray(
                    env.hand.get_joint_position().as_array(actuator_joint_names),
                    dtype=np.float32,
                )
                cube_pos = env.data.xpos[env._cube_body_id].astype(np.float32)
                cube_quat = env.data.xquat[env._cube_body_id].astype(np.float32)

                cam_images_hwc: dict[str, np.ndarray] = {}
                obs_raw: dict = {
                    "observation.state": torch.from_numpy(hand_qpos_rad),
                    "observation.cube_pos": torch.from_numpy(cube_pos),
                    "observation.cube_quat": torch.from_numpy(cube_quat),
                }
                for cam_name in camera_names:
                    renderer.update_scene(env.data, camera=cam_name)
                    img = np.asarray(renderer.render(), dtype=np.uint8)
                    cam_images_hwc[cam_name] = img
                    # CHW float [0,1], unbatched — preprocessor/scratch path batches.
                    obs_raw[f"observation.images.{cam_name}"] = (
                        torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0)
                    )

                # 2) Policy step
                try:
                    if preprocessor is not None:
                        # Trained path: processors batch + move to device + normalize,
                        # then unnormalize the action back to physical radians.
                        obs_p = preprocessor(obs_raw)
                        with torch.inference_mode():
                            action_t = policy.select_action(obs_p)
                        action_t = postprocessor(action_t)
                    else:
                        # Scratch path: identity norm, batch + device manually.
                        batch = {k: v.unsqueeze(0).to(device) for k, v in obs_raw.items()}
                        with torch.inference_mode():
                            action_t = policy.select_action(batch)
                    action_np = action_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
                except Exception:
                    logger.exception("policy.select_action failed; holding last action.")
                    action_np = last_action_rad

                # 3) Step the sim with the predicted action (radians, OrcaHand actuator order)
                try:
                    env.step(action_np)
                    last_action_rad = action_np
                except Exception:
                    logger.exception("env.step failed; aborting episode.")
                    break

                # 4) Record the frame
                if dataset is not None:
                    frame: dict = {
                        "observation.state": hand_qpos_rad,
                        "observation.cube_pos": cube_pos,
                        "observation.cube_quat": cube_quat,
                        "action": action_np,
                        "task": args.task,
                    }
                    for cam_name, img in cam_images_hwc.items():
                        frame[f"observation.images.{cam_name}"] = img
                    try:
                        dataset.add_frame(frame)
                    except Exception:
                        logger.exception("dataset.add_frame failed; aborting episode.")
                        break
                n_frames += 1

                next_tick += period
                sleep_s = next_tick - time.monotonic()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    next_tick = time.monotonic()

            ep_dur = time.monotonic() - ep_t0
            logger.info("Episode %d: %d frames, %.1fs.", ep_idx + 1, n_frames, ep_dur)
            if dataset is not None and n_frames > 0:
                try:
                    dataset.save_episode()
                except Exception:
                    logger.exception("save_episode() failed.")
            if (
                ep_idx + 1 < args.num_episodes
                and interrupt["count"] < 2
                and not keyboard.quit_requested
            ):
                rest_end = time.monotonic() + max(args.rest_seconds, 0.0)
                while (
                    time.monotonic() < rest_end
                    and interrupt["count"] < 2
                    and not keyboard.quit_requested
                ):
                    time.sleep(0.05)
    finally:
        if dataset is not None:
            try:
                n_eps = dataset.num_episodes
            except Exception:
                n_eps = "?"
            logger.info("Dataset now contains %s episode(s) at %s", n_eps, dataset.root)
        try:
            keyboard.stop()
        except Exception:
            logger.exception("keyboard.stop() failed")
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
