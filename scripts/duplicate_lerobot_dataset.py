"""Duplicate every episode of a LeRobotDataset N times into a new dataset.

Use case: blow up a tiny demo set (e.g. 5 episodes recorded with
teleop_panda_quest_bare.py) into a 50x copy so BC training has enough
samples to overfit / converge for an end-to-end pipeline sanity check
before recording proper data.

Example
-------

    python scripts/duplicate_lerobot_dataset.py \\
        --source-repo-id fracapuano/orcapanda-base \\
        --dest-repo-id fracapuano/orcapanda-base-50x \\
        --repeats 50

    # push the result so a training run can pull it cleanly
    python scripts/duplicate_lerobot_dataset.py --repeats 50 --push-to-hub
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from orca_teleop.panda_quest.lerobot_recorder import default_lerobot_root

logger = logging.getLogger("duplicate_lerobot_dataset")

_INTERNAL_FRAME_KEYS = {"index", "frame_index", "episode_index", "task_index", "timestamp"}


def _image_to_uint8_hwc(value) -> np.ndarray:
    """Round-trip an image so add_frame writes the same format the recorder did.

    LeRobotDataset.__getitem__ decodes videos to float32 CHW in [0, 1].
    The MuJoCo recorder originally wrote uint8 HWC, and add_frame's video
    encoder is happiest with that shape, so convert back.
    """
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return arr


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo-id", default="fracapuano/orcapanda-base")
    parser.add_argument("--dest-repo-id", default="fracapuano/orcapanda-base-50x")
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--source-root", default=None, type=Path)
    parser.add_argument("--dest-root", default=None, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--private", action="store_true")
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

    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")

    logger.info("Loading source dataset %s ...", args.source_repo_id)
    src = LeRobotDataset(args.source_repo_id, root=args.source_root)
    logger.info(
        "Source: %d episodes, %d frames, fps=%d, features=%s",
        src.num_episodes,
        src.num_frames,
        src.fps,
        list(src.features),
    )

    dest_root = (
        args.dest_root if args.dest_root is not None else default_lerobot_root(args.dest_repo_id)
    )
    if args.overwrite and dest_root.exists():
        logger.info("--overwrite: removing existing dataset at %s", dest_root)
        shutil.rmtree(dest_root)

    image_feature_keys = {
        name for name, info in src.features.items() if info.get("dtype") in ("image", "video")
    }
    user_feature_keys = [k for k in src.features if k not in _INTERNAL_FRAME_KEYS]

    logger.info(
        "Creating destination dataset %s at %s (repeats=%d, %d → %d episodes)",
        args.dest_repo_id,
        dest_root,
        args.repeats,
        src.num_episodes,
        src.num_episodes * args.repeats,
    )
    dst = LeRobotDataset.create(
        repo_id=args.dest_repo_id,
        fps=src.fps,
        features=src.features,
        root=dest_root,
        robot_type=getattr(src.meta, "robot_type", None),
        use_videos=any(info.get("dtype") == "video" for info in src.features.values()),
    )

    n_src_episodes = src.num_episodes
    n_dest_episodes = n_src_episodes * args.repeats
    written_frames = 0
    episodes = src.meta.episodes

    for dest_ep_idx in range(n_dest_episodes):
        src_ep_idx = dest_ep_idx % n_src_episodes
        ep_row = episodes[src_ep_idx]
        from_idx = int(ep_row["dataset_from_index"])
        to_idx = int(ep_row["dataset_to_index"])

        for frame_idx in range(from_idx, to_idx):
            item = src[frame_idx]
            task = item.get("task", "")
            frame = {}
            for key in user_feature_keys:
                if key in image_feature_keys:
                    frame[key] = _image_to_uint8_hwc(item[key])
                else:
                    frame[key] = _to_numpy(item[key])
            frame["task"] = task
            dst.add_frame(frame)
            written_frames += 1

        dst.save_episode()

        if (dest_ep_idx + 1) % max(1, n_src_episodes) == 0:
            logger.info(
                "Wrote %d/%d destination episodes (%d frames so far)",
                dest_ep_idx + 1,
                n_dest_episodes,
                written_frames,
            )

    finalize = getattr(dst, "finalize", None)
    if callable(finalize):
        finalize()
    logger.info(
        "Done: %d destination episodes, %d total frames at %s",
        n_dest_episodes,
        written_frames,
        dst.root,
    )

    if args.push_to_hub:
        logger.info("Pushing %s to the Hugging Face Hub ...", args.dest_repo_id)
        dst.push_to_hub(private=args.private)
        logger.info("Push complete.")


if __name__ == "__main__":
    main()
