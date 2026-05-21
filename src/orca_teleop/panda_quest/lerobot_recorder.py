from __future__ import annotations

import logging
import os
import queue
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from orca_teleop.panda_quest.mujoco_panda import MujocoPandaArm

logger = logging.getLogger(__name__)

_SAVE_EPISODE = object()


def default_lerobot_root(repo_id: str) -> Path:
    base = os.environ.get("HF_LEROBOT_HOME")
    if base:
        return Path(base) / repo_id
    return Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id


@dataclass(frozen=True)
class LeRobotRecordingConfig:
    repo_id: str = "fracapuano/orca-panda-test"
    task: str = "Teleoperate the OrcaPanda to manipulate cubes."
    fps: int = 30
    root: Path | None = None
    overwrite: bool = False
    push_to_hub: bool = False
    private: bool = False
    camera_names: tuple[str, ...] = ()
    image_width: int = 640
    image_height: int = 480
    queue_size: int = 64


class LeRobotPandaRecorder:
    """Asynchronous LeRobotDataset writer for the standalone Panda MuJoCo demo."""

    def __init__(self, arm: MujocoPandaArm, config: LeRobotRecordingConfig) -> None:
        self.arm = arm
        self.config = config
        self._dataset = None
        self._renderer = None
        self._rec_q: queue.Queue = queue.Queue(maxsize=config.queue_size)
        self._thread: threading.Thread | None = None
        self._started = False
        self._episode_saved = False
        self._frame_index = 0
        self._last_record_time = 0.0
        self._dropped_frames = 0
        self._writer_error: BaseException | None = None

        camera_names = config.camera_names or tuple(
            arm.model.camera(idx).name for idx in range(arm.model.ncam)
        )
        if not camera_names:
            raise ValueError(f"No MuJoCo cameras found in {arm.model_path}.")
        self.camera_names = tuple(camera_names)
        for camera_name in self.camera_names:
            arm.camera_id(camera_name)

        self.joint_names = tuple(arm.record_joint_names())

    @property
    def dataset_root(self) -> Path | None:
        if self._dataset is None:
            return None
        return Path(self._dataset.root)

    def start(self) -> None:
        if self._started:
            return

        import mujoco
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        root = self.config.root
        target_root = root if root is not None else default_lerobot_root(self.config.repo_id)
        if self.config.overwrite and target_root.exists():
            logger.info("--record-overwrite: removing existing dataset at %s", target_root)
            shutil.rmtree(target_root)

        state_shape = (len(self.joint_names),)
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": state_shape,
                "names": list(self.joint_names),
            },
            "action": {
                "dtype": "float32",
                "shape": state_shape,
                "names": list(self.joint_names),
            },
        }
        for camera_name in self.camera_names:
            features[f"observation.images.{camera_name}"] = {
                "dtype": "video",
                "shape": (self.config.image_height, self.config.image_width, 3),
                "names": ["height", "width", "channels"],
            }

        self._dataset = LeRobotDataset.create(
            repo_id=self.config.repo_id,
            fps=self.config.fps,
            features=features,
            root=root,
            robot_type="orcapanda_mujoco",
            use_videos=True,
        )
        self._renderer = mujoco.Renderer(
            self.arm.model,
            height=self.config.image_height,
            width=self.config.image_width,
        )
        self._thread = threading.Thread(
            target=self._writer_loop,
            name="lerobot-panda-recorder",
        )
        self._thread.start()
        self._started = True
        logger.info(
            "Recording LeRobotDataset %s at %s with cameras=%s joints=%d.",
            self.config.repo_id,
            self._dataset.root,
            list(self.camera_names),
            len(self.joint_names),
        )

    def maybe_record_step(self, *, force: bool = False) -> None:
        if not self._started:
            return
        now = time.perf_counter()
        period = 1.0 / float(self.config.fps)
        if not force and self._last_record_time and now - self._last_record_time < period:
            return

        frame = {
            "observation.state": self.arm.record_joint_qpos().astype(np.float32),
            "action": self.arm.record_joint_ctrl().astype(np.float32),
            "task": self.config.task,
        }
        assert self._renderer is not None
        for camera_name in self.camera_names:
            self._renderer.update_scene(self.arm.data, camera=camera_name)
            frame[f"observation.images.{camera_name}"] = np.asarray(
                self._renderer.render(),
                dtype=np.uint8,
            )

        try:
            self._rec_q.put_nowait(frame)
            self._frame_index += 1
            self._last_record_time = now
        except queue.Full:
            self._dropped_frames += 1
            if self._dropped_frames == 1 or self._dropped_frames % self.config.fps == 0:
                logger.warning(
                    "LeRobot recorder queue full; dropped %d frame(s).",
                    self._dropped_frames,
                )

    def close(self) -> None:
        if not self._started:
            return
        self._rec_q.put(_SAVE_EPISODE)
        self._rec_q.put(None)
        if self._thread is not None:
            self._thread.join()
        if self._writer_error is not None:
            raise RuntimeError("LeRobot recorder writer thread failed.") from self._writer_error
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._dataset is not None:
            finalize = getattr(self._dataset, "finalize", None)
            if callable(finalize):
                finalize()
            logger.info(
                "LeRobotDataset finalized at %s (%d frame(s), %d dropped).",
                self._dataset.root,
                self._frame_index,
                self._dropped_frames,
            )
            if self.config.push_to_hub:
                logger.info("Pushing %s to the Hugging Face Hub...", self.config.repo_id)
                self._dataset.push_to_hub(private=self.config.private)
                logger.info("Push complete.")
        self._started = False

    def _writer_loop(self) -> None:
        assert self._dataset is not None
        try:
            while True:
                item = self._rec_q.get()
                if item is None:
                    break
                if item is _SAVE_EPISODE:
                    if self._frame_index > 0 and not self._episode_saved:
                        self._dataset.save_episode(parallel_encoding=False)
                        self._episode_saved = True
                        logger.info("LeRobot episode saved.")
                    continue
                self._dataset.add_frame(item)
        except BaseException as exc:
            self._writer_error = exc
            logger.exception("LeRobot recorder writer failed.")
            while True:
                item = self._rec_q.get()
                if item is None:
                    break
                continue
