"""Run a LeRobot ACT policy closed-loop in orca_sim.

Example:

    mjpython scripts/run_lerobot_policy_sim.py \
        --policy-path outputs/train/orca-panda-test-act/checkpoints/001000/pretrained_model \
        --dataset-repo-id fracapuano/orca-panda-test-50x \
        --sim-version v1 \
        --camera-name topdown
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from orca_teleop.policies import LeRobotPolicyAdapter, reset_policy
from orca_teleop.sim import OrcaHandSimSink, SimCameraConfig

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", default="fracapuano/orca-panda-test-act")
    parser.add_argument("--dataset-repo-id", default="fracapuano/orca-panda-test-50x")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sim-env", choices=["left", "right"], default="right")
    parser.add_argument("--sim-version", default=None)
    parser.add_argument("--render-mode", default="human")
    parser.add_argument("--camera-name", default="frontal")
    parser.add_argument("--camera-width", type=int, default=320)
    parser.add_argument("--camera-height", type=int, default=240)
    parser.add_argument(
        "--task", required=True, help="Task description fed to the policy at every step"
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    policy = LeRobotPolicyAdapter.from_pretrained(
        args.policy_path,
        args.dataset_repo_id,
        dataset_root=str(args.dataset_root) if args.dataset_root is not None else None,
        device=args.device,
    )
    sink = OrcaHandSimSink(
        env_name=args.sim_env,
        version=args.sim_version,
        render_mode=args.render_mode,
        camera_config=SimCameraConfig(
            name=args.camera_name,
            width=args.camera_width,
            height=args.camera_height,
        ),
    )

    sink.connect()
    reset_policy(policy)
    try:
        for step_idx in range(args.steps):
            action = sink.step_policy(policy, task=args.task)
            if step_idx % 50 == 0:
                first_joint = action.as_array(sink.joint_ids)[0]
                logger.info("step=%d first_joint=%.3f", step_idx, first_joint)
            time.sleep(sink.control_dt)  # simulator already steps once; keep human render readable
    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        sink.close()


if __name__ == "__main__":
    main(sys.argv[1:])
