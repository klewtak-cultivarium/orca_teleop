"""Pink-based bimanual IK for the OrcaArm.

5-DOF-per-side partial-pose IK matching on the carpals frame,
with optional posture regularization.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import orca_arm
import pink
import pinocchio as pin
from pink.limits import ConfigurationLimit

logger = logging.getLogger(__name__)

ARM_JOINTS_PER_SIDE = 5
SIDES = ("left", "right")

# Carpals frame naming pattern: orcahand_{side}_{L|R}-Carpals_{hash}
CARPALS_SIDE_PREFIX = {"left": "L", "right": "R"}


def _find_carpals_frame_name(model: pin.Model, side: str) -> str:
    """Find the carpals frame name by pattern."""
    prefix = f"orcahand_{side}_{CARPALS_SIDE_PREFIX[side]}-Carpals_"
    for frame in model.frames:
        frame_name = frame.name
        if frame_name.startswith(prefix) and "to_" not in frame_name:
            return frame_name
    raise ValueError(f"Carpals frame not found for side={side!r}")


def _arm_joint_names() -> set[str]:
    """Return the bimanual arm joints that remain active in IK."""
    return {f"openarm_{side}_joint{i}" for side in SIDES for i in range(1, ARM_JOINTS_PER_SIDE + 1)}


def _build_arm_only_model() -> pin.Model:
    """Build a reduced model whose generalized coordinates are arm joints only."""
    full_model = pin.buildModelFromUrdf(orca_arm.URDF_PATH)
    active_joint_names = _arm_joint_names()
    neutral_configuration = pin.neutral(full_model)
    joints_to_lock = [
        joint_id
        for joint_id in range(1, full_model.njoints)
        if full_model.names[joint_id] not in active_joint_names
    ]
    return pin.buildReducedModel(full_model, joints_to_lock, neutral_configuration)


@dataclass(frozen=True)
class IKResult:
    q: np.ndarray
    position_error: dict[str, float]
    orientation_error: dict[str, float]
    converged: dict[str, bool]


class BimanualIKSolver:
    """Pink-based full-pose IK for both arms of the OrcaArm.

    One pinocchio model, one config vector. Uses pink's QP-based
    differential IK with FrameTask (position + orientation) to match
    the full 6D wrist pose. Finger and hand joints are excluded from
    the IK variables by reducing the model to the bimanual arm joints.
    """

    def __init__(
        self,
        max_iterations: int = 100,
        time_step: float = 0.1,
        position_tolerance: float = 1e-3,
        orientation_tolerance: float = 0.01,
        solver: str = "quadprog",
        orientation_cost: float | Sequence[float] = 0.0,
        posture_cost: float = 0.0,
    ) -> None:
        self._max_iterations = max_iterations
        self._time_step = time_step
        self._position_tolerance = position_tolerance
        self._orientation_tolerance = orientation_tolerance
        self._solver = solver

        # NOTE on orientation_cost: the OrcaArm has 5 DOF per side, but a 6D
        # SE(3) wrist pose has 6 dimensions of freedom. Asking the IK to track
        # full 6D pose with any orientation_cost > 0 forces the QP to trade
        # position for orientation along the unreachable direction, costing
        # tens of mm of position error even for in-reach targets. Pass a
        # 3-vector (e.g. [1, 1, 0]) to leave one body-frame axis free for a
        # 5-DOF tracking formulation that the arm CAN satisfy exactly.
        self._orientation_cost = orientation_cost

        # NOTE on posture_cost: a small positive value (e.g. 1e-3) regularizes
        # the IK against branch flips at near-singular configs. The posture
        # target is re-anchored to the initial configuration on every
        # ``solve`` call, so the task penalizes frame-to-frame *change*
        # without biasing toward any specific posture.
        self._posture_cost = posture_cost

        self._model = _build_arm_only_model()
        self._data = self._model.createData()

        self._limits = [ConfigurationLimit(self._model)]

        # Per-side frame names, tasks, and joint indices
        self._carpals_names: dict[str, str] = {}
        self._tasks: dict[str, pink.FrameTask] = {}
        self._arm_joint_indices: dict[str, list[int]] = {}

        for side in SIDES:
            self._carpals_names[side] = _find_carpals_frame_name(self._model, side)
            self._tasks[side] = pink.FrameTask(
                self._carpals_names[side],
                position_cost=1.0,
                orientation_cost=self._orientation_cost,
            )
            joint_names = [f"openarm_{side}_joint{i}" for i in range(1, ARM_JOINTS_PER_SIDE + 1)]
            self._arm_joint_indices[side] = [
                self._model.joints[self._model.getJointId(joint_name)].idx_q
                for joint_name in joint_names
            ]

        # Compatibility for existing demo scripts that still read the old
        # private attribute directly. New code should use ``arm_joint_indices``.
        self._arm_idx_q = self._arm_joint_indices

        self._posture_task: pink.PostureTask | None = (
            pink.PostureTask(cost=self._posture_cost) if self._posture_cost > 0.0 else None
        )

    @property
    def neutral_q(self) -> np.ndarray:
        return pin.neutral(self._model).copy()

    @property
    def arm_idx_q(self) -> dict[str, list[int]]:
        """Per-side q-vector indices of the 5 arm joints."""
        return self._arm_joint_indices

    @property
    def arm_joint_indices(self) -> dict[str, list[int]]:
        """Per-side configuration-vector indices of the 5 arm joints."""
        return self._arm_joint_indices

    @property
    def arm_joint_names(self) -> dict[str, list[str]]:
        """Joint names at each entry of ``self._arm_joint_indices[side]``, reverse-
        looked-up from the pinocchio model. Used to validate that this class
        and its consumers (e.g. the sink) agree on per-side joint orderings."""
        joint_names_by_side: dict[str, list[str]] = {}
        for side, indices in self._arm_joint_indices.items():
            names = []
            for configuration_index in indices:
                joint_id = next(
                    candidate_joint_id
                    for candidate_joint_id in range(self._model.njoints)
                    if self._model.joints[candidate_joint_id].idx_q == configuration_index
                )
                names.append(self._model.names[joint_id])
            joint_names_by_side[side] = names
        return joint_names_by_side

    def forward_kinematics(self, configuration: np.ndarray, side: str) -> np.ndarray:
        """Return the 3-D world position of the wrist for *configuration*."""
        return self.forward_kinematics_full(configuration, side)[:3, 3]

    def forward_kinematics_full(self, configuration: np.ndarray, side: str) -> np.ndarray:
        """Return the 4x4 world transform of the wrist for *configuration*."""
        pin.forwardKinematics(self._model, self._data, configuration)
        frame_id = self._model.getFrameId(self._carpals_names[side])
        pin.updateFramePlacement(self._model, self._data, frame_id)
        return self._data.oMf[frame_id].homogeneous.copy()

    def sample_reachable_target(self, side: str, random_generator: np.random.Generator) -> pin.SE3:
        """FK at a random arm joint config → guaranteed reachable SE3 pose."""
        configuration = pin.neutral(self._model)
        for configuration_index in self._arm_joint_indices[side]:
            lower_limit = self._model.lowerPositionLimit[configuration_index]
            upper_limit = self._model.upperPositionLimit[configuration_index]
            configuration[configuration_index] = random_generator.uniform(lower_limit, upper_limit)
        pin.forwardKinematics(self._model, self._data, configuration)
        frame_id = self._model.getFrameId(self._carpals_names[side])
        pin.updateFramePlacement(self._model, self._data, frame_id)
        return pin.SE3(self._data.oMf[frame_id])

    def solve(
        self,
        targets: dict[str, pin.SE3],
        initial_configuration: np.ndarray,
    ) -> IKResult:
        """Solve full-pose IK for one or both arms.

        Args:
            targets: ``{side: SE3 target pose}`` for each arm to solve.
            initial_configuration: full robot config to start from.

        Returns:
            IKResult with the solved config and per-side errors.
        """
        configuration = pink.Configuration(self._model, self._data, initial_configuration.copy())

        # Set targets on the frame tasks
        active_tasks = []
        for side, target_pose in targets.items():
            self._tasks[side].set_target(target_pose)
            active_tasks.append(self._tasks[side])
        if self._posture_task is not None:
            self._posture_task.set_target(initial_configuration.copy())
            active_tasks.append(self._posture_task)

        for _ in range(self._max_iterations):
            velocity = pink.solve_ik(
                configuration,
                active_tasks,
                self._time_step,
                solver=self._solver,
                limits=self._limits,
            )
            configuration.integrate_inplace(velocity, self._time_step)

            # Check convergence for all sides
            all_converged = True
            for side in targets:
                current_transform = configuration.get_transform_frame_to_world(
                    self._carpals_names[side]
                )
                position_error = np.linalg.norm(
                    current_transform.translation - targets[side].translation
                )
                orientation_error = np.linalg.norm(
                    pin.log3(current_transform.rotation.T @ targets[side].rotation)
                )
                if (
                    position_error > self._position_tolerance
                    or orientation_error > self._orientation_tolerance
                ):
                    all_converged = False
            if all_converged:
                break

        # Collect final results
        result_configuration = configuration.q
        pos_errors: dict[str, float] = {}
        ori_errors: dict[str, float] = {}
        converged: dict[str, bool] = {}
        for side, target_pose in targets.items():
            current_transform = configuration.get_transform_frame_to_world(
                self._carpals_names[side]
            )
            pos_errors[side] = float(
                np.linalg.norm(current_transform.translation - target_pose.translation)
            )
            ori_errors[side] = float(
                np.linalg.norm(pin.log3(current_transform.rotation.T @ target_pose.rotation))
            )
            converged[side] = (
                pos_errors[side] < self._position_tolerance
                and ori_errors[side] < self._orientation_tolerance
            )

        return IKResult(
            q=result_configuration,
            position_error=pos_errors,
            orientation_error=ori_errors,
            converged=converged,
        )
