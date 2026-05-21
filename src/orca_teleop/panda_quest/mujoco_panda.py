from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from orca_core import OrcaJointPositions

from orca_teleop.panda_quest.transforms import make_transform, rotation_vector_from_matrix

READY_ARM_QPOS = np.array([-0.1, -1.6, -0.1, -3.0718, -0.15, 2.85, -1.4027], dtype=np.float64)
PANDA_JOINT_NAMES = tuple(f"panda_joint{idx}" for idx in range(1, 8))
DEFAULT_CUBE_STACKING_KEYFRAME = "orcapanda_home"
_RIGHT_HAND_SCENE_JOINT_WRIST = "orcahand_right_R-Carpals_8d1f1041_to_TopTower-Model_4a80d30e"
_RIGHT_HAND_SCENE_JOINT_THUMB_BASE = "orcahand_right_T-TP-R_1c2b802d_to_R-Carpals_8d1f1041"
_RIGHT_HAND_SCENE_JOINT_THUMB_ABD = "orcahand_right_R-T-AP_a9723101_to_T-TP-R_1c2b802d"
_RIGHT_HAND_SCENE_JOINT_THUMB_KNUCKLE = "orcahand_right_T-PP_68395e98_to_R-T-AP_a9723101"
_RIGHT_HAND_SCENE_JOINT_THUMB_TIP = "orcahand_right_T-DP_b7429e50_to_T-PP_68395e98"

RIGHT_HAND_SCENE_JOINT_BY_CANONICAL = {
    "wrist": "orcahand_right_R-Carpals_8d1f1041_to_TopTower-Model_4a80d30e",
    "pinky_abd": "orcahand_right_P-AP_f5e42b61_to_R-Carpals_8d1f1041",
    "pinky_mcp": "orcahand_right_P-PP_1d411b9b_to_P-AP_f5e42b61",
    "pinky_pip": "orcahand_right_P-FingerTipAssembly_cd219176_to_P-PP_1d411b9b",
    "ring_abd": "orcahand_right_M-AP_6ec59111_to_R-Carpals_8d1f1041",
    "ring_mcp": "orcahand_right_M-PP_8660a1eb_to_M-AP_6ec59111",
    "ring_pip": "orcahand_right_M-FingerTipAssembly_424a8e75_to_M-PP_8660a1eb",
    "middle_abd": "orcahand_right_M-AP_e04a96f2_to_R-Carpals_8d1f1041",
    "middle_mcp": "orcahand_right_M-PP_08efa608_to_M-AP_e04a96f2",
    "middle_pip": "orcahand_right_M-FingerTipAssembly_34afb748_to_M-PP_08efa608",
    "index_abd": "orcahand_right_I-AP-R_d95d02d1_to_R-Carpals_8d1f1041",
    "index_mcp": "orcahand_right_I-PP_bacbd481_to_I-AP-R_d95d02d1",
    "index_pip": "orcahand_right_I-FingerTipAssembly_ec49c16c_to_I-PP_bacbd481",
    "thumb_mcp": _RIGHT_HAND_SCENE_JOINT_THUMB_BASE,
    "thumb_abd": _RIGHT_HAND_SCENE_JOINT_THUMB_ABD,
    "thumb_pip": _RIGHT_HAND_SCENE_JOINT_THUMB_KNUCKLE,
    "thumb_dip": _RIGHT_HAND_SCENE_JOINT_THUMB_TIP,
}
RIGHT_HAND_SCENE_JOINT_BY_V2_CANONICAL = {
    **{
        name: scene_joint_name
        for name, scene_joint_name in RIGHT_HAND_SCENE_JOINT_BY_CANONICAL.items()
        if name != "thumb_pip"
    },
    "wrist": _RIGHT_HAND_SCENE_JOINT_WRIST,
    "thumb_cmc": _RIGHT_HAND_SCENE_JOINT_THUMB_BASE,
    "thumb_mcp": _RIGHT_HAND_SCENE_JOINT_THUMB_KNUCKLE,
}
RIGHT_HAND_V2_JOINT_NAMES = tuple(RIGHT_HAND_SCENE_JOINT_BY_V2_CANONICAL)


def resolve_legacy_orcapanda_xml(model_path: str | None = None) -> Path:
    if model_path is not None:
        path = Path(model_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    try:
        import orca_arm

        return Path(orca_arm.ORCAPANDA_MJCF_PATH).expanduser().resolve()
    except Exception:
        pass

    local_sibling = Path.home() / "Documents" / "orca_arm" / "orca_arm" / "orcapanda.xml"
    if local_sibling.exists():
        return local_sibling.resolve()

    raise FileNotFoundError(
        "Could not resolve the legacy orcapanda.xml. Pass --model-path or install orca_arm."
    )


def resolve_orcapanda_cube_stacking_xml(scene_path: str | None = None) -> Path:
    if scene_path is not None:
        path = Path(scene_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    try:
        import orca_sim

        path = Path(orca_sim.__file__).resolve().parent / "scenes" / "orcapanda_cube_stacking.xml"
        if path.exists():
            return path.resolve()
    except Exception:
        pass

    local_sibling = (
        Path.home()
        / "Documents"
        / "orca_sim"
        / "src"
        / "orca_sim"
        / "scenes"
        / "orcapanda_cube_stacking.xml"
    )
    if local_sibling.exists():
        return local_sibling.resolve()

    raise FileNotFoundError(
        "Could not resolve orcapanda_cube_stacking.xml. Pass --scene-path or install orca_sim."
    )


def resolve_panda_model_path(
    *,
    scene: str = "cube-stacking",
    model_path: str | None = None,
    scene_path: str | None = None,
) -> Path:
    if model_path is not None and scene_path is not None:
        raise ValueError("Pass either --model-path or --scene-path, not both.")
    if model_path is not None:
        return resolve_legacy_orcapanda_xml(model_path)
    if scene_path is not None:
        return resolve_orcapanda_cube_stacking_xml(scene_path)
    if scene == "legacy":
        return resolve_legacy_orcapanda_xml()
    if scene == "cube-stacking":
        return resolve_orcapanda_cube_stacking_xml()
    raise ValueError(f"Unsupported scene {scene!r}.")


@dataclass
class PandaIkConfig:
    site_name: str = "attachment_site"
    damping: float = 0.08
    max_iters: int = 16
    step_scale: float = 0.65
    position_gain: float = 1.0
    rotation_gain: float = 0.35
    max_joint_step: float = 0.12


class MujocoPandaArm:
    def __init__(
        self,
        model_path: str | None = None,
        pose: str = "ready",
        scene: str = "legacy",
        scene_path: str | None = None,
        ik_config: PandaIkConfig | None = None,
    ) -> None:
        import mujoco

        self.mujoco = mujoco
        self.model_path = resolve_panda_model_path(
            scene=scene,
            model_path=model_path,
            scene_path=scene_path,
        )
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.ik_config = PandaIkConfig() if ik_config is None else ik_config

        self.joint_ids = [self._joint_id(name) for name in PANDA_JOINT_NAMES]
        self.dof_ids = np.array(
            [self.model.jnt_dofadr[joint_id] for joint_id in self.joint_ids],
            dtype=np.int32,
        )
        self.qpos_ids = np.array(
            [self.model.jnt_qposadr[joint_id] for joint_id in self.joint_ids],
            dtype=np.int32,
        )
        self.actuator_ids = np.array(
            [self._actuator_id_for_joint(joint_id) for joint_id in self.joint_ids]
        )
        self.hand_actuator_id_by_joint = self._resolve_right_hand_actuator_mapping(
            RIGHT_HAND_SCENE_JOINT_BY_CANONICAL
        )
        self.hand_actuator_id_by_v2_joint = self._resolve_right_hand_actuator_mapping(
            RIGHT_HAND_SCENE_JOINT_BY_V2_CANONICAL
        )
        self.joint_ranges = self.model.jnt_range[self.joint_ids].astype(np.float64)
        self.site_id = self._site_id(self.ik_config.site_name)

        self.reset(pose)

    def reset(self, pose: str = "ready") -> None:
        if pose == "qpos0":
            self.mujoco.mj_resetData(self.model, self.data)
        elif pose == "ready":
            self.mujoco.mj_resetData(self.model, self.data)
            self.set_arm_qpos(READY_ARM_QPOS)
        else:
            self._reset_keyframe(pose)

        self.hold_current_qpos()
        self.mujoco.mj_forward(self.model, self.data)

    def _reset_keyframe(self, keyframe_name: str) -> None:
        key_id = self.mujoco.mj_name2id(
            self.model,
            self.mujoco.mjtObj.mjOBJ_KEY,
            keyframe_name,
        )
        if key_id < 0 and keyframe_name == "home":
            key_id = self.mujoco.mj_name2id(
                self.model,
                self.mujoco.mjtObj.mjOBJ_KEY,
                DEFAULT_CUBE_STACKING_KEYFRAME,
            )
        if key_id < 0:
            key_names = [self.model.key(idx).name for idx in range(self.model.nkey)]
            raise ValueError(
                f"This MJCF does not define keyframe {keyframe_name!r}. "
                f"Available keyframes: {key_names}"
            )
        self.mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)

    def arm_qpos(self) -> np.ndarray:
        return self.data.qpos[self.qpos_ids].copy()

    def set_arm_qpos(self, qpos: np.ndarray) -> None:
        self.data.qpos[self.qpos_ids] = self._clip_arm_qpos(qpos)

    def hold_current_qpos(self) -> None:
        for actuator_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            if joint_id < 0:
                continue
            qpos_addr = self.model.jnt_qposadr[joint_id]
            self.data.ctrl[actuator_id] = self.data.qpos[qpos_addr]

    def set_arm_ctrl(self, qpos: np.ndarray) -> None:
        self.data.ctrl[self.actuator_ids] = self._clip_arm_qpos(qpos)

    def set_hand_ctrl(self, action: OrcaJointPositions) -> None:
        action_by_joint = action.as_dict()
        actuator_id_by_joint = (
            self.hand_actuator_id_by_v2_joint
            if "thumb_cmc" in action_by_joint
            else self.hand_actuator_id_by_joint
        )
        for joint_name, actuator_id in actuator_id_by_joint.items():
            if joint_name not in action_by_joint:
                continue
            low, high = self.model.actuator_ctrlrange[actuator_id]
            self.data.ctrl[actuator_id] = float(
                np.clip(np.deg2rad(action_by_joint[joint_name]), low, high)
            )

    def record_joint_names(self) -> list[str]:
        return [*PANDA_JOINT_NAMES, *RIGHT_HAND_V2_JOINT_NAMES]

    def record_joint_qpos(self) -> np.ndarray:
        return np.array(
            [
                self.data.qpos[self.model.jnt_qposadr[joint_id]]
                for joint_id in self._record_joint_ids()
            ],
            dtype=np.float64,
        )

    def record_joint_ctrl(self) -> np.ndarray:
        hand_ctrl_by_joint = {
            name: self.data.ctrl[actuator_id]
            for name, actuator_id in self.hand_actuator_id_by_v2_joint.items()
        }
        return np.array(
            [
                *self.data.ctrl[self.actuator_ids],
                *(hand_ctrl_by_joint[name] for name in RIGHT_HAND_V2_JOINT_NAMES),
            ],
            dtype=np.float64,
        )

    def end_effector_matrix(self) -> np.ndarray:
        self.mujoco.mj_forward(self.model, self.data)
        return make_transform(
            self.data.site_xmat[self.site_id].reshape(3, 3).copy(),
            self.data.site_xpos[self.site_id].copy(),
        )

    def solve_ik(
        self,
        target_matrix: np.ndarray,
        initial_qpos: np.ndarray | None = None,
    ) -> np.ndarray:
        cfg = self.ik_config
        qpos = (
            self.arm_qpos()
            if initial_qpos is None
            else np.asarray(initial_qpos, dtype=np.float64).copy()
        )
        target_pos = target_matrix[:3, 3]
        target_rot = target_matrix[:3, :3]

        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)

        for _ in range(cfg.max_iters):
            self.set_arm_qpos(qpos)
            self.mujoco.mj_forward(self.model, self.data)

            current_pos = self.data.site_xpos[self.site_id].copy()
            current_rot = self.data.site_xmat[self.site_id].reshape(3, 3).copy()
            pos_error = target_pos - current_pos
            rot_error = rotation_vector_from_matrix(target_rot @ current_rot.T)
            error = np.concatenate([cfg.position_gain * pos_error, cfg.rotation_gain * rot_error])

            if np.linalg.norm(error[:3]) < 0.004 and np.linalg.norm(error[3:]) < 0.035:
                break

            self.mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
            jac = np.vstack([jacp[:, self.dof_ids], cfg.rotation_gain * jacr[:, self.dof_ids]])
            lhs = jac @ jac.T + (cfg.damping**2) * np.eye(6)
            delta = jac.T @ np.linalg.solve(lhs, error)
            delta = np.clip(delta, -cfg.max_joint_step, cfg.max_joint_step)
            qpos = self._clip_arm_qpos(qpos + cfg.step_scale * delta)

        return qpos

    def step(self, target_qpos: np.ndarray, nstep: int = 1) -> None:
        self.set_arm_ctrl(target_qpos)
        self.mujoco.mj_step(self.model, self.data, nstep=nstep)

    def _clip_arm_qpos(self, qpos: np.ndarray) -> np.ndarray:
        qpos = np.asarray(qpos, dtype=np.float64)
        return np.clip(qpos, self.joint_ranges[:, 0], self.joint_ranges[:, 1])

    def _joint_id(self, name: str) -> int:
        joint_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"Missing joint {name!r} in {self.model_path}")
        return joint_id

    def _site_id(self, name: str) -> int:
        site_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id < 0:
            raise ValueError(f"Missing site {name!r} in {self.model_path}")
        return site_id

    def camera_id(self, name: str) -> int:
        camera_id = self.mujoco.mj_name2id(
            self.model,
            self.mujoco.mjtObj.mjOBJ_CAMERA,
            name,
        )
        if camera_id < 0:
            camera_names = [self.model.camera(idx).name for idx in range(self.model.ncam)]
            raise ValueError(
                f"Missing camera {name!r} in {self.model_path}. "
                f"Available cameras: {camera_names}"
            )
        return camera_id

    def _record_joint_ids(self) -> list[int]:
        scene_joint_names = [
            *PANDA_JOINT_NAMES,
            *(RIGHT_HAND_SCENE_JOINT_BY_V2_CANONICAL[name] for name in RIGHT_HAND_V2_JOINT_NAMES),
        ]
        return [self._joint_id(name) for name in scene_joint_names]

    def _actuator_id_for_joint(self, joint_id: int) -> int:
        for actuator_id in range(self.model.nu):
            if int(self.model.actuator_trnid[actuator_id, 0]) == joint_id:
                return actuator_id
        joint_name = self.model.joint(joint_id).name
        raise ValueError(f"Could not find an actuator for joint {joint_name!r}.")

    def _resolve_right_hand_actuator_mapping(
        self,
        scene_joint_by_canonical: dict[str, str],
    ) -> dict[str, int]:
        mapping: dict[str, int] = {}
        for canonical_name, scene_joint_name in scene_joint_by_canonical.items():
            joint_id = self.mujoco.mj_name2id(
                self.model,
                self.mujoco.mjtObj.mjOBJ_JOINT,
                scene_joint_name,
            )
            if joint_id < 0:
                continue
            mapping[canonical_name] = self._actuator_id_for_joint(joint_id)
        return mapping


class RelativeControllerMapper:
    def __init__(self, translation_scale: float = 1.0, rotation_enabled: bool = True) -> None:
        self.translation_scale = translation_scale
        self.rotation_enabled = rotation_enabled
        self._controller0: np.ndarray | None = None
        self._ee0: np.ndarray | None = None

    @property
    def calibrated(self) -> bool:
        return self._controller0 is not None and self._ee0 is not None

    def calibrate(self, controller_matrix: np.ndarray, ee_matrix: np.ndarray) -> None:
        self._controller0 = controller_matrix.copy()
        self._ee0 = ee_matrix.copy()

    def reset(self) -> None:
        self._controller0 = None
        self._ee0 = None

    def target_matrix(self, controller_matrix: np.ndarray) -> np.ndarray:
        if self._controller0 is None or self._ee0 is None:
            raise RuntimeError(
                "RelativeControllerMapper must be calibrated before target_matrix()."
            )

        target = self._ee0.copy()
        controller_delta_p = controller_matrix[:3, 3] - self._controller0[:3, 3]
        target[:3, 3] = self._ee0[:3, 3] + self.translation_scale * controller_delta_p
        if self.rotation_enabled:
            delta_r = controller_matrix[:3, :3] @ self._controller0[:3, :3].T
            target[:3, :3] = delta_r @ self._ee0[:3, :3]
        return target
