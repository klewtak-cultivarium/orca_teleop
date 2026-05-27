"""Static grasp diagnostic for the OrcaPanda cube_stacking scene.

Bypasses teleop, the Quest bridge, and the retargeter. Loads the scene
directly, resets to the home keyframe, teleports the red cube to a
configurable offset from the palm body, applies a hard-coded closed-grasp
ctrl vector to the OrcaHand actuators, and steps the simulation while
logging:

  - number of active contacts where the cube is one of the geoms
  - total contact-force magnitude on the cube
  - cube position drift from its spawn position

Three failure modes the output discriminates between:

  - cube_contacts is 0 throughout
        the fingers never actually touch the cube. The closed-grasp pose
        leaves a gap, or the cube spawn offset is wrong. Tune
        ``--cube-offset`` and re-run.
  - cube_contacts > 0, but drift grows rapidly and force is huge
        contacts form but solver/mass/stiffness mismatch ejects the cube.
        Physics config issue.
  - cube_contacts > 0, drift stays small, force in a stable range
        the static grasp works. The teleop-time failure is a control
        problem (Quest noise, kp tracking, arm jerk) and no further
        XML tuning will help.

Usage:
    python scripts/test_static_grasp.py
    python scripts/test_static_grasp.py --cube-offset 0 -0.04 -0.02
    python scripts/test_static_grasp.py --no-grasp     # gravity only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

SCENE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scene_snapshots"
    / "orcapanda_cube_stacking_local_2026-05-22"
    / "orcapanda_cube_stacking.xml"
)
KEYFRAME_NAME = "orcapanda_home"
PALM_BODY = "orcahand_right_R-Carpals_8d1f1041"
RED_CUBE_BODY = "red_cube"
RED_CUBE_GEOM = "red_cube_geom"

GRASP_CTRL: dict[str, float] = {
    "act_orcahand_right_R-Carpals_8d1f1041_to_TopTower-Model_4a80d30e": 0.0,
    # Index
    "act_orcahand_right_I-AP-R_d95d02d1_to_R-Carpals_8d1f1041": 0.0,
    "act_orcahand_right_I-PP_bacbd481_to_I-AP-R_d95d02d1": 1.2,
    "act_orcahand_right_I-FingerTipAssembly_ec49c16c_to_I-PP_bacbd481": 1.0,
    # Middle
    "act_orcahand_right_M-AP_e04a96f2_to_R-Carpals_8d1f1041": 0.0,
    "act_orcahand_right_M-PP_08efa608_to_M-AP_e04a96f2": 1.2,
    "act_orcahand_right_M-FingerTipAssembly_34afb748_to_M-PP_08efa608": 1.0,
    # Ring
    "act_orcahand_right_M-AP_6ec59111_to_R-Carpals_8d1f1041": 0.0,
    "act_orcahand_right_M-PP_8660a1eb_to_M-AP_6ec59111": 1.2,
    "act_orcahand_right_M-FingerTipAssembly_424a8e75_to_M-PP_8660a1eb": 1.0,
    # Pinky
    "act_orcahand_right_P-AP_f5e42b61_to_R-Carpals_8d1f1041": 0.0,
    "act_orcahand_right_P-PP_1d411b9b_to_P-AP_f5e42b61": 1.2,
    "act_orcahand_right_P-FingerTipAssembly_cd219176_to_P-PP_1d411b9b": 1.0,
    # Thumb
    "act_orcahand_right_T-TP-R_1c2b802d_to_R-Carpals_8d1f1041": 0.4,
    "act_orcahand_right_R-T-AP_a9723101_to_T-TP-R_1c2b802d": 0.5,
    "act_orcahand_right_T-PP_68395e98_to_R-T-AP_a9723101": 0.8,
    "act_orcahand_right_T-DP_b7429e50_to_T-PP_68395e98": 0.6,
}

N_STEPS = 1000
LOG_EVERY = 50


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cube-offset",
        type=float,
        nargs=3,
        default=[0.0, -0.04, -0.02],
        metavar=("DX", "DY", "DZ"),
        help="World-frame offset from the palm body where the cube spawns.",
    )
    parser.add_argument(
        "--no-grasp",
        action="store_true",
        help="Skip the closed-grasp ctrl. Hand stays at keyframe pose.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=N_STEPS,
        help=f"Number of simulation steps to run (default {N_STEPS}).",
    )
    args = parser.parse_args()

    if not SCENE_PATH.exists():
        print(f"Scene file not found: {SCENE_PATH}", file=sys.stderr)
        return 1

    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)

    keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, KEYFRAME_NAME)
    if keyframe_id < 0:
        print(f"Keyframe '{KEYFRAME_NAME}' not found.", file=sys.stderr)
        return 1
    mujoco.mj_resetDataKeyframe(model, data, keyframe_id)
    mujoco.mj_forward(model, data)

    palm_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PALM_BODY)
    palm_pos = data.xpos[palm_body_id].copy()
    palm_rot = data.xmat[palm_body_id].reshape(3, 3).copy()
    print(f"Palm body world position: {palm_pos}")
    print("Palm rotation rows (world axes of palm-local x,y,z):")
    for axis_name, row in zip("xyz", palm_rot.T):
        print(f"  palm-{axis_name} (world) = {row}")
    print()

    cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, RED_CUBE_BODY)
    cube_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, RED_CUBE_GEOM)
    cube_joint_id = model.body_jntadr[cube_body_id]
    cube_qposadr = model.jnt_qposadr[cube_joint_id]
    cube_dofadr = model.jnt_dofadr[cube_joint_id]

    cube_spawn = palm_pos + np.asarray(args.cube_offset, dtype=np.float64)
    data.qpos[cube_qposadr : cube_qposadr + 3] = cube_spawn
    data.qpos[cube_qposadr + 3 : cube_qposadr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
    data.qvel[cube_dofadr : cube_dofadr + 6] = 0.0

    if not args.no_grasp:
        for actuator_name, ctrl_value in GRASP_CTRL.items():
            actuator_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
            )
            if actuator_id < 0:
                print(f"Actuator '{actuator_name}' not found.", file=sys.stderr)
                return 1
            data.ctrl[actuator_id] = ctrl_value

    mujoco.mj_forward(model, data)

    initial_cube_pos = data.qpos[cube_qposadr : cube_qposadr + 3].copy()
    timestep_ms = model.opt.timestep * 1000.0
    print(f"Cube spawn (world):       {initial_cube_pos}")
    print(f"Cube offset from palm:    {args.cube_offset}")
    print(f"Closed grasp applied:     {not args.no_grasp}")
    print(f"Timestep:                 {timestep_ms:.2f} ms")
    print(f"Running:                  {args.steps} steps ({args.steps*timestep_ms/1000:.2f} s)")
    print()
    print(
        f"{'t_ms':>7} {'ncon':>5} {'cube_con':>9} {'force_N':>9} {'drift_mm':>9}"
    )

    force_scratch = np.zeros(6, dtype=np.float64)

    for step in range(args.steps):
        mujoco.mj_step(model, data)

        if step % LOG_EVERY == 0 or step == args.steps - 1:
            cube_contacts = 0
            total_force = 0.0
            for i in range(data.ncon):
                con = data.contact[i]
                if con.geom1 == cube_geom_id or con.geom2 == cube_geom_id:
                    cube_contacts += 1
                    mujoco.mj_contactForce(model, data, i, force_scratch)
                    total_force += float(np.linalg.norm(force_scratch[:3]))
            cube_pos = data.qpos[cube_qposadr : cube_qposadr + 3]
            drift_mm = float(np.linalg.norm(cube_pos - initial_cube_pos) * 1000.0)
            t_ms = (step + 1) * timestep_ms
            print(
                f"{t_ms:>7.1f} {int(data.ncon):>5} {cube_contacts:>9}"
                f" {total_force:>9.3f} {drift_mm:>9.2f}"
            )

    final_pos = data.qpos[cube_qposadr : cube_qposadr + 3]
    drift = final_pos - initial_cube_pos
    print()
    print(f"Final cube position (world): {final_pos}")
    print(f"Drift vector (m):            {drift}")
    print(f"Drift norm:                  {np.linalg.norm(drift)*1000:.2f} mm")

    contacting_geoms: set[int] = set()
    for i in range(data.ncon):
        con = data.contact[i]
        if con.geom1 == cube_geom_id:
            contacting_geoms.add(int(con.geom2))
        elif con.geom2 == cube_geom_id:
            contacting_geoms.add(int(con.geom1))
    if contacting_geoms:
        print("\nFinal-step geoms in contact with the cube:")
        for gid in sorted(contacting_geoms):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"<geom_{gid}>"
            print(f"  - {name}")
    else:
        print("\nFinal-step geoms in contact with the cube: NONE")

    return 0


if __name__ == "__main__":
    sys.exit(main())
