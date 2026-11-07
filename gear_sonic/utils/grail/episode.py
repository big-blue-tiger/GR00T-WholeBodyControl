"""Shared scene reset for live control and simulation-clock data collection."""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .observations import rotation
from .plan import sample_carry_plan, table_outward_normal


def reset_scene(model, data, cfg, manifest, initial, rng, planner):
    mujoco.mj_resetData(model, data)
    top = rng.uniform(*cfg["table"]["top_height_range"])
    table_pos = np.r_[cfg["table"]["position_xy"], top - 0.02]
    table_quat = cfg["table"]["quaternion_wxyz"]
    normal = table_outward_normal(table_quat, cfg["table"]["size"], [0, 1])
    tangent = np.array([normal[1], -normal[0]])
    samples = rng.random(4)
    robot_pos = np.r_[
        table_pos[:2] + tangent * (2 * samples[0] - 1) + normal * 1.5,
        initial["root_trans_offset"][2] + cfg["robot"]["init_z_offset"],
    ]
    if "initial_position_xy" in cfg["robot"]:
        robot_pos[:2] = cfg["robot"]["initial_position_xy"]
    direction = table_pos[:2] - robot_pos[:2]
    yaw = np.arctan2(direction[1], direction[0])
    initial_rot = Rotation.from_quat(initial["root_rot"])
    initial_yaw = np.arctan2(initial_rot.as_matrix()[1, 0], initial_rot.as_matrix()[0, 0])
    quat = (Rotation.from_euler("z", yaw - initial_yaw) * initial_rot).as_quat()[[3, 0, 1, 2]]
    data.qpos[:7] = np.r_[robot_pos, quat]
    for field, names in (
        ("dof", cfg["robot"]["initial_body_joint_names"]),
        ("hand_dof_pos", cfg["robot"]["initial_hand_joint_names"]),
    ):
        for name, value in zip(names, initial[field]):
            data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
    obj_quat = np.array(cfg["object"]["quaternion_wxyz"])
    obj_quat /= np.linalg.norm(obj_quat)
    vertices = np.asarray(manifest["vertices"]) @ rotation(obj_quat).T
    rest = np.r_[
        table_pos[:2] + tangent * (samples[1] - 0.5) + normal * (0.25 * samples[2]),
        top - vertices[:, 2].min(),
    ]
    plan = sample_carry_plan(rest, obj_quat, yaw, normal, rng, planner)
    address = model.jnt_qposadr[model.joint("grail_object_joint").id]
    data.qpos[address:address + 7] = np.r_[
        rest + [0, 0, cfg["object"]["init_z_offset"]], obj_quat
    ]
    poses = [
        ("grail_table", table_pos, table_quat),
        ("grail_carry", plan.carry_position, plan.carry_quaternion),
        ("grail_release", plan.final_position, plan.final_quaternion),
    ]
    for name, pos, orientation in poses:
        index = model.body_mocapid[model.body(name).id]
        data.mocap_pos[index] = pos
        data.mocap_quat[index] = orientation
    mujoco.mj_forward(model, data)
    details = dict(
        table_top=top, object_rest=rest.tolist(),
        object_initial=data.qpos[address:address + 7].tolist(),
        carry=plan.carry_position.tolist(), release=plan.final_position.tolist(),
        base_goal_xy=plan.base_goal_xy.tolist(),
        plan_drop_s=plan.hold_end_frame / 25, collision=manifest["collision_note"],
    )
    return plan, details
