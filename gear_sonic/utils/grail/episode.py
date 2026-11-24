"""Shared scene reset for live control and simulation-clock data collection."""

import json
from pathlib import Path
import pickle

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .observations import rotation
from .plan import sample_carry_plan, table_outward_normal


def load_initial_pose(path):
    path = Path(path)
    if path.suffix.lower() == ".json":
        initial = json.loads(path.read_text())
    else:
        with path.open("rb") as stream:
            initial = pickle.load(stream)
    for name, size in (
        ("dof", 29), ("hand_dof_pos", 14), ("root_rot", 4), ("root_trans_offset", 3)
    ):
        field = np.asarray(initial[name])
        if field.shape != (size,) or not np.isfinite(field).all():
            raise ValueError(f"Invalid initial pose field {name}")
    for name in ("table_normal_offset", "right_lateral_limit"):
        if name in initial:
            offset = initial[name]
            if not isinstance(offset, (int, float)) or not np.isfinite(offset) or offset <= 0:
                raise ValueError(f"{name} must be a positive distance in metres")
    if "face_object" in initial and not isinstance(initial["face_object"], bool):
        raise ValueError("face_object must be a boolean")
    if "spawn_arc" in initial:
        arc = initial["spawn_arc"]
        if not isinstance(arc, dict):
            raise ValueError("spawn_arc requires radius and angle_range_deg")
        radius = arc.get("radius")
        if not isinstance(radius, (int, float)) or not np.isfinite(radius) or radius <= 0:
            raise ValueError("spawn_arc.radius must be a positive distance in metres")
        angles = arc.get("angle_range_deg")
        if (
            not isinstance(angles, (list, tuple)) or len(angles) != 2
            or not all(isinstance(angle, (int, float)) for angle in angles)
            or not -90 < angles[0] < 0 < angles[1] < 90
        ):
            raise ValueError("spawn_arc.angle_range_deg must straddle zero within (-90, 90) degrees")
    return initial


def reset_scene(model, data, cfg, manifest, initial, rng, planner, *, release_position=None):
    mujoco.mj_resetData(model, data)
    top = rng.uniform(*cfg["table"]["top_height_range"])
    table_pos = np.r_[cfg["table"]["position_xy"], top - 0.02]
    table_quat = cfg["table"]["quaternion_wxyz"]
    normal = table_outward_normal(table_quat, cfg["table"]["size"], [0, 1])
    tangent = np.array([normal[1], -normal[0]])
    samples = rng.random(4)
    object_lateral = samples[1] - 0.5
    robot_lateral = 2 * samples[0] - 1
    object_xy = table_pos[:2] + tangent * object_lateral + normal * (0.25 * samples[2])
    normal_offset = initial.get("table_normal_offset", 1.5)
    robot_pos = np.r_[
        table_pos[:2] + tangent * robot_lateral + normal * normal_offset,
        initial["root_trans_offset"][2] + cfg["robot"]["init_z_offset"],
    ]
    if "initial_position_xy" in cfg["robot"]:
        robot_pos[:2] = cfg["robot"]["initial_position_xy"]
    elif "spawn_arc" in initial:
        arc = initial["spawn_arc"]
        right, left = arc["angle_range_deg"]
        # Preserve the sampled side of the object and sample uniformly on each side of the arc.
        angle = np.deg2rad(np.interp(robot_lateral, (-1., object_lateral, 1.), (right, 0., left)))
        robot_pos[:2] = object_xy + arc["radius"] * (np.cos(angle) * normal + np.sin(angle) * tangent)
    elif "right_lateral_limit" in initial:
        # The tangent points left for a robot approaching the table.
        right_offset = (object_xy - robot_pos[:2]) @ tangent
        robot_pos[:2] += max(right_offset - initial["right_lateral_limit"], 0.) * tangent
    facing_target = object_xy if initial.get("face_object", False) else table_pos[:2]
    direction = facing_target - robot_pos[:2]
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
    rest = np.r_[object_xy, top - vertices[:, 2].min()]
    plan = sample_carry_plan(rest, obj_quat, yaw, normal, rng, planner, release_position=release_position)
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
