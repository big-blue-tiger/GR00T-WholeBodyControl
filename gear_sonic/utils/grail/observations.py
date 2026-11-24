"""MuJoCo truth -> GRAIL structured observations, all quaternions wxyz."""

import re

import mujoco
import numpy as np


def rotation(quaternion):
    result = np.empty(9)
    mujoco.mju_quat2Mat(result, np.asarray(quaternion, dtype=np.float64))
    return result.reshape(3, 3)


def relative_pose(frame_pos, frame_rot, position, orientation):
    return np.concatenate(
        (frame_rot.T @ (position - frame_pos), (frame_rot.T @ orientation)[:, :2].reshape(6))
    )


def table_edge_line(table_pos, table_rot, table_size, pelvis_pos, pelvis_rot, hand_pos, hand_rot):
    """Encode the fixed table-local -Y edge as an infinite line in both frames."""
    _, hy, hz = np.asarray(table_size) * 0.5
    midpoint = np.array([0., -hy, hz])
    direction = np.array([1., 0., 0.])
    values = []
    for origin, orientation in ((pelvis_pos, pelvis_rot), (hand_pos, hand_rot)):
        origin_t = table_rot.T @ (origin - table_pos)
        closest_delta = midpoint + np.dot(origin_t - midpoint, direction) * direction - origin_t
        to_frame = orientation.T @ table_rot
        values.extend((to_frame @ closest_delta, to_frame @ direction))
    return np.concatenate(values)


class Observations:
    def __init__(self, model, config, bps):
        self.model, self.config = model, config
        names = config["robot"]["observation_joint_names"]
        ids = np.array([model.joint(name).id for name in names])
        if np.any(model.jnt_type[ids] != mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError("Robot observation joints must be hinges")
        self.qadr, self.vadr = model.jnt_qposadr[ids], model.jnt_dofadr[ids]
        defaults = config["robot"]["default_joint_pos"]
        self.defaults = np.array(
            [
                sum(value for pattern, value in defaults.items() if re.fullmatch(pattern, name))
                for name in names
            ]
        )
        self.bps = np.asarray(bps)
        self.pelvis = model.body("pelvis").id
        self.obj = model.body("grail_object").id
        self.table = model.body("grail_table").id
        self.table_flex = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, "Tabletop")
        self.hand = [model.body(name).id for name in config["hand_frame_links"]]
        # FrameTransformer returns index_1 first in the reference-free scene.
        self.hand_orientation = model.body("right_hand_index_1_link").id
        self.contact_ids = [model.body(name).id for name in config["contact_links"]]
        self.contact_indices = {body: i for i, body in enumerate(self.contact_ids)}
        self.history = None
        self.last_action = np.zeros(66, dtype=np.float32)

    def reset(self):
        self.history = None
        self.last_action[:] = 0

    def contacts(self, data):
        forces, table_force = self.contact_forces(data)
        return np.linalg.norm(forces, axis=1), float(np.linalg.norm(table_force))

    def contact_forces(self, data):
        forces = np.zeros((8, 3))
        table_force = np.zeros(3)
        wrench = np.empty(6)
        for i in range(data.ncon):
            contact = data.contact[i]
            a, b = [self.model.geom_bodyid[geom] if geom >= 0 else -1 for geom in contact.geom]
            if self.obj not in (a, b):
                continue
            other = b if a == self.obj else a
            mujoco.mj_contactForce(self.model, data, i, wrench)
            # Isaac Lab's force_matrix_w sums world-frame normal forces per pair.
            normal = contact.frame.reshape(3, 3)[0]
            force = wrench[0] * normal
            force *= -1 if a == self.obj else 1
            if other in self.contact_indices:
                forces[self.contact_indices[other]] += force
            elif other == self.table or (self.table_flex >= 0 and self.table_flex in contact.flex):
                table_force += force
        return forces, table_force

    def root_velocity(self, data):
        rot = data.xmat[self.pelvis].reshape(3, 3)
        velocity = np.empty(6)
        # mjOBJ_BODY's local axes are its inertial frame, which need not equal
        # the actor frame. Read COM velocity in world axes and rotate explicitly
        # to match Isaac Lab root_{lin,ang}_vel_b (COM velocity, actor axes).
        mujoco.mj_objectVelocity(
            self.model, data, mujoco.mjtObj.mjOBJ_BODY, self.pelvis, velocity, 0
        )
        return rot.T @ velocity[:3], rot.T @ velocity[3:]

    def proprio(self, data):
        rot = data.xmat[self.pelvis].reshape(3, 3)
        angular, linear = self.root_velocity(data)
        frame = np.concatenate(
            (
                data.qpos[self.qadr] - self.defaults,
                data.qvel[self.vadr],
                rot.T @ np.array([0.0, 0.0, -1.0]),
                angular,
                linear,
                self.last_action,
            )
        ).astype(np.float32)
        if frame.shape != (161,):
            raise ValueError(f"Invalid proprio frame {frame.shape}")
        if self.history is None:
            self.history = np.tile(frame, (5, 1))
        else:
            self.history[:-1] = self.history[1:]
            self.history[-1] = frame
        return self.history.copy()

    def privileged(self, data, task):
        pelvis_pos, pelvis_rot = data.xpos[self.pelvis], data.xmat[self.pelvis].reshape(3, 3)
        hand_pos = data.xpos[self.hand].mean(axis=0)
        hand_rot = data.xmat[self.hand_orientation].reshape(3, 3)
        obj_pos, obj_rot = data.xpos[self.obj], data.xmat[self.obj].reshape(3, 3)
        edge = (
            np.zeros(12) if task.playing else table_edge_line(
                data.xpos[self.table], data.xmat[self.table].reshape(3, 3),
                self.config["table"]["size"], pelvis_pos, pelvis_rot, hand_pos, hand_rot,
            )
        )

        def goal(position, quaternion):
            rot = rotation(quaternion)
            return np.concatenate(
                (
                    relative_pose(pelvis_pos, pelvis_rot, position, rot),
                    relative_pose(hand_pos, hand_rot, position, rot),
                )
            )

        goal_visible = np.linalg.norm(pelvis_pos[:2] - obj_pos[:2]) < 1.5
        release = (
            goal(task.plan.final_position, task.plan.final_quaternion)
            if task.playing and goal_visible
            else np.zeros(18)
        )
        carry = (
            goal(task.plan.carry_position, task.plan.carry_quaternion)
            if not task.playing and goal_visible
            else np.zeros(18)
        )
        forces, _ = self.contacts(data)
        sparse = task.command(pelvis_pos)
        values = np.concatenate(
            (
                self.bps,
                edge,
                relative_pose(pelvis_pos, pelvis_rot, obj_pos, obj_rot),
                relative_pose(hand_pos, hand_rot, obj_pos, obj_rot),
                [float(np.any(forces > 0.01))],
                release,
                carry,
                sparse,
            )
        ).astype(np.float32)
        if values.shape != (79,) or not np.isfinite(values).all():
            raise ValueError("Invalid task observation")
        return values
