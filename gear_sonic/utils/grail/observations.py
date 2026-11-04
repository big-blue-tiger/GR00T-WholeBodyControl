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
        self.hand = [model.body(name).id for name in config["hand_frame_links"]]
        self.contact_ids = [model.body(name).id for name in config["contact_links"]]
        self.contact_indices = {body: i for i, body in enumerate(self.contact_ids)}
        self.history = None
        self.last_action = np.zeros(66, dtype=np.float32)

    def reset(self):
        self.history = None
        self.last_action[:] = 0

    def contacts(self, data):
        forces = np.zeros((8, 3))
        table_force = np.zeros(3)
        wrench = np.empty(6)
        for i in range(data.ncon):
            contact = data.contact[i]
            a, b = self.model.geom_bodyid[[contact.geom1, contact.geom2]]
            if self.obj not in (a, b):
                continue
            other = b if a == self.obj else a
            mujoco.mj_contactForce(self.model, data, i, wrench)
            # frame rows are contact axes. Sum world vectors per object/link
            # pair before taking norms, matching force_matrix_w aggregation.
            force = contact.frame.reshape(3, 3).T @ wrench[:3]
            force *= -1 if a == self.obj else 1
            if other in self.contact_indices:
                forces[self.contact_indices[other]] += force
            elif other == self.table:
                table_force += force
        return np.linalg.norm(forces, axis=1), float(np.linalg.norm(table_force))

    def proprio(self, data):
        rot = data.xmat[self.pelvis].reshape(3, 3)
        velocity = np.empty(6)
        # mjOBJ_BODY reports velocity at the body's COM in body orientation,
        # matching Isaac Lab root_{lin,ang}_vel_b (COM velocity, actor axes).
        mujoco.mj_objectVelocity(
            self.model, data, mujoco.mjtObj.mjOBJ_BODY, self.pelvis, velocity, 1
        )
        frame = np.concatenate(
            (
                data.qpos[self.qadr] - self.defaults,
                data.qvel[self.vadr],
                rot.T @ np.array([0.0, 0.0, -1.0]),
                velocity[:3],
                velocity[3:],
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
        hand_rot = data.xmat[self.hand[0]].reshape(3, 3)
        obj_pos, obj_rot = data.xpos[self.obj], data.xmat[self.obj].reshape(3, 3)
        half = np.array(self.config["table"]["size"]) / 2
        corners = np.array(
            [[x, y, half[2]] for x in (-half[0], half[0]) for y in (-half[1], half[1])]
        )
        corners = corners @ data.xmat[self.table].reshape(3, 3).T + data.xpos[self.table]
        corners = ((corners - pelvis_pos) @ pelvis_rot).reshape(12)

        def goal(position, quaternion):
            rot = rotation(quaternion)
            return np.concatenate(
                (
                    relative_pose(pelvis_pos, pelvis_rot, position, rot),
                    relative_pose(hand_pos, hand_rot, position, rot),
                )
            )

        release = (
            goal(task.plan.release_position, task.plan.release_quaternion)
            if task.playing
            else np.zeros(18)
        )
        carry = (
            np.zeros(18)
            if task.playing
            else goal(task.plan.carry_position, task.plan.carry_quaternion)
        )
        forces, _ = self.contacts(data)
        sparse, drop = task.command()
        values = np.concatenate(
            (
                self.bps,
                np.zeros(12) if task.playing else corners,
                relative_pose(pelvis_pos, pelvis_rot, obj_pos, obj_rot),
                relative_pose(hand_pos, hand_rot, obj_pos, obj_rot),
                forces,
                release,
                carry,
                sparse,
                [drop],
            )
        ).astype(np.float32)
        if values.shape != (87,) or not np.isfinite(values).all():
            raise ValueError("Invalid task observation")
        return values
