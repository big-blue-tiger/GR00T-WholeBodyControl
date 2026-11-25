"""GRAIL CarryRewardValue inference from live MuJoCo state and DDS joint targets."""

from dataclasses import dataclass
import time

import mujoco
import numpy as np
import torch
from torch import nn
import yaml

from .checkpoints import resolve_checkpoint
from .observations import relative_pose, rotation, table_edge_line
from .policy import load_checkpoint

# Field order and widths from GRAIL's CarryObservation, independent of actor input.
PROPRIO_FIELDS = (
    ("joint_pos_rel", 43), ("joint_vel_rel", 43), ("projected_gravity", 3),
    ("base_ang_vel", 3), ("base_lin_vel", 3), ("last_joint_target", 43),
)
HISTORY_LENGTH = 5
OBSERVATION_FIELDS = (
    ("proprio_history", 690), ("body_pos", 42), ("body_ori", 84),
    ("object_pos_root", 3), ("object_pose_pelvis", 9), ("object_pose_hand", 9),
    ("object_bps", 10), ("object_lin_vel", 3), ("object_ang_vel", 3),
    ("table_pose", 9), ("table_edge_line", 12), ("table_size", 3),
    ("carry_goal", 18), ("release_goal", 18), ("hand_forces", 24),
    ("hand_object_contact", 1), ("object_table_contact", 1), ("robot_table_contact", 1),
    ("sparse_command", 2), ("base_goal_xy", 2), ("picked_up", 1), ("plan_started", 1),
    ("dropped", 1), ("success", 1), ("plan_time", 1), ("remaining_time", 1),
)
ACTION_FIELDS = (("joint_target", 43),)
OBSERVATION_DIM = sum(size for _, size in OBSERVATION_FIELDS)
REWARD_INPUT_DIM = OBSERVATION_DIM + ACTION_FIELDS[0][1]
SCORE_FIELDS = ("reward", "value", "next_value", "td_advantage")


def _prediction_head(input_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 512), nn.SiLU(), nn.Linear(512, 256), nn.SiLU(),
        nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 1),
    )


class CarryRewardValue(nn.Module):
    def __init__(self):
        super().__init__()
        self.reward = _prediction_head(REWARD_INPUT_DIM)
        self.value = _prediction_head(OBSERVATION_DIM)


@dataclass(frozen=True)
class CarryScore:
    reward: float
    value: float
    next_value: float
    td_advantage: float


class CarryRewardValueScorer:
    """Load the two heads and share them across independent episode streams."""

    def __init__(self, checkpoint):
        path = resolve_checkpoint(checkpoint)
        carry = load_checkpoint(path).get("carry_reward_value")
        if carry is None:
            raise ValueError(f"Checkpoint has no CarryRewardValue weights: {path}")
        layout = carry["layout"]
        expected = dict(
            observation_fields=OBSERVATION_FIELDS, proprio_fields=PROPRIO_FIELDS,
            proprio_history=HISTORY_LENGTH, action_fields=ACTION_FIELDS, table_edge="-Y",
        )
        if any(layout.get(name) != value for name, value in expected.items()):
            raise ValueError(f"Unsupported CarryRewardValue observation layout: {path}")
        config = yaml.safe_load(path.with_name("config.yaml").read_text())
        self.gamma = float(config["algo"]["config"]["gamma"])
        self.path, self.layout = path, layout
        self.net = CarryRewardValue().eval()
        self.net.load_state_dict(carry["models"])
        torch.set_num_threads(1)
        self.default_pos = np.asarray(layout["default_joint_pos"], dtype=np.float32)
        self.default_vel = np.asarray(layout["default_joint_vel"], dtype=np.float32)

    def score(self, state, target, value, bootstrap):
        reward_input = torch.cat((state, torch.from_numpy(target)))
        reward = self.net.reward(reward_input).item()
        return CarryScore(reward, value, bootstrap, reward + self.gamma * bootstrap - value)


class CarryRewardValueStream:
    """Pack state and score completed transitions at the 50 Hz boundary."""

    def __init__(self, scorer, duration):
        self.scorer, self.duration = scorer, duration
        self.reset()

    def bind(self, model, data, observations, command_qadr):
        layout = self.scorer.layout
        self.model, self.data = model, data
        self.observations = observations
        joints = np.array([self.model.joint(name).id for name in layout["joint_names"]])
        self.qadr, self.vadr = self.model.jnt_qposadr[joints], self.model.jnt_dofadr[joints]
        command_index = {address: i for i, address in enumerate(command_qadr)}
        self.command_order = [command_index[address] for address in self.qadr]
        self.links = [self.model.body(name).id for name in layout["body_names"]]
        self.hand = [self.model.body(name).id for name in layout["hand_frame_links"]]
        self.contact_order = [
            observations.contact_indices[self.model.body(name).id] for name in layout["contact_links"]
        ]
        self.robot_bodies = self.model.body_rootid == self.model.body_rootid[observations.pelvis]
        self.table_size = np.asarray(observations.config["table"]["size"])

    def reset(self):
        self.history = self.state = self.value = self.target = None
        self.success = False

    def set_target(self, command):
        self.target = np.asarray(command, dtype=np.float32)[self.command_order]

    def robot_table_contact(self):
        obs, data, model = self.observations, self.data, self.model
        forces = np.zeros((model.nbody, 3))
        wrench = np.empty(6)
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            bodies = [model.geom_bodyid[geom] if geom >= 0 else 0 for geom in contact.geom]
            for side, body in enumerate(bodies):
                if not self.robot_bodies[body]:
                    continue
                is_table = bodies[1 - side] == obs.table
                is_table_flex = obs.table_flex >= 0 and contact.flex[1 - side] == obs.table_flex
                if is_table or is_table_flex:
                    mujoco.mj_contactForce(model, data, contact_id, wrench)
                    forces[body] += (2 * side - 1) * wrench[0] * contact.frame[:3]
        return np.linalg.norm(forces, axis=1).sum() > 0.01

    def velocity(self, body):
        velocity = np.empty(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0)
        return velocity

    def observe(self, task, contacts):
        data, obs = self.data, self.observations
        pelvis_pos, pelvis_rot = data.xpos[obs.pelvis], data.xmat[obs.pelvis].reshape(3, 3)
        hand_pos, hand_rot = data.xpos[self.hand].mean(axis=0), data.xmat[self.hand[0]].reshape(3, 3)
        obj_pos, obj_rot = data.xpos[obs.obj], data.xmat[obs.obj].reshape(3, 3)
        table_pos, table_rot = data.xpos[obs.table], data.xmat[obs.table].reshape(3, 3)
        root_velocity, obj_velocity = self.velocity(obs.pelvis), self.velocity(obs.obj)
        frame = np.concatenate((
            data.qpos[self.qadr] - self.scorer.default_pos, data.qvel[self.vadr] - self.scorer.default_vel,
            -pelvis_rot.T[:, 2], pelvis_rot.T @ root_velocity[:3], pelvis_rot.T @ root_velocity[3:],
            self.scorer.default_pos if self.target is None else self.target,
        )).astype(np.float32)
        if self.history is None:
            self.history = np.tile(frame, (HISTORY_LENGTH, 1))
        else:
            self.history[:-1] = self.history[1:]
            self.history[-1] = frame

        def goal(position, quaternion):
            orientation = rotation(quaternion)
            return np.r_[relative_pose(pelvis_pos, pelvis_rot, position, orientation),
                         relative_pose(hand_pos, hand_rot, position, orientation)]

        plan = task.plan
        self.success |= task.playing and np.linalg.norm(obj_pos - plan.final_position) < 0.1
        base_goal = np.r_[plan.base_goal_xy - pelvis_pos[:2], 0.0]
        forces, table_force = contacts
        link_rotation = data.xmat[self.links].reshape(-1, 3, 3)
        features = np.concatenate((
            self.history.ravel(), ((data.xpos[self.links] - pelvis_pos) @ pelvis_rot).ravel(),
            (pelvis_rot.T @ link_rotation)[:, :, :2].ravel(), pelvis_rot.T @ (obj_pos - pelvis_pos),
            relative_pose(pelvis_pos, pelvis_rot, obj_pos, obj_rot),
            relative_pose(hand_pos, hand_rot, obj_pos, obj_rot), obs.bps,
            pelvis_rot.T @ obj_velocity[3:], pelvis_rot.T @ obj_velocity[:3],
            relative_pose(pelvis_pos, pelvis_rot, table_pos, table_rot),
            table_edge_line(table_pos, table_rot, self.table_size, pelvis_pos, pelvis_rot, hand_pos, hand_rot),
            self.table_size, goal(plan.carry_position, plan.carry_quaternion),
            goal(plan.final_position, plan.final_quaternion), (forces[self.contact_order] @ pelvis_rot).ravel(),
            [np.any(np.linalg.norm(forces, axis=1) > 0.01), np.linalg.norm(table_force) > 0.01,
             self.robot_table_contact()],
            task.command(pelvis_pos), (pelvis_rot.T @ base_goal)[:2],
            [task.picked_up, task.playing, task.dropped, self.success,
             (task.steps - task.start_step) * task.dt if task.playing else 0.0,
             max(0.0, self.duration - data.time)],
        )).astype(np.float32)
        return torch.from_numpy(features)

    @torch.inference_mode()
    def advance(self, task, contacts, *, timeout=False):
        state = self.observe(task, contacts)
        value = self.scorer.net.value(state).item()
        score = self.finish(self.value if timeout else value)
        self.state, self.value = state, value
        return score

    @torch.inference_mode()
    def finish(self, bootstrap):
        if self.state is not None and self.target is not None:
            return self.scorer.score(self.state, self.target, self.value, bootstrap)


class CarryRewardValueMonitor(CarryRewardValueStream):
    """Score each live control interval using its last applied DDS position target."""

    def __init__(self, checkpoint, env, observations, duration):
        scorer = CarryRewardValueScorer(checkpoint)
        super().__init__(scorer, duration)
        self.env = env
        self.bind_scene(observations)
        self.last_report = -float("inf")
        print(f"[CarryRewardValue] Checkpoint {scorer.path} | gamma={scorer.gamma:g}", flush=True)

    def bind_scene(self, observations):
        env = self.env
        self.bind(env.mj_model, env.mj_data, observations,
                  np.r_[env.body_qpos_adr, env.left_hand_qpos_adr, env.right_hand_qpos_adr])

    def capture_target(self):
        # Read under the same DDS locks as the torque calculation for this substep.
        self.set_target(np.r_[self.env.compute_body_qpos(), self.env.compute_hand_qpos()])

    def sample(self, episode, step, task, contacts):
        self.report(episode, step, self.advance(task, contacts))

    def finish_episode(self, episode, step, *, timeout):
        # Match GRAIL's PPO timeout correction. A fall is terminal.
        self.report(episode, step, self.finish(self.value if timeout else 0.0))

    def report(self, episode, step, score):
        now = time.monotonic()
        if score is not None and now - self.last_report >= 0.1:
            print(
                f"[CarryRewardValue | Episode {episode} | Step {step}] "
                f"Reward: {score.reward:.9g} | Value: {score.value:.9g} | TD Advantage: {score.td_advantage:.9g}",
                flush=True,
            )
            self.last_report = now
