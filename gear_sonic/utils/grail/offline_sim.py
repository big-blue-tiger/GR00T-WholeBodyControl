"""MuJoCo CPU workers. No GPU, DDS, policy, or renderer imports."""

import json
from os.path import relpath
from pathlib import Path
import resource
import time
import traceback

import mujoco
import numpy as np

from .episode import load_initial_pose, reset_scene
from .observations import Observations, rotation
from .offline_data import distance_label, object_position_columns
from .offline_task import describe_task
from .plan import PLAN_FPS, CarryPlan, CarryPlannerConfig
from .task import CarryTask

BODY_NAMES = (
    [f"{s}_{j}_joint" for s in ("left", "right")
     for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
    + [f"waist_{j}_joint" for j in ("yaw", "roll", "pitch")]
    + [f"{s}_{j}_joint" for s in ("left", "right") for j in
       ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
)
HAND_NAMES = [f"{s}_hand_{j}_joint" for s in ("left", "right") for j in
              ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")]
COMMAND_NAMES = BODY_NAMES + HAND_NAMES
COMMAND_KEYS = ("q", "dq", "kp", "kd", "tau")
BOUNDARY_KEYS = ("qpos", "qvel", "mocap_pos", "mocap_quat")
INTEGRATION_STATE = mujoco.mjtState.mjSTATE_INTEGRATION
HANDOFF_STEPS = 10


def joint_addresses(model, names):
    ids = np.array([model.joint(name).id for name in names])
    return model.jnt_qposadr[ids], model.jnt_dofadr[ids]


def recorded_plan(env, metadata, trace):
    model = env.model
    carry, release = [model.body_mocapid[model.body(f"grail_{name}").id]
                      for name in ("carry", "release")]
    carry_quat, release_quat = trace["mocap_quat"][0, [carry, release]]
    turn = rotation(release_quat) @ rotation(carry_quat).T
    angle = float(np.arctan2(turn[1, 0], turn[0, 0]))
    hold_end = round(metadata["reset"]["plan_drop_s"] * PLAN_FPS)
    turn_end = 0 if abs(angle) <= 1e-6 else max(1, round(abs(angle) / env.planner.angular_speed * PLAN_FPS))
    walk_end = hold_end - max(1, round(env.planner.hold_s * PLAN_FPS))
    if not 0 <= turn_end <= walk_end < hold_end:
        raise ValueError("Invalid carry phase lengths in trace")
    sparse = np.zeros((hold_end + 1, 2), dtype=np.float32)
    sparse[:turn_end, 1] = angle
    carry_pos, release_pos = trace["mocap_pos"][0, [carry, release]]
    return CarryPlan(carry_pos, carry_quat, release_pos, release_quat, sparse,
                     np.asarray(metadata["reset"]["base_goal_xy"]), turn_end, walk_end,
                     hold_end, angle, (walk_end - turn_end) * env.planner.walk_speed / PLAN_FPS)


def load_boundary(env, trace, index):
    if "mujoco_state" in trace:
        mujoco.mj_setState(env.model, env.data, trace["mujoco_state"][index], INTEGRATION_STATE)
    else:
        for key in BOUNDARY_KEYS:
            getattr(env.data, key)[:] = trace[key][index]
        env.data.time = trace["sim_time"][index]
    mujoco.mj_forward(env.model, env.data)


class OfflineEnvironment:
    def __init__(self, model, scene, initial, env_id, output, steps, scene_dir, *, release_by_distance=True):
        self.model, self.scene, self.initial = model, scene, initial
        self.data = mujoco.MjData(model)
        self.cfg, self.manifest = scene["config"], scene["manifest"]
        self.env_id, self.output, self.steps = env_id, Path(output), steps
        self.scene_dir = scene_dir
        self.release_by_distance = release_by_distance
        self.recovery = None
        self.observations = Observations(model, self.cfg, self.manifest["bps"])
        self.qadr, self.vadr = joint_addresses(model, COMMAND_NAMES)
        self.object_position_columns = object_position_columns(model)
        joints = [model.joint(name).id for name in COMMAND_NAMES]
        self.actuators = np.array([
            int(np.flatnonzero(model.actuator_trnid[:, 0] == joint)[0]) for joint in joints
        ])
        self.planner = CarryPlannerConfig(**self.cfg["planner"])
        self.timing = dict(physics_s=0., mj_step_s=0., pd_s=0., task_contact_s=0.,
                           observation_s=0., forward_s=0., write_s=0.)

    def start_episode(self, episode, seed):
        self.recovery = None
        self.episode, self.seed, self.step_index = episode, seed, 0
        self.plan, self.reset_details = reset_scene(
            self.model, self.data, self.cfg, self.manifest, self.initial,
            np.random.default_rng(seed), self.planner,
            release_position=self.scene.get("release_position"),
        )
        self.task = CarryTask(
            self.plan, self.planner.pickup_settle_s, release_by_distance=self.release_by_distance,
        )
        self.task_description = describe_task(
            self.scene["task"], self.scene["object_key"], self.plan.sparse[0, 1]
        )
        self.observations.reset()
        n = self.steps
        self.trace = {
            "qpos": np.empty((n + 1, self.model.nq)),
            "qvel": np.empty((n + 1, self.model.nv)),
            "mocap_pos": np.empty((n + 1, self.model.nmocap, 3)),
            "mocap_quat": np.empty((n + 1, self.model.nmocap, 4)),
            "sim_time": np.arange(n + 1, dtype=np.float64) / 50,
            "action": np.empty((n, 78), dtype=np.float32),
        }
        self.snapshot()

    def reset(self, episode, seed):
        self.start_episode(episode, seed)
        self.initialize_policy_trace()
        return self.observe()

    def initialize_policy_trace(self):
        n = self.steps
        shapes = dict(policy_proprio=(5, 161), policy_task=(79,), expert_action=(66,),
                      decoder_observation=(994,), decoder_action=(29,))
        shapes.update({f"command_{key}": (len(COMMAND_NAMES),) for key in COMMAND_KEYS})
        for key, shape in shapes.items():
            if key not in self.trace:
                self.trace[key] = np.full((n, *shape), np.nan, dtype=np.float32)
        self.trace.setdefault("release_requested", np.zeros(n, dtype=bool))

    def start_replay(self, metadata, trace):
        self.step_index = 0
        mujoco.mj_resetData(self.model, self.data)
        load_boundary(self, trace, 0)
        self.plan = recorded_plan(self, metadata, trace)
        self.task = CarryTask(
            self.plan, self.planner.pickup_settle_s,
            release_by_distance=metadata.get("release_by_distance", False),
        )

    def restart_pickup(self):
        self.task = CarryTask(
            self.plan, self.planner.pickup_settle_s, release_by_distance=self.task.release_by_distance,
        )
        self.task.steps = self.step_index

    def record_integration_state(self):
        self.trace["mujoco_state"] = np.empty((
            self.steps + 1, mujoco.mj_stateSize(self.model, INTEGRATION_STATE),
        ))
        self.snapshot()

    def snapshot(self):
        for key in BOUNDARY_KEYS:
            self.trace[key][self.step_index] = getattr(self.data, key)
        if "mujoco_state" in self.trace:
            mujoco.mj_getState(self.model, self.data, self.trace["mujoco_state"][self.step_index],
                               INTEGRATION_STATE)

    def decoder_request(self, angular=None):
        if angular is None:
            angular, _ = self.observations.root_velocity(self.data)
            angular = angular.astype(np.float32)
        return dict(
            env_id=self.env_id, episode_id=self.episode, step=self.step_index,
            q=self.data.qpos[self.qadr[:29]].tolist(),
            dq=self.data.qvel[self.vadr[:29]].tolist(),
            quat=self.data.xquat[self.observations.pelvis].tolist(),
            angular=angular.tolist(), hand_q=self.data.qpos[self.qadr[29:]].tolist(),
        )

    def observe(self):
        start = time.perf_counter()
        proprio = self.observations.proprio(self.data)
        task = self.observations.privileged(self.data, self.task)
        self.current_observation = proprio, task
        request = self.decoder_request(proprio[-1, 89:92])
        self.timing["observation_s"] += time.perf_counter() - start
        return dict(env_id=self.env_id, episode_id=self.episode, proprio=proprio, task=task, request=request,
                    release_requested=bool(self.task.release_requested))

    def advance(self, reply, expert, action):
        self.step(reply, action, expert=expert)
        if self.step_index == self.steps:
            path = self.save()
            start = 0 if self.recovery is None else self.recovery["step"]
            return dict(env_id=self.env_id, done=True, episode=str(path), transitions=self.steps - start)
        return self.observe()

    def step(self, reply, action, *, expert=None):
        i = self.step_index
        if (reply["env_id"], reply["episode_id"], reply["step"]) != (self.env_id, self.episode, i):
            raise ValueError("decoder reply does not match environment control boundary")
        self.trace["action"][i] = action
        command = {key: np.asarray(reply[key]) for key in COMMAND_KEYS}
        if self.recovery is not None:
            if not all(np.isfinite(values).all() for values in (*command.values(), action, reply["raw_action"])):
                raise ValueError(f"episode {self.episode}, step {i}: nonfinite recovery action or command")
            progress = min((i - self.recovery["step"]) / HANDOFF_STEPS, 1.)
            weight = progress**2 * (3. - 2. * progress)
            self.trace["recovery_weight"][i] = weight
            if weight < 1.:
                command = {key: (1. - weight) * self.handoff_command[key] + weight * values
                           for key, values in command.items()}
        if expert is not None:
            self.trace["policy_proprio"][i], self.trace["policy_task"][i] = self.current_observation
            self.trace["release_requested"][i] = self.task.release_requested
            self.trace["expert_action"][i] = expert
            self.trace["decoder_observation"][i] = reply["observation"]
            self.observations.last_action[:] = expert
        if "decoder_action" in self.trace:
            self.trace["decoder_action"][i] = reply["raw_action"]
        if "command_q" in self.trace:
            for key, values in command.items():
                self.trace[f"command_{key}"][i] = values
        self.advance_physics(command)
        if self.recovery is not None and (
            not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all()
        ):
            raise RuntimeError(f"episode {self.episode}, step {i}: invalid MuJoCo state")
        self.update_task()
        self.snapshot()

    def advance_physics(self, command, *, forward=True):
        start = time.perf_counter()
        for _ in range(4):
            substep = time.perf_counter()
            torques = (command["tau"] + command["kp"] * (command["q"] - self.data.qpos[self.qadr])
                       + command["kd"] * (command["dq"] - self.data.qvel[self.vadr]))
            self.data.ctrl[self.actuators] = np.clip(
                torques, self.model.actuator_ctrlrange[self.actuators, 0],
                self.model.actuator_ctrlrange[self.actuators, 1],
            )
            integrated = time.perf_counter()
            self.timing["pd_s"] += integrated - substep
            mujoco.mj_step(self.model, self.data)
            self.timing["mj_step_s"] += time.perf_counter() - integrated
        self.timing["physics_s"] += time.perf_counter() - start
        if forward:
            start = time.perf_counter()
            mujoco.mj_forward(self.model, self.data)
            self.timing["forward_s"] += time.perf_counter() - start
        self.step_index += 1
        if abs(self.data.time - self.step_index / 50) > 1e-8:
            raise RuntimeError("MuJoCo time reset during the trajectory; episode was not saved")

    def update_task(self):
        start = time.perf_counter()
        contacts = self.observations.contact_forces(self.data)
        forces, table = contacts
        self.task.update(
            self.step_index, np.linalg.norm(forces, axis=1).sum(), np.linalg.norm(table),
            self.data.xpos[self.observations.obj],
        )
        self.timing["task_contact_s"] += time.perf_counter() - start
        return contacts

    def save(self):
        start = time.perf_counter()
        result = distance_label(self.trace["qpos"][:, self.object_position_columns], self.plan.final_position)
        directory = self.output / f"episode_{self.episode:06d}"
        directory.mkdir(exist_ok=self.recovery is not None)
        np.savez_compressed(directory / "trace.npz", **self.trace)
        if self.recovery is None:
            metadata = dict(
                episode_index=self.episode, env_id=self.env_id, seed=self.seed,
                fps=50, physics_dt=.005, transitions=self.steps, task=self.task_description,
                object_key=self.scene["object_key"],
                scene=relpath(self.scene_dir / "scene.json", directory),
                model=relpath(self.scene_dir / "model.mjb", directory),
                checkpoint=relpath(self.scene_dir / self.scene["checkpoint"], directory),
                decoder=relpath(self.scene_dir / self.scene["decoder"], directory),
                initial_pose=relpath(self.scene_dir / self.scene["initial_pose"], directory),
                release_by_distance=self.task.release_by_distance,
                task_events=self.task.events, reset=self.reset_details, command_joint_names=COMMAND_NAMES,
                **result,
            )
        else:
            if self.recovery["regrasp"]:
                self.recovery["task_events"] = self.task.events
            metadata = dict(
                self.source_metadata, recovery=self.recovery, **result,
                task_events=self.task.events | self.prefix_task_events,
            )
        pending_metadata = directory / "episode.json.tmp"
        pending_metadata.write_text(json.dumps(metadata, indent=2))
        pending_metadata.replace(directory / "episode.json")
        self.timing["write_s"] += time.perf_counter() - start
        self.trace = None
        return directory


def load_environment_scene(scene_dir):
    scene = json.loads((scene_dir / "scene.json").read_text())
    model = mujoco.MjModel.from_binary_path(str(scene_dir / "model.mjb"))
    initial = load_initial_pose(scene_dir / scene["initial_pose"])
    return model, scene, initial


def simulation_worker(connection, options):
    try:
        envs = {}
        cpu_start = time.process_time()
        while True:
            try:
                operation, payload = connection.recv()
            except EOFError:
                break
            if operation == "close":
                totals = {k: sum(env.timing[k] for env in envs.values()) for k in next(iter(envs.values())).timing}
                connection.send(dict(timing=totals, cpu_s=time.process_time() - cpu_start,
                                     peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
                break
            if operation == "reset":
                scenes = {env.scene_dir: (env.model, env.scene, env.initial) for env in envs.values()}
                results = []
                for env_id, episode, seed, directory in payload:
                    scene_dir = Path(directory)
                    previous = envs.get(env_id)
                    if previous is None or previous.scene_dir != scene_dir:
                        if scene_dir not in scenes:
                            scenes[scene_dir] = load_environment_scene(scene_dir)
                        env = OfflineEnvironment(
                            *scenes[scene_dir], env_id, options["output"], options["steps"], scene_dir,
                        )
                        if previous is not None:
                            env.timing = previous.timing
                        envs[env_id] = env
                    results.append(envs[env_id].reset(episode, seed))
                scenes.clear()
            elif operation == "recover":
                from .offline_recovery import restore_episode

                scenes = {env.recording_assets: (env.model, env.scene) for env in envs.values()}
                results = []
                for job in payload:
                    env, observation = restore_episode(job, scenes, options["output"], options["hand_primitives"])
                    previous = envs.get(env.env_id)
                    if previous is not None:
                        for key, elapsed in previous.timing.items():
                            env.timing[key] += elapsed
                    envs[env.env_id] = env
                    results.append(observation)
            elif operation == "step":
                results = [envs[item["env_id"]].advance(item["reply"], item["expert"], item["action"])
                           for item in payload]
            else:
                raise ValueError(f"unknown offline simulation operation: {operation}")
            connection.send(results)
    except BaseException:
        connection.send(dict(error=traceback.format_exc()))
    finally:
        connection.close()
