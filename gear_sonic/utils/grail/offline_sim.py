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
from .observations import Observations
from .offline_data import distance_label, object_position_columns
from .offline_task import describe_task
from .plan import CarryPlannerConfig
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


def joint_addresses(model, names):
    ids = np.array([model.joint(name).id for name in names])
    return model.jnt_qposadr[ids], model.jnt_dofadr[ids]


class OfflineEnvironment:
    def __init__(self, model, scene, initial, env_id, output, steps, scene_dir):
        self.model, self.scene, self.initial = model, scene, initial
        self.data = mujoco.MjData(model)
        self.cfg, self.manifest = scene["config"], scene["manifest"]
        self.env_id, self.output, self.steps = env_id, Path(output), steps
        self.scene_dir = scene_dir
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
        self.episode, self.seed, self.step_index = episode, seed, 0
        self.plan, self.reset_details = reset_scene(
            self.model, self.data, self.cfg, self.manifest, self.initial,
            np.random.default_rng(seed), self.planner,
            release_position=self.scene.get("release_position"),
        )
        self.task = CarryTask(self.plan, self.planner.pickup_settle_s)
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
        n = self.steps
        self.trace.update({
            "policy_proprio": np.empty((n, 5, 161), dtype=np.float32),
            "policy_task": np.empty((n, 79), dtype=np.float32),
            "release_requested": np.empty(n, dtype=bool),
            "expert_action": np.empty((n, 66), dtype=np.float32),
            "decoder_observation": np.empty((n, 994), dtype=np.float32),
            "decoder_action": np.empty((n, 29), dtype=np.float32),
            **{f"command_{key}": np.empty((n, 43), dtype=np.float32)
               for key in ("q", "dq", "kp", "kd", "tau")},
        })
        return self.observe()

    def snapshot(self):
        for key in ("qpos", "qvel", "mocap_pos", "mocap_quat"):
            self.trace[key][self.step_index] = getattr(self.data, key)

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
            return dict(env_id=self.env_id, done=True, episode=str(path))
        return self.observe()

    def step(self, reply, action, *, expert=None):
        i = self.step_index
        if (reply["env_id"], reply["episode_id"], reply["step"]) != (self.env_id, self.episode, i):
            raise ValueError("decoder reply does not match environment control boundary")
        self.trace["action"][i] = action
        command = {key: np.asarray(reply[key]) for key in ("q", "dq", "kp", "kd", "tau")}
        if expert is not None:
            self.trace["policy_proprio"][i], self.trace["policy_task"][i] = self.current_observation
            self.trace["release_requested"][i] = self.task.release_requested
            self.trace["expert_action"][i] = expert
            self.trace["decoder_observation"][i] = reply["observation"]
            self.trace["decoder_action"][i] = reply["raw_action"]
            for key, values in command.items():
                self.trace[f"command_{key}"][i] = values
            self.observations.last_action[:] = expert
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
        start = time.perf_counter()
        mujoco.mj_forward(self.model, self.data)
        self.timing["forward_s"] += time.perf_counter() - start
        self.step_index += 1
        if abs(self.data.time - self.step_index / 50) > 1e-8:
            raise RuntimeError("MuJoCo time reset during the trajectory; episode was not saved")
        start = time.perf_counter()
        forces, table = self.observations.contacts(self.data)
        self.task.update(self.step_index, forces.sum(), table)
        self.timing["task_contact_s"] += time.perf_counter() - start
        self.snapshot()

    def save(self):
        start = time.perf_counter()
        result = distance_label(self.trace["qpos"][:, self.object_position_columns], self.plan.final_position)
        directory = self.output / f"episode_{self.episode:06d}"
        directory.mkdir()  # Existing episodes must be explicitly removed before recollecting.
        np.savez_compressed(directory / "trace.npz", **self.trace)
        metadata = dict(
            episode_index=self.episode, env_id=self.env_id, seed=self.seed,
            fps=50, physics_dt=.005, transitions=self.steps, task=self.task_description,
            object_key=self.scene["object_key"],
            scene=relpath(self.scene_dir / "scene.json", directory),
            model=relpath(self.scene_dir / "model.mjb", directory),
            checkpoint=relpath(self.scene_dir / self.scene["checkpoint"], directory),
            decoder=relpath(self.scene_dir / self.scene["decoder"], directory),
            initial_pose=relpath(self.scene_dir / self.scene["initial_pose"], directory),
            task_events=self.task.events, reset=self.reset_details, command_joint_names=COMMAND_NAMES,
            **result,
        )
        (directory / "episode.json").write_text(json.dumps(metadata, indent=2))
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
            else:
                results = [envs[item["env_id"]].advance(item["reply"], item["expert"], item["action"])
                           for item in payload]
            connection.send(results)
    except BaseException:
        connection.send(dict(error=traceback.format_exc()))
    finally:
        connection.close()
