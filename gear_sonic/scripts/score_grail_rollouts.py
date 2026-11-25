"""Replay GRAIL and GR00T traces to score the executed 50 Hz transitions."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import mujoco
import numpy as np

from gear_sonic.utils.grail.carry_reward_value import SCORE_FIELDS, CarryRewardValueScorer, CarryRewardValueStream
from gear_sonic.utils.grail.checkpoints import DEFAULT_CHECKPOINT
from gear_sonic.utils.grail.gr00t_recovery import RecoveryConfig, RecoveryGate
from gear_sonic.utils.grail.observations import rotation
from gear_sonic.utils.grail.offline_data import episodes, load_episode
from gear_sonic.utils.grail.offline_sim import COMMAND_NAMES, OfflineEnvironment
from gear_sonic.utils.grail.plan import PLAN_FPS, CarryPlan
from gear_sonic.utils.grail.task import CarryTask

COMMAND_KEYS = ("q", "dq", "kp", "kd", "tau")


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
    for key in ("qpos", "qvel", "mocap_pos", "mocap_quat"):
        getattr(env.data, key)[:] = trace[key][index]
    mujoco.mj_forward(env.model, env.data)


def replay(env, metadata, trace, scorer):
    if list(metadata["command_joint_names"]) != COMMAND_NAMES:
        raise ValueError("Unsupported command joint order")
    commands = {key: trace[f"command_{key}"] for key in COMMAND_KEYS}
    for key, values in commands.items():
        if values.shape != (env.steps, len(COMMAND_NAMES)):
            raise ValueError(f"Invalid command_{key} shape: {values.shape}")
    env.step_index = 0
    mujoco.mj_resetData(env.model, env.data)
    load_boundary(env, trace, 0)
    env.task = CarryTask(recorded_plan(env, metadata, trace), env.planner.pickup_settle_s)
    stream = CarryRewardValueStream(scorer, env.steps / 50)
    stream.bind(env.model, env.data, env.observations, env.qadr)
    stream.advance(env.task, env.observations.contact_forces(env.data))
    gate = RecoveryGate(RecoveryConfig())
    decision = None

    obs, obj, plan = env.observations, env.object_position_columns, env.task.plan
    n = env.steps
    result = {key: np.empty(n, dtype=np.float32) for key in SCORE_FIELDS}
    result["time_s"] = trace["sim_time"][1:]
    for key in ("hand_distance_m", "upright", "object_speed_m_s"):
        result[key] = np.empty(n, dtype=np.float32)
    for key in ("picked_up", "carrying", "dropped", "release_requested"):
        result[key] = np.empty(n, dtype=bool)
    result["pelvis_pos"] = trace["qpos"][1:, :3]
    result["target_distance_m"] = np.linalg.norm(trace["qpos"][1:, obj] - plan.final_position, axis=1)
    result["goal_plan"] = plan.final_position
    for key in ("observation_step", "chunk_index", "held_action"):
        if key in trace:
            result[key] = trace[key]
    for i in range(n):
        command = {key: values[i] for key, values in commands.items()}
        env.advance_physics(command, forward=False)
        # Use recorded state boundaries even if MuJoCo's solver warm start differs.
        load_boundary(env, trace, i + 1)
        contacts = env.update_task()
        stream.set_target(command["q"])
        score = stream.advance(env.task, contacts, timeout=i + 1 == n)
        for key in SCORE_FIELDS:
            result[key][i] = getattr(score, key)
        object_position = env.data.xpos[obs.obj]
        if decision is None and i + 1 < n:
            decision = gate.update(
                score.td_advantage, release_requested=env.task.release_requested,
                target_distance_m=np.linalg.norm(object_position - plan.final_position),
            )
            if decision is not None:
                decision.update(step=i + 1, time_s=(i + 1) * env.task.dt)
        result["hand_distance_m"][i] = np.linalg.norm(env.data.xpos[obs.hand].mean(axis=0) - object_position)
        result["upright"][i] = env.data.xmat[obs.pelvis, 8]
        result["object_speed_m_s"][i] = np.linalg.norm(stream.velocity(obs.obj)[3:])
        for key in ("picked_up", "carrying", "dropped", "release_requested"):
            result[key][i] = getattr(env.task, key)
    return result, decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--episode", type=int, action="append", help="Score only these source episode indices")
    args = parser.parse_args()
    jobs = defaultdict(list)
    for root in args.traces:
        for path in episodes(root):
            metadata = json.loads(path.read_text())
            if args.episode and metadata["episode_index"] not in args.episode:
                continue
            trace_path = path.with_name("trace.npz")
            with np.load(trace_path, allow_pickle=False) as archive:
                missing = [f"command_{key}" for key in COMMAND_KEYS if f"command_{key}" not in archive.files]
            if missing:
                parser.error(f"{trace_path} is missing recorded decoder commands: {', '.join(missing)}")
            scene = (path.parent / metadata["scene"]).resolve()
            model = (path.parent / metadata["model"]).resolve()
            jobs[(root.resolve(), scene, model)].append(path)
    if not jobs:
        parser.error("no matching episodes")
    args.output.mkdir(parents=True, exist_ok=True)
    scorer = CarryRewardValueScorer(args.checkpoint)
    print(f"[Score] {scorer.path} | gamma={scorer.gamma:g}", flush=True)
    multiple = len(args.traces) > 1
    with (args.output / "episodes.jsonl").open("x") as summaries:
        episode_id = 0
        for (root, scene_path, model_path), paths in jobs.items():
            scene = json.loads(scene_path.read_text())
            model = mujoco.MjModel.from_binary_path(str(model_path))
            name = root.parent.name if root.name == "traces" else root.name
            directory = args.output / name if multiple else args.output
            directory.mkdir(parents=True, exist_ok=True)
            for path in paths:
                metadata, trace = load_episode(path)
                env = OfflineEnvironment(model, scene, None, 0, root, metadata["transitions"], scene_path.parent)
                result, decision = replay(env, metadata, trace, scorer)
                target = directory / f"{path.parent.name}.npz"
                np.savez_compressed(target, **result)
                summary = dict(trace=str(path.resolve()), score=str(target.resolve()),
                               checkpoint=str(scorer.path), gamma=scorer.gamma, recovery_decision=decision)
                summaries.write(json.dumps(summary) + "\n")
                summaries.flush()
                episode_id += 1
                print(f"[Score] {episode_id} {name}/{path.parent.name}", flush=True)


if __name__ == "__main__":
    main()
