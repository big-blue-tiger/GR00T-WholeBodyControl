"""Continue scored GR00T episodes with the offline GRAIL collector."""

import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
from zipfile import BadZipFile

import mujoco
import numpy as np

from .checkpoints import resolve_checkpoint
from .offline_collect import run_collection
from .offline_data import episodes, load_episode, score_rows, update_episode_metadata
from .offline_sim import (
    BOUNDARY_KEYS,
    COMMAND_KEYS,
    COMMAND_NAMES,
    HANDOFF_STEPS,
    INTEGRATION_STATE,
    OfflineEnvironment,
    load_boundary,
)

DECODER_HISTORY = 10
GRAIL_HISTORY = 5
REPLAY_QPOS_TOLERANCE = 1e-4
REPLAY_QVEL_TOLERANCE = 1e-2
EPISODE_PATH_KEYS = ("scene", "model", "initial_pose", "checkpoint", "decoder")


def takeover_step(metadata, trace, decision, score):
    step = decision["step"]
    n = metadata["transitions"]
    if type(step) is not int or not 0 < step < n:
        raise ValueError(f"invalid takeover step {step!r} for {n} transitions")
    time = trace["sim_time"][step]
    if not np.isclose(decision["time_s"], time, rtol=0, atol=1e-6):
        raise ValueError(f"takeover step {step} has timestamp {time:g}, expected {decision['time_s']!r}")
    with np.load(score, allow_pickle=False) as archive:
        times = archive["time_s"]
        if times.shape != (n,) or not np.isclose(times[step - 1], time, rtol=0, atol=1e-6):
            raise ValueError(f"score timestamps do not match the trace at step {step}: {score}")
        if "pelvis_pos" in archive:
            pelvis = archive["pelvis_pos"]
            if pelvis.shape != (n, 3) or not np.allclose(pelvis[step - 1], trace["qpos"][step, :3],
                                                        rtol=0, atol=1e-6):
                raise ValueError(f"score pelvis position does not match the trace at step {step}: {score}")
    return step


def validate_history(env, metadata, trace, step):
    n, model = env.steps, env.model
    if (metadata["fps"], metadata["physics_dt"]) != (50, .005) or model.opt.timestep != .005:
        raise ValueError("expected 50 Hz control and 5 ms physics steps")
    if metadata["command_joint_names"] != COMMAND_NAMES:
        raise ValueError("unsupported command joint order")
    shapes = dict(qpos=(n + 1, model.nq), qvel=(n + 1, model.nv),
                  mocap_pos=(n + 1, model.nmocap, 3), mocap_quat=(n + 1, model.nmocap, 4),
                  action=(n, 78))
    shapes.update({f"command_{key}": (n, len(COMMAND_NAMES)) for key in COMMAND_KEYS})
    if "mujoco_state" in trace:
        shapes["mujoco_state"] = n + 1, mujoco.mj_stateSize(model, INTEGRATION_STATE)
    if "decoder_action" in trace:
        shapes["decoder_action"] = n, 29
    for key, shape in shapes.items():
        if key not in trace or trace[key].shape != shape:
            raise ValueError(f"missing or invalid {key}, expected shape {shape}")
        end = step + (shape[0] == n + 1)
        if not np.isfinite(trace[key][:end]).all():
            raise ValueError(f"nonfinite {key} before or at takeover step {step}")


def grail_previous_action(action, hand_primitives):
    # Pick the center of the recorded FSQ bin. A GR00T hand target maps to
    # the nearest of the GRAIL open/closed primitives in decoder joint order.
    half = 31 * (1 + 1e-3) / 2
    bins = np.clip(np.rint(action[:64] * 16), -16, 15)
    latent = np.arctanh((bins + .5) / half) - np.arctanh(.5 / half)
    distance = np.square(hand_primitives - action[64:].reshape(2, 7)).sum(axis=-1)
    fingers = np.where(distance[1] < distance[0], 1., -1.)
    return np.concatenate((latent, fingers)).astype(np.float32)


def decoder_previous(trace, index):
    if index == 0:
        return dict(raw=[0.] * 29)
    if "decoder_action" in trace:
        return dict(raw=trace["decoder_action"][index - 1].tolist())
    return dict(targets=trace["command_q"][index - 1, :29].tolist())


def restore_history(env, metadata, trace, step, hand_primitives):
    env.start_replay(metadata, trace)
    env.observations.reset()
    exact = "mujoco_state" in trace
    decoder_history = []
    decoder_start = max(0, step - DECODER_HISTORY + 1)
    grail_start = max(0, step - GRAIL_HISTORY + 1)
    for index in range(step + 1):
        if index:
            if exact:
                env.step_index = index
            else:
                command = {key: trace[f"command_{key}"][index - 1] for key in COMMAND_KEYS}
                env.advance_physics(command, forward=False)
                if index == step:
                    qpos_error = np.max(np.abs(env.data.qpos - trace["qpos"][index]))
                    qvel_error = np.max(np.abs(env.data.qvel - trace["qvel"][index]))
                    if not (qpos_error <= REPLAY_QPOS_TOLERANCE and qvel_error <= REPLAY_QVEL_TOLERANCE):
                        print(
                            f"[Recovery] WARNING: episode_{env.episode:06d}: legacy replay diverged at step {step}: "
                            f"qpos error {qpos_error:.6g}, qvel error {qvel_error:.6g}",
                            file=sys.stderr, flush=True,
                        )
            load_boundary(env, trace, index)
            env.update_task()
        if decoder_start <= index < step:
            request = env.decoder_request()
            frame = {key: request[key] for key in ("q", "dq", "quat", "angular")}
            frame["previous"] = decoder_previous(trace, index)
            decoder_history.append(frame)
        if index >= grail_start:
            env.observations.last_action[:] = (
                grail_previous_action(trace["action"][index - 1], hand_primitives) if index else 0.
            )
            if index < step:
                env.observations.proprio(env.data)
    if not all(np.array_equal(getattr(env.data, key), trace[key][step]) for key in BOUNDARY_KEYS):
        raise ValueError(f"restored integration state does not match the recorded boundary at step {step}")
    if abs(env.data.time - trace["sim_time"][step]) > 1e-8:
        raise ValueError(f"restored simulation time does not match the recorded boundary at step {step}")
    return decoder_history


def restore_episode(job, scenes, output, hand_primitives):
    path = Path(job["path"])
    try:
        metadata, trace = load_episode(path)
        step = job["step"]
        scene_path = (path.parent / metadata["scene"]).resolve()
        model_path = (path.parent / metadata["model"]).resolve()
        assets = scene_path, model_path
        if assets not in scenes:
            scenes[assets] = (
                mujoco.MjModel.from_binary_path(str(model_path)),
                json.loads(scene_path.read_text()),
            )
        model, scene = scenes[assets]
        env = OfflineEnvironment(model, scene, None, job["env_id"], output, metadata["transitions"],
                                 scene_path.parent)
        env.recording_assets = assets
        env.episode = metadata["episode_index"]
        validate_history(env, metadata, trace, step)
        history = restore_history(env, metadata, trace, step, hand_primitives)
        previous = decoder_previous(trace, step)
        env.handoff_command = {key: trace[f"command_{key}"][step - 1].copy() for key in COMMAND_KEYS}
        env.source_metadata = job["output_metadata"]
        env.prefix_task_events = env.task.events.copy()
        regrasp = env.task.dropped
        if regrasp:
            env.restart_pickup()
        destination = env.output / path.parent.name
        env.recovery = dict(
            job["decision"], regrasp=bool(regrasp),
            state_restore="integration_state" if "mujoco_state" in trace else "command_replay",
            handoff_steps=HANDOFF_STEPS,
            checkpoint=os.path.relpath(job["checkpoint"], destination),
            decoder=os.path.relpath(job["decoder"], destination),
        )
        env.trace = trace
        env.initialize_policy_trace()
        env.trace["recovery_weight"] = np.zeros(env.steps, dtype=np.float32)
        for key, sentinel in (("observation_step", -1), ("chunk_index", -1), ("held_action", False)):
            if key in env.trace:
                env.trace[key][step:] = sentinel
        observation = env.observe()
        observation["request"].update(history=history, previous=previous)
        return env, observation
    except (KeyError, OSError, ValueError, RuntimeError) as error:
        raise ValueError(f"{path.parent.name}: {error}") from error


def recovery_workspace(args, output):
    pending = output.with_name(f".{output.name}.partial")
    existing = [path for path in (output, pending) if os.path.lexists(path)]
    if args.resume_from is not None:
        legacy = args.resume_from
        if (
            legacy.parent.resolve() != output.parent or not legacy.name.startswith(".traces_recovery-")
            or legacy.is_symlink() or not legacy.is_dir()
        ):
            raise ValueError("--resume-from requires a legacy .traces_recovery-* directory next to the output")
        if existing:
            raise FileExistsError(f"{existing[0]} already exists; use --resume or choose a new --output-dir")
        legacy.rename(pending)
        return pending
    if existing and not args.resume:
        raise FileExistsError(f"{existing[0]} already exists; use --resume or choose a new --output-dir")
    if len(existing) > 1:
        raise FileExistsError(f"both {output} and {pending} exist; choose which directory to resume")
    workspace = existing[0] if existing else pending
    if os.path.lexists(workspace) and (workspace.is_symlink() or not workspace.is_dir()):
        raise ValueError(f"expected a recovery directory: {workspace}")
    return workspace


def completed_episode(path, original, decision):
    """A resume candidate must contain a readable trace for the expected episode and decision."""
    if path.parent.is_symlink() or (path.parent.exists() and not path.parent.is_dir()):
        raise ValueError(f"expected an episode directory: {path.parent}")
    try:
        metadata, _ = load_episode(path)
    except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile):
        return None
    recovery = metadata.get("recovery")
    if (
        any(metadata.get(key) != original.get(key)
            for key in ("episode_index", "transitions", "task", "object_key", "reset"))
        or (decision is None and recovery is not None)
        or (decision is not None and (
            not isinstance(recovery, dict) or any(recovery.get(key) != value for key, value in decision.items())
        ))
    ):
        raise ValueError(f"{path} does not match the source episode or recovery decision")
    return metadata


def recover(args):
    run = args.run_dir.resolve()
    traces = (run / "traces").resolve()
    output = args.output_dir or run / "traces_recovery"
    output = output.parent.resolve() / output.name
    if output.is_relative_to(traces) or traces.is_relative_to(output):
        raise ValueError("--output-dir must not overlap the source traces directory")
    scores = (args.value_scores or (run / "gr00t_reward_value_scores")).resolve()
    rows = score_rows(scores, traces)
    paths = {path.parent.name: path for path in episodes(traces)}
    if args.episode:
        selected = {f"episode_{index:06d}" for index in args.episode}
        if missing := selected - paths.keys():
            raise ValueError(f"requested episodes not found in {traces}: {', '.join(sorted(missing))}")
        paths = {name: path for name, path in paths.items() if name in selected}
    if not paths:
        raise ValueError(f"no episodes in {traces}")
    if missing := paths.keys() - rows.keys():
        raise ValueError(f"missing recovery score for {', '.join(sorted(missing))} in {scores}")

    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.with_name(f".{output.name}.lock")
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another recovery is writing to {output}") from error
        workspace = recovery_workspace(args, output)
        recover_episodes(args, traces, output, workspace, paths, rows)


def recover_episodes(args, traces, output, workspace, paths, rows):
    def rebase(record, keys, directory):
        references = {}
        for key in keys:
            if key not in record:
                continue
            target = (traces / directory / record[key]).resolve()
            if target.is_relative_to(traces):
                relative = target.relative_to(traces)
                if (workspace / relative).exists():
                    target = output / relative
            references[key] = os.path.relpath(target, output / directory)
        return references

    jobs, copies, decoders = [], [], set()
    completed = {}
    originals = {}
    for name, path in paths.items():
        row = rows[name]
        decision = row["recovery_decision"]
        metadata = json.loads(path.read_text())
        if name != f"episode_{metadata['episode_index']:06d}":
            raise ValueError(f"episode index does not match directory: {path}")
        originals[name] = metadata
        if workspace.exists():
            previous = completed_episode(workspace / name / path.name, metadata, decision)
            if previous is not None:
                completed[name] = previous
                continue
        if decision is None:
            copies.append(path.parent)
            continue
        try:
            with np.load(path.with_name("trace.npz"), allow_pickle=False) as trace:
                step = takeover_step(metadata, trace, decision, row["local_score"])
                commands = [f"command_{key}" for key in COMMAND_KEYS if f"command_{key}" not in trace]
                if commands:
                    raise ValueError(f"missing recorded commands: {', '.join(commands)}")
        except (KeyError, OSError, ValueError) as error:
            raise ValueError(f"{name}: {error}") from error
        if args.decoder_model is None:
            decoders.add((path.parent / metadata["decoder"]).resolve())
        jobs.append(dict(path=str(path), step=step, decision=decision))
    if jobs:
        if args.decoder_model is None:
            if len(decoders) != 1:
                raise ValueError("episodes reference different decoder models; specify --decoder-model")
            args.decoder_model = decoders.pop()
        args.decoder_model = args.decoder_model.resolve()
        if not args.decoder_model.is_file():
            raise FileNotFoundError(f"decoder model not found: {args.decoder_model}; specify --decoder-model")
        args.checkpoint = resolve_checkpoint(args.checkpoint)
        for name, metadata in completed.items():
            recovery = metadata.get("recovery")
            if recovery is None:
                continue
            for key, expected in (("checkpoint", args.checkpoint), ("decoder", args.decoder_model)):
                if key in recovery and (workspace / name / recovery[key]).resolve() != expected:
                    raise ValueError(f"{name}: completed recovery uses a different {key}; "
                                     "resume with the original settings or choose a new --output-dir")

    print(f"[Recovery] {len(jobs)} to recover, {len(copies)} to copy, {len(completed)} complete", flush=True)
    workspace.mkdir(exist_ok=True)
    print(f"[Recovery] Working directory: {workspace}", flush=True)
    if workspace != output or jobs or copies:
        for source in traces.iterdir():
            if source.is_dir():
                if not source.name.startswith("episode_"):
                    shutil.copytree(source, workspace / source.name, dirs_exist_ok=True)
            else:
                shutil.copy2(source, workspace / source.name)
        if output.parent != traces.parent:
            for path in (traces / "objects").rglob("scene.json"):
                scene = json.loads(path.read_text())
                directory = path.parent.relative_to(traces)
                scene.update(rebase(scene, ("initial_pose", "checkpoint", "decoder", "scene_config"), directory))
                scene["manifest"].update(rebase(scene["manifest"], ("source",), directory))
                background = scene["config"].get("isaac_scene", {})
                background.update(rebase(background, ("asset_path",), directory))
                update_episode_metadata(workspace / path.relative_to(traces), scene)

    for name, metadata in originals.items():
        if output.parent != traces.parent:
            metadata.update(rebase(metadata, EPISODE_PATH_KEYS, name))
        if name in completed:
            previous = completed[name]
            updated = dict(previous, **{key: metadata[key] for key in EPISODE_PATH_KEYS if key in metadata})
            if updated != previous:
                update_episode_metadata(workspace / name / "episode.json", updated)

    for source in copies:
        destination = workspace / source.name
        shutil.copytree(source, destination, dirs_exist_ok=True, ignore=shutil.ignore_patterns("episode.json"))
        update_episode_metadata(destination / "episode.json", originals[source.name],
                                mode_from=source / "episode.json")

    if jobs:
        def assign(env_id, index):
            job = jobs[index]
            return dict(job, env_id=env_id, checkpoint=str(args.checkpoint), decoder=str(args.decoder_model),
                        output_metadata=originals[Path(job["path"]).parent.name])

        try:
            run_collection(args, workspace, len(jobs), assign)
        except RuntimeError as error:
            log = workspace / "decoder.log"
            if log.is_file():
                tail = "\n".join(log.read_text(errors="replace").splitlines()[-8:])
                if tail:
                    raise RuntimeError(f"{error}\nDecoder log:\n{tail}") from error
            raise
    if workspace != output:
        if os.path.lexists(output):
            raise FileExistsError(output)
        workspace.rename(output)
    print(f"[Recovery] {len(paths)} episodes -> {output}", flush=True)
