"""Replay GRAIL and GR00T traces to score the executed 50 Hz transitions."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import closing
from dataclasses import dataclass
import fcntl
from functools import lru_cache
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
from zipfile import BadZipFile

import mujoco
import numpy as np

from gear_sonic.utils.grail.carry_reward_value import SCORE_FIELDS, CarryRewardValueScorer, CarryRewardValueStream
from gear_sonic.utils.grail.checkpoints import DEFAULT_CHECKPOINT, resolve_checkpoint
from gear_sonic.utils.grail.gr00t_recovery import RecoveryConfig, RecoveryGate
from gear_sonic.utils.grail.offline_data import episodes, load_episode
from gear_sonic.utils.grail.offline_sim import COMMAND_KEYS, COMMAND_NAMES, OfflineEnvironment, load_boundary
from gear_sonic.utils.grail.task import CONTACT_STABLE_STEPS

RELEASE_ADVANTAGE_DISTANCE_M = 0.3
RELEASE_ADVANTAGE_LEAD_FRAMES = 3
RELEASE_ADVANTAGE_RULE = 2


def final_release_start(carrying, target_distance_m):
    """Find the start of the score tail to zero for a near-target release."""
    if len(carrying) < CONTACT_STABLE_STEPS:
        return None
    contacts = np.lib.stride_tricks.sliding_window_view(carrying, CONTACT_STABLE_STEPS).sum(axis=-1)
    held = np.flatnonzero(contacts == CONTACT_STABLE_STEPS)
    if not held.size:
        return None
    after_hold = held[-1] + CONTACT_STABLE_STEPS
    released = np.flatnonzero(contacts[after_hold:] == 0)
    if not released.size:
        return None
    first_clear = after_hold + released[0]
    if target_distance_m[first_clear - 1] <= RELEASE_ADVANTAGE_DISTANCE_M:
        confirmed = first_clear + CONTACT_STABLE_STEPS - 1
        return max(0, confirmed - RELEASE_ADVANTAGE_LEAD_FRAMES)
    return None


def replay(env, metadata, trace, scorer):
    if list(metadata["command_joint_names"]) != COMMAND_NAMES:
        raise ValueError("Unsupported command joint order")
    commands = {key: trace[f"command_{key}"] for key in COMMAND_KEYS}
    for key, values in commands.items():
        if values.shape != (env.steps, len(COMMAND_NAMES)):
            raise ValueError(f"Invalid command_{key} shape: {values.shape}")
    env.start_replay(metadata, trace)
    recovery = metadata.get("recovery", {})
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
        if recovery.get("regrasp") and recovery["step"] == env.step_index:
            env.restart_pickup()
    start = final_release_start(result["carrying"], result["target_distance_m"])
    if start is not None:
        result["td_advantage"][start:] = 0.0
    return result, decision


@dataclass(frozen=True)
class ScoreJob:
    path: Path
    target: Path
    scene: Path
    model: Path
    episode_index: int
    transitions: int
    inputs: str

    @property
    def root(self):
        return self.path.parent.parent


def file_signature(path):
    stat = path.stat()
    return [stat.st_size, stat.st_mtime_ns]


def score_inputs(metadata, *paths):
    replay_metadata = {key: value for key, value in metadata.items()
                       if key not in ("label", "min_target_distance_m")}
    inputs = (replay_metadata, [file_signature(path) for path in paths])
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()


@lru_cache(maxsize=1)
def load_scene(scene_path, model_path):
    return json.loads(scene_path.read_text()), mujoco.MjModel.from_binary_path(str(model_path))


@lru_cache(maxsize=1)
def load_scorer(checkpoint):
    return CarryRewardValueScorer(checkpoint)


def score_episode(job, checkpoint):
    try:
        scorer = load_scorer(checkpoint)
        metadata, trace = load_episode(job.path)
        scene, model = load_scene(job.scene, job.model)
        env = OfflineEnvironment(model, scene, None, 0, job.root, metadata["transitions"], job.scene.parent)
        result, decision = replay(env, metadata, trace, scorer)
    except Exception as error:
        raise RuntimeError(f"scoring failed for {job.path}: {error}") from error
    return result, decision, scorer.gamma


def scored_jobs(jobs, checkpoint, workers):
    if workers == 1:
        for job in jobs:
            yield job, score_episode(job, checkpoint)
        return

    pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"))
    try:
        futures = {pool.submit(score_episode, job, checkpoint): job for job in jobs}
        for future in as_completed(futures):
            yield futures.pop(future), future.result()
    finally:
        pool.shutdown(cancel_futures=True)


def complete_score(job, row, checkpoint_files):
    if row is None or "recovery_decision" not in row:
        return False
    trace_path = job.path.with_name("trace.npz")
    try:
        if "inputs" in row:
            if row["inputs"] != job.inputs:
                return False
        else:
            # Old manifests have no input signature. Check file times and recorded boundaries.
            inputs = (trace_path, job.scene, job.model, *checkpoint_files)
            if any(path.stat().st_mtime_ns > job.target.stat().st_mtime_ns for path in inputs):
                return False
        with np.load(job.target, allow_pickle=False) as result, np.load(trace_path, allow_pickle=False) as trace:
            return (
                all(result[key].shape == (job.transitions,) for key in SCORE_FIELDS)
                and np.array_equal(result["time_s"], trace["sim_time"][1:])
                and np.array_equal(result["pelvis_pos"], trace["qpos"][1:, :3])
            )
    except (OSError, EOFError, ValueError, KeyError, TypeError, BadZipFile):
        return False


def write_score(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as file:
        np.savez_compressed(file, **result)
    temporary.replace(path)


def upgrade_legacy_score(job, row):
    # Previous scores already include the two diagnostics needed by this rule.
    try:
        with np.load(job.target, allow_pickle=False) as cached:
            carrying, distance = cached["carrying"], cached["target_distance_m"]
            if carrying.shape != (job.transitions,) or distance.shape != (job.transitions,):
                return False
            start = final_release_start(carrying, distance)
            updated = dict(cached) if start is not None and np.any(cached["td_advantage"][start:] != 0) else None
    except (OSError, EOFError, ValueError, KeyError, TypeError, BadZipFile):
        return False
    if updated is not None:
        updated["td_advantage"][start:] = 0.0
        write_score(job.target, updated)
    row["release_advantage_rule"] = RELEASE_ADVANTAGE_RULE
    return True


def write_manifest(path, rows, source_order):
    def order(row):
        trace = Path(row["trace"])
        return source_order[trace.parent.parent], row["episode_index"], trace

    temporary = path.with_suffix(".jsonl.tmp")
    with temporary.open("w") as file:
        for row in sorted(rows.values(), key=order):
            file.write(json.dumps(row) + "\n")
    temporary.replace(path)


def run_scoring(args, jobs, roots):
    checkpoint = resolve_checkpoint(args.checkpoint)
    checkpoint_files = (checkpoint, checkpoint.with_name("config.yaml"))
    checkpoint_signature = [file_signature(path) for path in checkpoint_files]
    manifest = args.output / "episodes.jsonl"
    rows = {}
    targets = {}
    source_order = {root: index for index, root in enumerate(dict.fromkeys(roots))}
    if manifest.exists():
        if not args.resume:
            raise FileExistsError(f"{manifest} already exists; use --resume or a new --output")
        lines = manifest.read_text().splitlines(keepends=True)
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if index == len(lines) - 1 and not line.endswith("\n"):
                    print(f"[Score] Ignoring incomplete final row in {manifest}", flush=True)
                    break
                raise
            path, target = Path(row["trace"]), Path(row["score"])
            if path in targets or target in rows:
                raise ValueError(f"duplicate episode in {manifest}: {path}")
            if (row.get("checkpoint") != str(checkpoint)
                    or row.get("checkpoint_signature", checkpoint_signature) != checkpoint_signature):
                raise ValueError(f"checkpoint has changed since {manifest} was written; use a new --output")
            if "episode_index" not in row:
                row["episode_index"] = (json.loads(path.read_text())["episode_index"] if path.is_file()
                                        else int(path.parent.name.removeprefix("episode_")))
            rows[target] = row
            targets[path] = target
            source_order.setdefault(path.parent.parent, len(source_order))

    pending = []
    upgraded = 0
    for job in jobs:
        row = rows.get(job.target)
        if targets.get(job.path, job.target) != job.target or (row and row["trace"] != str(job.path)):
            raise ValueError(f"output layout or trace does not match {manifest}: {job.target}")
        if args.resume and complete_score(job, row, checkpoint_files):
            if row.get("release_advantage_rule") == RELEASE_ADVANTAGE_RULE:
                continue
            if row.get("release_advantage_rule") in (None, 1) and upgrade_legacy_score(job, row):
                upgraded += 1
                continue
        if not args.resume and job.target.exists():
            raise FileExistsError(f"{job.target} already exists; use --resume or a new --output")
        rows.pop(job.target, None)
        pending.append(job)

    workers = min(args.workers, len(pending))
    skipped = len(jobs) - len(pending)
    print(f"[Score] {checkpoint} | {len(pending)} pending, {skipped} skipped, {workers} workers", flush=True)
    if upgraded:
        print(f"[Score] Applied the release Advantage rule to {upgraded} existing scores", flush=True)
    write_manifest(manifest, rows, source_order)
    if not pending:
        return

    # Only the parent publishes files. Each atomic index update follows its score file.
    with closing(scored_jobs(pending, checkpoint, workers)) as results:
        for completed, (job, (result, decision, gamma)) in enumerate(results, start=1):
            write_score(job.target, result)
            rows[job.target] = dict(
                trace=str(job.path), score=str(job.target), episode_index=job.episode_index,
                checkpoint=str(checkpoint), gamma=gamma, recovery_decision=decision,
                checkpoint_signature=checkpoint_signature, inputs=job.inputs,
                release_advantage_rule=RELEASE_ADVANTAGE_RULE,
            )
            write_manifest(manifest, rows, source_order)
            print(f"[Score] {skipped + completed}/{len(jobs)} {job.target.relative_to(args.output)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--traces", type=Path, nargs="+", required=True, help="One or more directories of recorded episodes"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--batch", action="store_true", help="Save scores under a source-named subdirectory even for one trace directory"
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--episode", type=int, action="append", help="Score only these source episode indices")
    parser.add_argument("--workers", type=int, default=1, help="CPU processes scoring independent episodes (default: 1)")
    parser.add_argument("--resume", action="store_true", help="Keep complete scores and finish the remaining episodes")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be a positive integer")
    args.output = args.output.resolve()
    roots = [root.resolve() for root in args.traces]
    jobs = defaultdict(list)
    targets = set()
    for root in roots:
        name = root.parent.name if root.name == "traces" else root.name
        directory = args.output / name if args.batch or len(args.traces) > 1 else args.output
        for path in episodes(root):
            metadata = json.loads(path.read_text())
            episode_index = metadata["episode_index"]
            if args.episode and episode_index not in args.episode:
                continue
            trace_path = path.with_name("trace.npz")
            with np.load(trace_path, allow_pickle=False) as archive:
                missing = [f"command_{key}" for key in COMMAND_KEYS if f"command_{key}" not in archive.files]
            if missing:
                parser.error(f"{trace_path} is missing recorded decoder commands: {', '.join(missing)}")
            scene = (path.parent / metadata["scene"]).resolve()
            model = (path.parent / metadata["model"]).resolve()
            target = directory / f"{path.parent.name}.npz"
            if target in targets:
                parser.error(f"multiple episodes would write to {target}")
            targets.add(target)
            inputs = score_inputs(metadata, trace_path, scene, model)
            jobs[(root, scene, model)].append(
                ScoreJob(path, target, scene, model, episode_index, metadata["transitions"], inputs)
            )
    if not jobs:
        parser.error("no matching episodes")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".score.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another scorer is writing to {args.output}") from error
        run_scoring(args, [job for group in jobs.values() for job in group], roots)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[Score] Interrupted. Rerun with --resume to continue.", flush=True)
        raise SystemExit(130) from None
