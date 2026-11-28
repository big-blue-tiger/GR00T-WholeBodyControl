"""Batched GRAIL actor, a shared C++ decoder, and CPU simulation workers."""

from contextlib import ExitStack
import fcntl
import json
import multiprocessing as mp
import os
from pathlib import Path
import resource
import subprocess
import tempfile
import time

import msgpack
import numpy as np
import zmq

from .offline_sim import simulation_worker


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def gpu_memory():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        )
    except FileNotFoundError:
        return None
    return int(result.stdout.splitlines()[0]) if result.returncode == 0 and result.stdout.strip() else None


class Decoder:
    def __init__(self, executable, model, log):
        with ExitStack() as startup:
            self.directory = tempfile.TemporaryDirectory(prefix="grail-offline-")
            startup.callback(self.directory.cleanup)
            socket_path = Path(self.directory.name) / "decoder.sock"
            self.endpoint = f"ipc://{socket_path}"
            self.log = startup.enter_context(open(log, "w"))
            self.context = zmq.Context()
            startup.callback(self.context.term)
            self.socket = self.context.socket(zmq.REQ)
            startup.callback(self.socket.close, linger=0)
            self.socket.setsockopt(zmq.LINGER, 0)
            self.socket.connect(self.endpoint)

            # The decoder may rebuild a shared TensorRT cache before binding its socket.
            with open(model, "rb") as model_file:
                fcntl.flock(model_file, fcntl.LOCK_EX)
                self.process = subprocess.Popen(
                    [str(executable), str(model), self.endpoint], stdout=self.log, stderr=subprocess.STDOUT,
                )
                try:
                    self.wait_until_ready(socket_path)
                except BaseException:
                    self.stop_process()
                    raise
            startup.pop_all()

    def wait_until_ready(self, socket_path):
        deadline = time.monotonic() + 120
        while not socket_path.is_socket():
            status = self.process.poll()
            if status is not None:
                raise RuntimeError(f"decoder exited with {status} during startup; see {self.log.name}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"decoder startup timed out; see {self.log.name}")
            time.sleep(0.05)

    def __call__(self, requests):
        self.socket.send(msgpack.packb(requests, use_bin_type=True))
        start = time.monotonic()
        while not self.socket.poll(1000):
            if self.process.poll() is not None:
                raise RuntimeError(f"decoder exited with {self.process.returncode}; see {self.log.name}")
            if time.monotonic() - start > 120:
                raise TimeoutError(f"decoder request timed out; see {self.log.name}")
        return msgpack.unpackb(self.socket.recv(), raw=False)

    def stop_process(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()

    def close(self):
        self.stop_process()
        self.socket.close()
        self.context.term()
        self.log.close()
        self.directory.cleanup()


def build_offline_scene(args, object_key):
    from .observations import rotation
    from .scene import build_scene

    model, config, manifest = build_scene(
        args.robot_scene.resolve(), args.scene_config.resolve(),
        args.dataset_dir, object_key, args.cache_dir,
    )
    background = config.get("isaac_scene", {})
    release_position = config.get("offline_release_position_scene")
    if release_position is not None:
        release_position = np.asarray(release_position, dtype=float)
        if release_position.shape != (3,) or not np.isfinite(release_position).all():
            raise ValueError("offline_release_position_scene requires three finite metre values")
        release_position = (
            rotation(background["quaternion_wxyz"]) @ release_position + background["translation"]
        ).tolist()
    return model, config, manifest, release_position


def prepare_scene(args, output, object_key):
    import mujoco

    model, config, manifest, release_position = build_offline_scene(args, object_key)
    background = config.get("isaac_scene", {})
    if "asset_path" in background:
        background["asset_path"] = os.path.relpath(background["asset_path"], output)
    manifest["source"] = os.path.relpath(manifest["source"], output)
    # Dependency names remain relative to the dataset's object_usd/ parent.
    manifest["dependencies"] = {
        os.path.relpath(args.dataset_dir / name, args.dataset_dir): digest
        for name, digest in manifest["dependencies"].items()
    }
    scene = dict(
        config=config, manifest=manifest, fps=50, width=640, height=480,
        initial_pose=os.path.relpath(args.initial_pose, output),
        checkpoint=os.path.relpath(args.checkpoint, output),
        decoder=os.path.relpath(args.decoder_model, output), task=args.task,
        scene_config=os.path.relpath(args.scene_config, output), object_key=object_key,
        release_position=release_position, release_override=not args.no_release_override,
        mujoco_version=mujoco.__version__,
    )
    output.mkdir(parents=True, exist_ok=True)
    mujoco.mj_saveModel(model, str(output / "model.mjb"))
    (output / "scene.json").write_text(json.dumps(scene, default=json_default, indent=2))


def collection_objects(dataset_dir, requested, cache_dir):
    from .assets import prepared_directory, prepared_object_keys

    keys = list(dict.fromkeys(requested or sorted(
        path.stem for path in (dataset_dir / "object_usd").glob("*.usd") if path.is_file()
    )))
    if not keys:
        raise ValueError(f"no USD objects in {dataset_dir / 'object_usd'}")
    missing = set(keys).difference(prepared_object_keys(dataset_dir, cache_dir))
    if missing:
        cache = prepared_directory(dataset_dir, cache_dir)
        raise FileNotFoundError(f"Missing USD, BPS or prepared assets in {cache}: {', '.join(sorted(missing))}")
    return keys


def receive(connection):
    result = connection.recv()
    if isinstance(result, dict) and "error" in result:
        raise RuntimeError(result["error"])
    return result


def collect(args):
    from .checkpoints import resolve_checkpoint

    keys = collection_objects(args.dataset_dir, args.object_key, args.cache_dir)
    episodes_per_object = args.episodes_per_object
    num_episodes = len(keys) * episodes_per_object
    if min(args.num_envs, args.cpu_workers, episodes_per_object) < 1 or num_episodes < args.num_envs:
        raise ValueError("require positive worker/environment counts and total episodes >= num_envs")
    steps = round(args.duration * 50)
    if steps < 1 or abs(steps / 50 - args.duration) > 1e-9:
        raise ValueError("duration must be a positive multiple of 20 ms")
    args.checkpoint = resolve_checkpoint(args.checkpoint)
    output = args.output.resolve()
    if any(output.glob("episode_*")):
        raise FileExistsError(f"collection directory already contains episodes: {output}")
    print(f"[Collect] Checkpoint: {args.checkpoint}", flush=True)
    print(f"[Collect] {len(keys)} objects × {episodes_per_object} episodes = {num_episodes}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    scene_dirs = {key: output / "objects" / key if len(keys) > 1 else output for key in keys}
    if len(keys) > 1:
        index = dict(task=args.task, episodes_per_object=episodes_per_object, objects={
            key: str((directory / "scene.json").relative_to(output)) for key, directory in scene_dirs.items()
        })
        (output / "scene.json").write_text(json.dumps(index, indent=2))
    prepared = set()

    def reset_assignment(env_id, episode):
        key = keys[episode // episodes_per_object]
        directory = scene_dirs[key]
        if key not in prepared:
            prepare_scene(args, directory, key)
            prepared.add(key)
        return env_id, episode, args.seed + episode, str(directory)

    stats = run_collection(args, output, num_episodes, reset_assignment, steps=steps)
    stats.update(num_objects=len(keys), episodes_per_object=episodes_per_object)
    (output / "collection_metrics.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2), flush=True)
    return stats


def run_collection(args, output, num_episodes, reset_assignment, *, steps=None):
    import torch

    from .policy import GrailActor, GrailPolicy, finger_targets

    num_envs = min(args.num_envs, num_episodes)
    count = min(args.cpu_workers, num_envs)
    torch.set_num_threads(args.threads)
    policy = GrailPolicy(args.checkpoint, args.device)
    policy(np.zeros((num_envs, 5, 161)), np.zeros((num_envs, 79)))
    actor = GrailActor(policy, release_override=not args.no_release_override)
    worker_options = dict(output=str(output), steps=steps)
    operation, label = "reset", "Collect"
    if steps is None:
        operation, label = "recover", "Recovery"
        worker_options["hand_primitives"] = np.asarray([
            finger_targets(np.full(66, sign), policy.config) for sign in (-1., 1.)
        ])
    ctx = mp.get_context("spawn")
    connections, processes, env_workers = [], [], {}
    decoder = None
    timings = dict(policy_s=0., decoder_roundtrip_s=0., decoder_inference_s=0., worker_wait_s=0.)
    cpu_start, start = time.process_time(), time.perf_counter()
    completed, transitions, next_episode, snapshots = 0, 0, num_envs, []
    try:
        decoder = Decoder(args.decoder_executable.resolve(), args.decoder_model.resolve(), output / "decoder.log")
        for worker in range(count):
            parent, child = ctx.Pipe()
            ids = list(range(worker, num_envs, count))
            env_workers.update({env_id: worker for env_id in ids})
            process = ctx.Process(target=simulation_worker, args=(child, worker_options))
            process.start()
            child.close()
            connections.append(parent)
            processes.append(process)
            parent.send((operation, [reset_assignment(i, i) for i in ids]))
        for connection in connections:
            snapshots.extend(receive(connection))
        last_report = time.perf_counter()
        peak_gpu = gpu_memory()
        while snapshots:
            tick = time.perf_counter()
            experts = []
            actions, requests = [], []
            for sample, (expert, token) in zip(snapshots, actor.act_batch(snapshots)):
                left, right = finger_targets(expert, policy.config)
                action = np.concatenate((token, left, right))
                experts.append(expert)
                actions.append(action)
                requests.append(dict(sample["request"], action=action.tolist()))
            timings["policy_s"] += time.perf_counter() - tick
            tick = time.perf_counter()
            replies = decoder(requests)
            timings["decoder_roundtrip_s"] += time.perf_counter() - tick
            if len(replies) != len(requests):
                raise RuntimeError("decoder reply count does not match the request batch")
            timings["decoder_inference_s"] += sum(r["inference_s"] for r in replies)
            batches = [[] for _ in connections]
            for sample, reply, expert, action in zip(snapshots, replies, experts, actions):
                batches[env_workers[sample["env_id"]]].append(
                    dict(env_id=sample["env_id"], reply=reply, expert=expert, action=action))
            tick = time.perf_counter()
            for connection, batch in zip(connections, batches):
                if batch:
                    connection.send(("step", batch))
            snapshots, replacements = [], [[] for _ in connections]
            for worker, (connection, batch) in enumerate(zip(connections, batches)):
                if not batch:
                    continue
                for result in receive(connection):
                    if result.get("done"):
                        completed += 1
                        transitions += result["transitions"]
                        print(f"[{label}] {completed}/{num_episodes}: {result['episode']}", flush=True)
                        if next_episode < num_episodes:
                            replacements[worker].append(reset_assignment(result["env_id"], next_episode))
                            next_episode += 1
                    else:
                        snapshots.append(result)
            for connection, replacement in zip(connections, replacements):
                if replacement:
                    connection.send((operation, replacement))
            for connection, replacement in zip(connections, replacements):
                if replacement:
                    snapshots.extend(receive(connection))
            timings["worker_wait_s"] += time.perf_counter() - tick
            if time.perf_counter() - last_report > 10:
                memory = gpu_memory()
                if memory is not None:
                    peak_gpu = max(peak_gpu or 0, memory)
                print(f"[{label}] active={len(snapshots)} completed={completed}; GPU={memory} MiB", flush=True)
                last_report = time.perf_counter()
        # Read while the server is still alive and waiting for its shutdown request.
        decoder_stat = Path(f"/proc/{decoder.process.pid}/stat").read_text().rpartition(")")[2].split()
        decoder_cpu_s = (float(decoder_stat[11]) + float(decoder_stat[12])) / os.sysconf("SC_CLK_TCK")
        decoder([])
        memory = gpu_memory()
        if memory is not None:
            peak_gpu = max(peak_gpu or 0, memory)
        for connection in connections:
            connection.send(("close", None))
        workers = [receive(connection) for connection in connections]
        for process in processes:
            process.join()
        elapsed = time.perf_counter() - start
        stats = dict(
            num_envs=num_envs, cpu_workers=count, episodes=completed, transitions=transitions,
            wall_s=elapsed, physics_steps_per_second=transitions * 4 / elapsed,
            episodes_per_hour=completed * 3600 / elapsed, timing=timings, workers=workers,
            parent_cpu_s=time.process_time() - cpu_start,
            parent_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            sampled_peak_gpu_mib=peak_gpu,
        )
        stats["worker_timing"] = {k: sum(w["timing"][k] for w in workers) for k in workers[0]["timing"]}
        stats["decoder_cpu_s"] = decoder_cpu_s
        stats["measured_cpu_core_equivalents"] = (
            stats["parent_cpu_s"] + decoder_cpu_s + sum(w["cpu_s"] for w in workers)
        ) / elapsed
        stats["cpu_affinity_count"] = len(os.sched_getaffinity(0))
        stats["measured_cpu_utilization_percent"] = (
            100 * stats["measured_cpu_core_equivalents"] / stats["cpu_affinity_count"]
        )
        return stats
    finally:
        for connection in connections:
            connection.close()
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join()
        if decoder is not None:
            decoder.close()
