"""Batched GRAIL actor, a shared C++ decoder, and CPU simulation workers."""

import json
import multiprocessing as mp
import os
from pathlib import Path
import resource
import subprocess
import sys
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
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    return int(result.stdout.splitlines()[0]) if result.returncode == 0 else None


class Decoder:
    def __init__(self, executable, model, log):
        self.directory = tempfile.TemporaryDirectory(prefix="grail-offline-")
        self.endpoint = "ipc://" + str(Path(self.directory.name) / "decoder.sock")
        self.log = open(log, "w")
        self.process = subprocess.Popen([str(executable), str(model), self.endpoint], stdout=self.log,
                                        stderr=subprocess.STDOUT)
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(self.endpoint)

    def __call__(self, requests):
        self.socket.send(msgpack.packb(requests, use_bin_type=True))
        start = time.monotonic()
        while not self.socket.poll(1000):
            if self.process.poll() is not None:
                raise RuntimeError(f"decoder exited with {self.process.returncode}; see {self.log.name}")
            if time.monotonic() - start > 120:
                raise TimeoutError(f"decoder request timed out; see {self.log.name}")
        return msgpack.unpackb(self.socket.recv(), raw=False)

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.socket.close()
        self.context.term()
        self.log.close()
        self.directory.cleanup()


def prepare_scene(args):
    import mujoco
    from .scene import build_scene

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if list(output.glob("episode_*")):
        raise FileExistsError(f"collection directory already contains episodes: {output}")
    # gr00t-sim gets pxr from Kit, not usd-core. Prepare in a fresh Kit process,
    # keeping all USD/native renderer libraries out of the collection process.
    if not (args.cache_dir / args.object_key / "latest.json").is_file():
        from .offline_render import kit_environment
        asset_job = output / "prepare_assets.json"
        asset_job.write_text(json.dumps(dict(
            prepare_assets=True, dataset_dir=str(args.dataset_dir.resolve()),
            object_key=args.object_key, cache_dir=str(args.cache_dir.resolve()),
        )))
        with open(output / "prepare_assets.log", "w") as log:
            subprocess.run([sys.executable, "-m", "gear_sonic.utils.grail.offline_render", str(asset_job)],
                           stdout=log, stderr=subprocess.STDOUT, check=True, env=kit_environment())
    model, config, manifest = build_scene(
        args.robot_scene.resolve(), args.scene_config.resolve(),
        args.dataset_dir, args.object_key, args.cache_dir,
    )
    mujoco.mj_saveModel(model, str(output / "model.mjb"))
    scene = dict(
        config=config, manifest=manifest, fps=50, width=640, height=480,
        initial_pose=str(args.initial_pose.resolve()), checkpoint=str(args.checkpoint.resolve()),
        decoder=str(args.decoder_model.resolve()), task=args.task,
        scene_config=str(args.scene_config.resolve()), object_key=args.object_key,
        release_override=not args.no_release_override,
        mujoco_version=mujoco.__version__,
    )
    (output / "scene.json").write_text(json.dumps(scene, default=json_default, indent=2))
    return output


def receive(connection):
    result = connection.recv()
    if isinstance(result, dict) and "error" in result:
        raise RuntimeError(result["error"])
    return result


def collect(args):
    import torch
    from .policy import GrailPolicy, finger_targets, fsq
    from .checkpoints import resolve_checkpoint

    if args.num_envs < 1 or args.cpu_workers < 1 or args.num_episodes < args.num_envs:
        raise ValueError("require positive worker/environment counts and num_episodes >= num_envs")
    steps = round(args.duration * 50)
    if steps < 1 or abs(steps / 50 - args.duration) > 1e-9:
        raise ValueError("duration must be a positive multiple of 20 ms")
    args.checkpoint = resolve_checkpoint(args.checkpoint)
    print(f"[Collect] Checkpoint: {args.checkpoint}", flush=True)
    output = prepare_scene(args)
    torch.set_num_threads(args.threads)
    policy = GrailPolicy(args.checkpoint, args.device)
    policy(np.zeros((args.num_envs, 5, 161)), np.zeros((args.num_envs, 79)))
    ctx = mp.get_context("spawn")
    count = min(args.cpu_workers, args.num_envs)
    connections, processes, env_workers = [], [], {}
    decoder = None
    timings = dict(policy_s=0., decoder_roundtrip_s=0., decoder_inference_s=0., worker_wait_s=0.)
    cpu_start, start = time.process_time(), time.perf_counter()
    completed, next_episode, snapshots = 0, args.num_envs, []
    try:
        decoder = Decoder(args.decoder_executable.resolve(), args.decoder_model.resolve(), output / "decoder.log")
        for worker in range(count):
            parent, child = ctx.Pipe()
            ids = list(range(worker, args.num_envs, count))
            env_workers.update({env_id: worker for env_id in ids})
            process = ctx.Process(target=simulation_worker, args=(
                child, dict(output=str(output), steps=steps), ids,
            ))
            process.start()
            child.close()
            connections.append(parent)
            processes.append(process)
            parent.send(("reset", [(i, i, args.seed + i) for i in ids]))
        for connection in connections:
            snapshots.extend(receive(connection))
        last_report = time.perf_counter()
        peak_gpu = gpu_memory()
        while snapshots:
            tick = time.perf_counter()
            proprio = np.stack([s["proprio"] for s in snapshots])
            task = np.stack([s["task"] for s in snapshots])
            raw = policy(proprio, task)
            tokens = fsq(raw[:, :64]).cpu().numpy()
            experts = raw.cpu().numpy().copy()
            if not args.no_release_override:
                release = np.array([s["release_requested"] for s in snapshots], dtype=bool)
                experts[release, 64:] = -1
            actions, requests = [], []
            for sample, token, expert in zip(snapshots, tokens, experts):
                left, right = finger_targets(expert, policy.config)
                action = np.concatenate((token, left, right))
                actions.append(action)
                requests.append(dict(sample["request"], action=action.tolist()))
            timings["policy_s"] += time.perf_counter() - tick
            tick = time.perf_counter()
            replies = decoder(requests)
            timings["decoder_roundtrip_s"] += time.perf_counter() - tick
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
                        print(f"[Collect] {completed}/{args.num_episodes}: {result['episode']}", flush=True)
                        if next_episode < args.num_episodes:
                            replacements[worker].append((result["env_id"], next_episode, args.seed + next_episode))
                            next_episode += 1
                    else:
                        snapshots.append(result)
            for connection, replacement in zip(connections, replacements):
                if replacement:
                    connection.send(("reset", replacement))
            for connection, replacement in zip(connections, replacements):
                if replacement:
                    snapshots.extend(receive(connection))
            timings["worker_wait_s"] += time.perf_counter() - tick
            if time.perf_counter() - last_report > 10:
                memory = gpu_memory()
                if memory is not None:
                    peak_gpu = max(peak_gpu or 0, memory)
                print(f"[Collect] active={len(snapshots)} completed={completed}; GPU={memory} MiB", flush=True)
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
        n = completed * steps
        stats = dict(
            num_envs=args.num_envs, cpu_workers=count, episodes=completed, transitions=n,
            wall_s=elapsed, physics_steps_per_second=n * 4 / elapsed,
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
        (output / "collection_metrics.json").write_text(json.dumps(stats, indent=2))
        print(json.dumps(stats, indent=2), flush=True)
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
