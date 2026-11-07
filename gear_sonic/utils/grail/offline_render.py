"""Replay saved trajectories in independent Isaac processes and encode aligned videos."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

VISUAL_CONFIG = dict(
    width=640,
    height=480,
    fps=50,
    renderer="RaytracedLighting",
    rt_subframes=1,
    anti_aliasing=0,
    codec="libx264",
    crf=18,
    preset="fast",
    pix_fmt="yuv420p",
)


def kit_environment():
    environment = os.environ.copy()
    for key, value in (
        ("__NV_PRIME_RENDER_OFFLOAD", "1"),
        ("__GLX_VENDOR_LIBRARY_NAME", "nvidia"),
        ("__VK_LAYER_NV_optimus", "NVIDIA_only"),
        ("OMP_NUM_THREADS", "1"),
        ("OPENBLAS_NUM_THREADS", "1"),
    ):
        environment.setdefault(key, value)
    return environment


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_fingerprint(path):
    path = Path(path)
    meta = json.loads(path.read_text())
    # Labels and task text do not change pixels; dynamics and visual inputs do.
    inputs = {"episode_index": meta["episode_index"], "transitions": meta["transitions"]}
    for key, filename in [
        ("trace", "trace.npz"),
        ("scene", meta["scene"]),
        ("model", meta["model"]),
    ]:
        inputs[key] = digest_file(path.parent / filename)
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()


def config_fingerprint(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def pending_episodes(paths, output, fingerprints, render_fingerprint, *, overwrite=False, resume=False):
    pending = []
    for path in paths:
        destination = output / path.parent.name
        video = destination / "ego_view.mp4"
        marker = destination / "render.json"
        if resume and video.is_file() and (destination / "frames.npz").is_file() and marker.is_file():
            try:
                result = json.loads(marker.read_text())
            except json.JSONDecodeError:
                result = {}
            if (result.get("source_fingerprint") == fingerprints[str(path)]
                    and result.get("render_fingerprint") == render_fingerprint):
                continue
        if video.exists() and not (overwrite or resume):
            raise FileExistsError(f"Video already exists: {video}; use --resume, --overwrite, or a new --output")
        pending.append(path)
    return pending


def gpu_info():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,driver_version",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(
            "GPU unavailable; run in the GPU-enabled gr00t-sim environment:\n" + result.stderr
        )
    name, total, used, driver = result.stdout.splitlines()[0].split(", ")
    return dict(name=name, total_mib=int(total), used_mib=int(used), driver=driver)


def run_children(jobs, directory, name):
    """Monitor every worker, and terminate the entire group on any failure."""
    import signal

    processes = []
    failed = True
    directory.mkdir(parents=True, exist_ok=True)
    try:
        for index, job in enumerate(jobs):
            config = directory / f"{name}_{index}.json"
            config.write_text(json.dumps(job))
            with (directory / f"{name}_{index}.log").open("w") as log:
                processes.append(
                    subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "gear_sonic.utils.grail.offline_render",
                            str(config),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=kit_environment(),
                        start_new_session=True,
                    )
                )
        while any(process.poll() is None for process in processes):
            for index, process in enumerate(processes):
                if process.poll() not in (None, 0):
                    raise RuntimeError(
                        f'Isaac worker failed; see {directory / f"{name}_{index}.log"}'
                    )
            time.sleep(0.2)
        if any(process.returncode for process in processes):
            raise RuntimeError(f"Isaac workers failed; see {directory / name}")
        failed = False
    finally:
        for process in processes:
            if failed or process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def render_identity(paths, requested, encode_workers, gpu):
    """Fingerprint assets and implementation without launching a renderer."""
    from importlib import metadata

    scene_path = paths[0].parent / json.loads(paths[0].read_text())["scene"]
    scene = json.loads(scene_path.read_text())
    assets = [
        scene["manifest"]["source"],
        Path(__file__).parents[2] / "data/robots/g1/g1_43dof_s3.usda",
        *scene["manifest"]["dependencies"],
    ]
    background = scene["config"].get("isaac_scene", {})
    if background.get("enabled"):
        assets.append(background["asset_path"])
    visual_dir = Path(__file__).parents[1] / "mujoco_sim"
    implementation = [
        Path(__file__), Path(__file__).with_name("render_pipeline.py"),
        *(visual_dir / name for name in ("isaac_visual.py", "scene_visual.py", "isaac_renderer.kit")),
    ]
    return dict(
        gpu=gpu["name"],
        memory=gpu["total_mib"],
        driver=gpu["driver"],
        isaac_versions={name: metadata.version(name)
                        for name in ("isaacsim", "isaacsim-kernel", "isaacsim-replicator")},
        python=sys.prefix,
        scene=digest_file(scene_path),
        config=VISUAL_CONFIG,
        requested=requested,
        encode_workers=encode_workers,
        assets={str(p): digest_file(p) for p in assets},
        implementation={str(p): digest_file(p) for p in implementation},
    )


def render(args):
    from .offline_data import episodes
    from .render_pipeline import atomic_json

    started = time.perf_counter()
    paths = [p.resolve() for p in episodes(args.traces)]
    if not paths:
        raise ValueError("no complete episodes to render")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not (args.overwrite or args.resume):
        for path in paths:
            video = output / path.parent.name / "ego_view.mp4"
            if video.exists():
                raise FileExistsError(f"Video already exists: {video}; use --resume, --overwrite, or a new --output")
    gpu = gpu_info()
    selected = dict(mode="isolated", envs=1, processes=min(args.render_envs, len(paths)))
    identity = render_identity(paths, args.render_envs, args.encode_workers, gpu)
    config = dict(VISUAL_CONFIG, **selected, selection_key=config_fingerprint(identity))
    fingerprint = config_fingerprint(config)
    atomic_json(output / "render_selection.json",
                dict(key=config["selection_key"], identity=identity, selected=selected))
    fingerprints = {str(p): source_fingerprint(p) for p in paths}
    pending = pending_episodes(
        paths, output, fingerprints, fingerprint, overwrite=args.overwrite, resume=args.resume
    )
    print(f"[OfflineRender] {len(pending)} pending, {len(paths)-len(pending)} skipped", flush=True)
    rendering_started = time.perf_counter()
    active_processes = min(args.render_envs, len(pending))
    if pending:
        jobs = [dict(
            episodes=[str(p) for p in pending[i::active_processes]],
            output=str(output),
            encode_workers=max(1, args.encode_workers // active_processes),
            fingerprints=fingerprints,
            render_fingerprint=fingerprint,
            render_config=config,
            worker_index=i,
        ) for i in range(active_processes)]
        run_children(jobs, output, "render_job")
    results = [json.loads((output / p.parent.name / "render.json").read_text()) for p in pending]
    elapsed = time.perf_counter() - started
    rendering_s = time.perf_counter() - rendering_started
    count = sum(r["frames"] for r in results)
    stages = [json.loads((output / f"worker_metrics_{i}.json").read_text())
              for i in range(active_processes)]
    pipeline_wall = (
        max(s["pipeline_finished_monotonic"] for s in stages)
        - min(s["pipeline_started_monotonic"] for s in stages)
    ) if stages else 0.0
    atomic_json(output / "batch_metrics.json", dict(
        wall_s=elapsed, rendering_wall_s=rendering_s, frames=count,
        frames_per_second=count / elapsed, rendering_frames_per_second=count / rendering_s,
        pipeline_wall_s=pipeline_wall, pipeline_frames_per_second=count / pipeline_wall if pipeline_wall else 0.0,
        completed=len(results), skipped=len(paths)-len(pending), gpu=gpu,
        selected=selected, active_processes=active_processes, stages=stages,
    ))


def render_episodes(visual, job, pool):
    from collections import deque
    from itertools import islice
    from types import SimpleNamespace

    import omni.timeline

    from .render_pipeline import EncoderPool, atomic_json, prepare_episode

    paths = iter(job["episodes"])
    # Keep the same three-episode lookahead while rendering one camera at a time.
    futures = deque((path, pool.submit(prepare_episode, path)) for path in islice(paths, 3))
    history = []
    encoders = EncoderPool(job["encode_workers"])
    prepare_s = pose_wait_s = frames = 0
    started = time.perf_counter()
    peak = gpu_info()["used_mib"]
    try:
        while futures:
            path, future = futures.popleft()
            begin = time.perf_counter()
            prepared = future.result()
            pose_wait_s += time.perf_counter() - begin
            prepare_s += prepared["prepare_s"]
            following = next(paths, None)
            if following is not None:
                futures.append((following, pool.submit(prepare_episode, following)))
            meta = prepared["meta"]
            identifier = meta["episode_index"]
            encoders.send("start", identifier, dict(
                destination=str(Path(job["output"]) / Path(path).parent.name),
                started=time.perf_counter(), expected=meta["transitions"], timestamps=prepared["timestamps"],
                result=dict(episode_index=identifier, capture_history=list(history),
                            source_env_id=meta["env_id"], source_fingerprint=job["fingerprints"][path],
                            render_fingerprint=job["render_fingerprint"], render_config=job["render_config"],
                            prepare_s=prepared["prepare_s"]),
            ))
            for index in range(meta["transitions"]):
                pose = SimpleNamespace(**{key: value[index] for key, value in prepared["poses"].items()})
                # Exactly one synchronous capture per saved state; no episode warm-up frames.
                encoders.frame(identifier, index, visual.render(pose))
                frames += 1
                if frames % 250 == 0:
                    peak = max(peak, gpu_info()["used_mib"])
                    encoders.poll()
                    print(f"[OfflineRender] {len(encoders.finished)} complete; {frames} frames; GPU={peak} MiB", flush=True)
            encoders.send("finish", identifier, dict(
                timeline_time=omni.timeline.get_timeline_interface().get_current_time()
            ))
            history.append(identifier)
        encoders.wait(history)
        finished = time.perf_counter()
        atomic_json(Path(job["output"]) / f'worker_metrics_{job["worker_index"]}.json', dict(
            wall_s=finished-started, pipeline_started_monotonic=started,
            pipeline_finished_monotonic=finished, frames=frames,
            prepare_s=prepare_s, pose_wait_s=pose_wait_s,
            startup_s=job["startup_s"], scene_load_s=job["scene_load_s"],
            submit_s=visual.submit_s, capture_s=visual.capture_s, readback_s=visual.readback_s,
            encoder_queue_wait_s=encoders.queue_wait_s, peak_gpu_mib=peak,
            encode_s=sum(r["encode_s"] for r in encoders.finished.values()),
            stats_s=sum(r["stats_s"] for r in encoders.finished.values()),
        ))
    finally:
        encoders.close(abort=sys.exc_info()[0] is not None)


def run_jobs(job):
    sys.argv[:] = sys.argv[:1]
    os.environ.setdefault("MUJOCO_GL", "egl")
    from gear_sonic.utils.mujoco_sim.render_worker import create_camera_app

    boot_started = time.perf_counter()
    app = create_camera_app()
    visual = None
    try:
        if job.get("prepare_assets"):
            from .assets import prepare_assets

            prepare_assets(job["dataset_dir"], job["object_key"], job["cache_dir"])
            return
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing as mp
        import mujoco

        from gear_sonic.utils.mujoco_sim.isaac_visual import IsaacVisual

        booted = time.perf_counter()
        first = Path(job["episodes"][0])
        meta = json.loads(first.read_text())
        scene_path = (first.parent / meta["scene"]).resolve()
        model_path = (first.parent / meta["model"]).resolve()
        scene = json.loads(scene_path.read_text())
        model = mujoco.MjModel.from_binary_path(str(model_path))
        visual = IsaacVisual(
            app, model, job["output"], scene["config"].get("isaac_scene"), scene["manifest"]["source"]
        )
        job["startup_s"] = booted - boot_started
        job["scene_load_s"] = time.perf_counter() - booted
        with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as pool:
            render_episodes(visual, job, pool)
    finally:
        failed = sys.exc_info()[0] is not None
        if failed:
            import traceback

            traceback.print_exc()
        if visual is not None:
            visual.close()
        # Kit's fast shutdown exits the interpreter; report failures before closing it.
        if failed:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
        app.close()


if __name__ == "__main__":
    run_jobs(json.loads(Path(sys.argv[1]).read_text()))
