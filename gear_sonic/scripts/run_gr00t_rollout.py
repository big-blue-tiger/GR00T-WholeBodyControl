"""Run the red-bin task with GR00T inference and one Isaac RGB frame per step."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from functools import partial
from itertools import cycle
import os
from pathlib import Path
import signal
import sys
import time
import traceback

from gear_sonic.utils.grail.offline_task import DEFAULT_TASK_TEMPLATE

ROOT = Path(__file__).resolve().parents[2]
INFERENCE_STEPS = 20
SYNC_INFERENCE_STEPS = 10
CHUNK_STEPS = 40
HOLD_STEPS = 10
# Indices in the offline simulator's COMMAND_NAMES, in the training export order.
# Kit and Pinocchio load incompatible native dependencies in the same process.
STATE_GROUPS = {
    "left_leg": slice(0, 6),
    "right_leg": slice(6, 12),
    "waist": slice(12, 15),
    "left_arm": slice(15, 22),
    "right_arm": slice(22, 29),
    "left_hand": [32, 33, 34, 35, 29, 30, 31],
    "right_hand": [39, 40, 41, 42, 36, 37, 38],
}


def policy_connection(context, port):
    import zmq

    socket = context.socket(zmq.REQ)
    socket.linger = 0
    socket.rcvtimeo = socket.sndtimeo = 15_000
    socket.connect(f"tcp://127.0.0.1:{port}")
    return socket


def preview_publisher(context, port):
    import zmq

    socket = context.socket(zmq.PUB)
    socket.linger = 0
    socket.sndhwm = 8
    socket.bind(f"tcp://0.0.0.0:{port}")
    return socket


def request_action(socket, episode_id, step, rgb, state, prompt):
    import msgpack
    import numpy as np
    import zmq

    started = time.monotonic()
    request = dict(
        endpoint="get_action_rgb",
        data=dict(episode_id=episode_id, step=step, rgb=rgb.tobytes(), state=state, prompt=prompt),
    )
    try:
        socket.send(msgpack.packb(request, use_bin_type=True))
        reply = msgpack.unpackb(socket.recv(), raw=False)
    except zmq.Again as exc:
        raise TimeoutError(f"GR00T timed out for episode {episode_id}, observation step {step}") from exc
    if "error" in reply:
        raise RuntimeError(reply["error"])
    if (reply["episode_id"], reply["step"]) != (episode_id, step):
        raise ValueError("GR00T reply does not match its observation")
    actions = np.frombuffer(reply["actions"], dtype="<f4").reshape(CHUNK_STEPS, 78).copy()
    if not np.isfinite(actions).all():
        raise ValueError("GR00T returned nonfinite actions")
    tokens = actions[:, :64]
    tokens[:] = np.clip(np.rint(tokens * 16), -16, 15) / 16
    return episode_id, step, actions, time.monotonic() - started


def submit_observation(worker, socket, env, rgb):
    from gear_sonic.utils.data_collection.transforms import compute_projected_gravity

    joints = env.data.qpos[env.qadr]
    state = {key: joints[indices] for key, indices in STATE_GROUPS.items()}
    state["projected_gravity"] = compute_projected_gravity(env.data.qpos[3:7])
    state_bytes = {key: values.astype("<f4", copy=False).tobytes() for key, values in state.items()}
    return worker.submit(
        request_action, socket, env.episode, env.step_index, rgb, state_bytes, env.task_description,
    )


def run_episode(env, visual, decoder, infer, encode, pending, *, synchronous=False):
    import numpy as np

    env.trace["observation_step"] = np.empty(env.steps, dtype=np.int32)
    env.trace["chunk_index"] = np.empty(env.steps, dtype=np.int32)
    env.trace["held_action"] = np.empty(env.steps, dtype=bool)
    for key in ("q", "dq", "kp", "kd", "tau"):
        env.trace[f"command_{key}"] = np.empty((env.steps, 43), dtype=np.float32)
    chunk, source_step = None, 0
    request_interval = SYNC_INFERENCE_STEPS if synchronous else INFERENCE_STEPS
    # Expire the sync chunk at the request boundary so physics waits for the next reply.
    usable_steps = request_interval if synchronous else CHUNK_STEPS + HOLD_STEPS
    last_request = -request_interval
    encoded = None
    for step in range(env.steps):
        started = time.monotonic()
        rgb = visual.render(env.data)
        while True:
            if pending is not None and pending.done():
                episode_id, observed, actions, elapsed = pending.result()
                pending = None
                if episode_id == env.episode and 0 <= step - observed < CHUNK_STEPS:
                    chunk, source_step = actions, observed
                    print(f"[GR00T] {episode_id}:{observed} -> step {step} in {elapsed:.3f}s", flush=True)
                else:
                    print(f"[GR00T] Discarded action chunk {episode_id}:{observed}", flush=True)

            offset = step - source_step
            has_action = chunk is not None and offset < usable_steps
            if pending is None and (not has_action or step - last_request >= request_interval):
                pending = infer(env, rgb)
                last_request = step
            if has_action:
                break
            # The current image is already available while physics waits for a valid chunk.
            pending.result()

        index = min(offset, CHUNK_STEPS - 1)
        action = chunk[index]
        if encoded is not None:
            encoded.result()
        reply, = decoder([dict(env.decoder_request(), action=action.tolist())])
        env.step(reply, action)
        env.trace["observation_step"][step] = source_step
        env.trace["chunk_index"][step] = index
        env.trace["held_action"][step] = offset >= CHUNK_STEPS
        encoded = encode(step, rgb)
        if not synchronous:
            time.sleep(max(0, 0.02 - (time.monotonic() - started)))

    encoded.result()
    return pending


def collect(args, app):
    import zmq

    from gear_sonic.utils.grail.assets import prepared_object_keys
    from gear_sonic.utils.grail.gr00t_video import EpisodeVideo
    from gear_sonic.utils.grail.offline_collect import Decoder, collection_objects, prepare_scene
    from gear_sonic.utils.grail.offline_render import scene_assets
    from gear_sonic.utils.grail.offline_sim import OfflineEnvironment, load_environment_scene
    from gear_sonic.utils.grail.render_pipeline import atomic_json
    from gear_sonic.utils.mujoco_sim.isaac_visual import IsaacVisual

    if args.cycle_objects and not args.object_key:
        keys = prepared_object_keys(args.dataset_dir, args.cache_dir)
    else:
        keys = collection_objects(args.dataset_dir, args.object_key, args.cache_dir)
    if args.cycle_objects and len(keys) < 2:
        raise ValueError(f"--cycle-objects requires at least two distinct prepared objects in {args.dataset_dir}")

    object_order = cycle(keys) if args.cycle_objects else keys
    traces, renders = args.output / "traces", args.output / "renders"
    traces.mkdir(parents=True)
    atomic_json(traces / "scene.json", dict(task=args.task, policy="gr00t", objects=keys))

    synchronous = args.inference_mode == "sync"
    with ExitStack() as stack:
        transport = stack.enter_context(zmq.Context())
        inference = stack.enter_context(ThreadPoolExecutor(max_workers=1))
        encoder = stack.enter_context(ThreadPoolExecutor(max_workers=1))
        client = inference.submit(policy_connection, transport, args.port).result()
        stack.callback(lambda: inference.submit(client.close).result())
        preview = None
        if not synchronous:
            preview = encoder.submit(preview_publisher, transport, args.preview_port).result()
            stack.callback(lambda: encoder.submit(preview.close).result())
        decoder = Decoder(args.decoder_executable, args.decoder_model, args.output / "decoder.log")
        stack.callback(decoder.close)
        infer = partial(submit_observation, inference, client)
        if preview is not None:
            print(f"[GR00T] H.264 preview listening on {args.preview_port}", flush=True)

        visual, pending = None, None
        episode = 0
        prepared = set()
        for key in object_order:
            scene_dir = traces / "objects" / key
            if key not in prepared:
                prepare_scene(args, scene_dir, key)
                prepared.add(key)
            model, scene, initial = load_environment_scene(scene_dir)
            env = OfflineEnvironment(model, scene, initial, 0, traces, args.steps, scene_dir)
            source, _, background = scene_assets(scene_dir / "scene.json")
            if visual is None:
                visual = IsaacVisual(
                    app, model, args.output, background, source, model_path=scene_dir / "model.mjb",
                )
                stack.callback(visual.close)
            else:
                visual.scene.set_object(model, source)

            for _ in range(args.episodes_per_object):
                env.start_episode(episode, args.seed + episode)
                print(f"[GR00T] Episode {episode} · {key} · {env.task_description}", flush=True)
                video = encoder.submit(
                    EpisodeVideo, renders / f"episode_{episode:06d}", f"{args.output.name}:{episode}", preview,
                ).result()
                started = time.monotonic()
                try:
                    encode = partial(encoder.submit, video.write)
                    pending = run_episode(env, visual, decoder, infer, encode, pending, synchronous=synchronous)
                    encoder.submit(video.finish, env.trace["sim_time"][:-1]).result()
                finally:
                    encoder.submit(video.close).result()
                path = env.save()
                rate = args.steps / (time.monotonic() - started)
                print(f"[GR00T] Saved {path} at {rate:.1f} control steps per wall second", flush=True)
                episode += 1
        decoder([])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model-dir", type=Path, required=True, help="Checkpoint used by the running policy server",
    )
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "gear_sonic/data/grail/100obj")
    parser.add_argument("--object-key", action="append")
    parser.add_argument(
        "--cycle-objects", action="store_true",
        help="Run until interrupted, advancing to the next prepared object after each episode",
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--initial-pose", type=Path, default=ROOT / "gear_sonic/config/grail/carry_initial_pose.json",
    )
    parser.add_argument("--scene-config", type=Path, default=ROOT / "gear_sonic/config/grail/apple_2.yaml")
    parser.add_argument(
        "--robot-scene", type=Path, default=ROOT / "gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml",
    )
    parser.add_argument(
        "--decoder-model", type=Path,
        default=ROOT / "gear_sonic_deploy/policy/grail_model/model_decoder_grail.onnx",
    )
    parser.add_argument(
        "--decoder-executable", type=Path, default=ROOT / "gear_sonic_deploy/target/release/grail_offline_decoder",
    )
    parser.add_argument("--episodes-per-object", type=int, default=1)
    parser.add_argument("--duration", type=float, default=25.)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", default=DEFAULT_TASK_TEMPLATE)
    parser.add_argument("--port", type=int, default=5550)
    parser.add_argument("--preview-port", type=int, default=5555)
    parser.add_argument(
        "--inference-mode", choices=("async", "sync"), default="async",
        help="Sync waits for a new chunk every 10 control steps and records locally without live preview",
    )
    args = parser.parse_args()
    args.steps = round(args.duration * 50)
    if args.steps < 1 or abs(args.steps / 50 - args.duration) > 1e-9:
        parser.error("--duration must be a positive multiple of 0.02 seconds")
    if args.episodes_per_object < 1:
        parser.error("--episodes-per-object must be positive")
    if args.inference_mode == "async" and args.port == args.preview_port:
        parser.error("policy and preview ports must be distinct in async mode")
    if args.cycle_objects and args.episodes_per_object != 1:
        parser.error("--cycle-objects requires one episode per object per pass")
    if not args.cycle_objects and args.object_key is None:
        args.object_key = ["pickup_table__alcohol_10__001"]
    args.output = args.output.resolve()
    args.dataset_dir = args.dataset_dir.resolve()
    args.checkpoint = args.model_dir.resolve()
    args.no_release_override = True

    from gear_sonic.utils.grail.offline_render import kit_environment
    from gear_sonic.utils.mujoco_sim.render_worker import create_camera_app

    os.environ.update(kit_environment())
    os.environ.setdefault("MUJOCO_GL", "egl")
    sys.argv[:] = sys.argv[:1]  # Kit also reads argv.
    app = create_camera_app()

    def interrupt(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    try:
        collect(args, app)
    except KeyboardInterrupt:
        print("[GR00T] Stopped. Completed episodes are ready to export.", flush=True)
    except Exception:
        # Kit's exit hook can discard Python's exit status on a failed run.
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    app.close(wait_for_replicator=False)


if __name__ == "__main__":
    main()
