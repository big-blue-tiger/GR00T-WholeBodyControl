"""Bounded, spawn-only CPU pose preparation and shared-memory video encoding.

No Isaac or USD imports: safe to import in the supervisor and spawned workers.
"""

from contextlib import ExitStack
from functools import lru_cache
import json
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
import queue
import time
import traceback

import numpy as np

SHAPE = (480, 640, 3)
ROBOT_MJCF = Path(__file__).resolve().parents[2] / "data/robot_model/model_data/g1/g1_29dof_with_hand.xml"


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


class PixelStats:
    """Exact integer histograms, normalized only once at episode completion."""

    def __init__(self):
        self.histogram = np.zeros((3, 256), dtype=np.int64)
        self.frames = 0

    def add(self, rgb):
        for channel in range(3):
            self.histogram[channel] += np.bincount(rgb[..., channel].ravel(), minlength=256)
        self.frames += 1

    def result(self):
        values = np.arange(256, dtype=np.float64) / 255
        weights = self.histogram / self.histogram.sum(axis=1, keepdims=True)
        mean = weights @ values
        second = weights @ (values * values)
        present = self.histogram != 0
        result = dict(
            min=present.argmax(axis=1) / 255,
            max=(255 - present[:, ::-1].argmax(axis=1)) / 255,
            mean=mean,
            std=np.sqrt(np.maximum(second - mean * mean, 0)),
        )
        return {
            **{k: v.reshape(3, 1, 1).tolist() for k, v in result.items()},
            "count": [self.frames],
        }


@lru_cache(maxsize=1)
def _reference_camera_model():
    import xml.etree.ElementTree as ET

    import mujoco

    camera = ET.parse(ROBOT_MJCF).find(".//camera[@name='head_camera']")
    return mujoco.MjModel.from_xml_string(
        f"<mujoco><worldbody>{ET.tostring(camera, encoding='unicode')}</worldbody></mujoco>"
    )


@lru_cache(maxsize=1)
def load_replay_model(path):
    """Apply the current RGB calibration to snapshots with the old stock camera."""
    import mujoco

    model = mujoco.MjModel.from_binary_path(str(path))
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
    if camera_id < 0:
        return model
    if (not model.cam_sensorsize[camera_id].any()
            and model.body(model.cam_bodyid[camera_id]).name == "torso_link"
            and np.allclose(model.cam_pos[camera_id], [0.06, 0.0, 0.45], rtol=0, atol=1e-9)):
        reference = _reference_camera_model()
        for field in ("cam_pos", "cam_quat", "cam_fovy", "cam_resolution", "cam_sensorsize", "cam_intrinsic"):
            getattr(model, field)[camera_id] = getattr(reference, field)[0]
    return model


def prepare_episode(path):
    """Calculate full forward kinematics off the Kit thread; no physics steps."""
    import mujoco

    from .offline_data import load_episode

    started = time.perf_counter()
    path = Path(path)
    meta, trace = load_episode(path)
    model = load_replay_model((path.parent / meta["model"]).resolve())
    data = mujoco.MjData(model)
    n = meta["transitions"]
    fields = ("xpos", "xmat", "geom_xpos", "geom_xmat", "cam_xpos", "cam_xmat")
    poses = {key: np.empty((n, *getattr(data, key).shape)) for key in fields}
    for index in range(n):
        for key in ("qpos", "qvel", "mocap_pos", "mocap_quat"):
            getattr(data, key)[:] = trace[key][index]
        data.time = float(trace["sim_time"][index])
        mujoco.mj_forward(model, data)
        for key in fields:
            poses[key][index] = getattr(data, key)
    return dict(
        path=str(path),
        meta=meta,
        poses=poses,
        timestamps=trace["sim_time"][:n],
        prepare_s=time.perf_counter() - started,
    )


def _encoder(messages, events, free, names):
    import av

    state = None
    try:
        with ExitStack() as stack:
            buffers = []
            for name in names:
                memory = SharedMemory(name=name)
                stack.callback(memory.close)
                buffers.append(np.ndarray(SHAPE, dtype=np.uint8, buffer=memory.buf))
            while True:
                message = messages.get()
                if message is None:
                    if state is not None:
                        raise RuntimeError("encoder stopped with unfinished episodes")
                    return
                kind, episode, payload = message
                if kind == "start":
                    destination = Path(payload["destination"])
                    destination.mkdir(parents=True, exist_ok=True)
                    (destination / "render.json").unlink(missing_ok=True)
                    container = av.open(str(destination / "ego_view.partial.mp4"), "w")
                    stream = container.add_stream("libx264", rate=50)
                    stream.width, stream.height, stream.pix_fmt = 640, 480, "yuv420p"
                    stream.options = {"crf": "18", "preset": "fast"}
                    # Avoid CPU oversubscription with four independent encoders.
                    stream.codec_context.thread_count = 2
                    state = dict(
                        container=container,
                        stream=stream,
                        stats=PixelStats(),
                        count=0,
                        stats_s=0.0,
                        encode_s=0.0,
                        **payload,
                    )
                elif kind == "frame":
                    index, block = payload
                    try:
                        if index != state["count"]:
                            raise ValueError(f"out-of-order frame {episode}: {index}")
                        rgb = buffers[block]
                        begin = time.perf_counter()
                        state["stats"].add(rgb)
                        state["stats_s"] += time.perf_counter() - begin
                        begin = time.perf_counter()
                        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                        frame.pts = index
                        for packet in state["stream"].encode(frame):
                            state["container"].mux(packet)
                        state["encode_s"] += time.perf_counter() - begin
                        state["count"] += 1
                    finally:
                        free.put(block)
                elif kind == "finish":
                    container = state["container"]
                    try:
                        begin = time.perf_counter()
                        for packet in state["stream"].encode():
                            container.mux(packet)
                        state["encode_s"] += time.perf_counter() - begin
                    finally:
                        container.close()
                    n = state["count"]
                    if n != state["expected"]:
                        raise ValueError(f'incomplete episode {episode}: {n}/{state["expected"]}')
                    destination = Path(state["destination"])
                    partial = destination / "ego_view.partial.mp4"
                    with av.open(str(partial)) as check:
                        video = check.streams.video[0]
                        if video.frames != n or video.average_rate != 50:
                            raise ValueError(f"invalid encoded frame count/rate: {episode}")
                    partial.replace(destination / "ego_view.mp4")
                    temporary = destination / "frames.partial.npz"
                    np.savez(temporary, frame_index=np.arange(n), timestamp=state["timestamps"])
                    temporary.replace(destination / "frames.npz")
                    result = dict(
                        state["result"],
                        **payload,
                        frames=n,
                        fps=50,
                        width=640,
                        height=480,
                        video_stats=state["stats"].result(),
                        stats_s=state["stats_s"],
                        encode_s=state["encode_s"],
                    )
                    elapsed = time.perf_counter() - state["started"]
                    result.update(wall_s=elapsed, frames_per_second=n / elapsed)
                    atomic_json(destination / "render.json", result)
                    events.put(("done", episode, result))
                    state = None
                else:
                    raise ValueError(f"unknown encoder message: {kind}")
    except BaseException:
        events.put(("error", None, traceback.format_exc()))
        raise
    finally:
        if state is not None:
            state["container"].close()


class EncoderPool:
    """Fixed-size RGB ring; every episode has one ordered encoder owner."""

    def __init__(self, count):
        context = mp.get_context("spawn")
        self.free = context.Queue()
        self.events = context.Queue()
        self.messages = [context.Queue(maxsize=32) for _ in range(count)]
        self.buffers = []
        self.processes = []
        self.finished = {}
        self.queue_wait_s = 0.0
        try:
            for index in range(count * 4):
                memory = SharedMemory(create=True, size=int(np.prod(SHAPE)))
                self.buffers.append(memory)
                self.free.put(index)
            names = [memory.name for memory in self.buffers]
            for messages in self.messages:
                process = context.Process(
                    target=_encoder, args=(messages, self.events, self.free, names)
                )
                process.start()
                self.processes.append(process)
        except BaseException:
            self.close(abort=True)
            raise

    def poll(self):
        while True:
            try:
                kind, episode, result = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "error":
                raise RuntimeError("video encoder failed:\n" + result)
            self.finished[episode] = result
        for process in self.processes:
            if process.exitcode is not None:
                raise RuntimeError(f"video encoder exited unexpectedly: {process.exitcode}")

    def send(self, kind, episode, payload):
        messages = self.messages[episode % len(self.messages)]
        while True:
            self.poll()
            try:
                messages.put((kind, episode, payload), timeout=0.2)
                return
            except queue.Full:
                pass

    def frame(self, episode, index, rgb):
        begin = time.perf_counter()
        while True:
            self.poll()
            try:
                block = self.free.get(timeout=0.2)
                break
            except queue.Empty:
                pass
        self.queue_wait_s += time.perf_counter() - begin
        np.copyto(np.ndarray(SHAPE, dtype=np.uint8, buffer=self.buffers[block].buf), rgb)
        self.send("frame", episode, (index, block))

    def wait(self, episodes):
        while not set(episodes).issubset(self.finished):
            self.poll()
            time.sleep(0.01)

    def close(self, abort=False):
        try:
            if not abort:
                for messages in self.messages:
                    messages.put(None, timeout=5)
            for process in self.processes:
                if abort and process.is_alive():
                    process.terminate()
                process.join(timeout=10)
                if process.is_alive():
                    process.kill()
                    process.join()
        finally:
            for memory in self.buffers:
                memory.close()
                memory.unlink()
            for messages in [*self.messages, self.events, self.free]:
                messages.cancel_join_thread()
                messages.close()
