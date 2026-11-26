"""A resident NuRec camera paired with the live Isaac mesh camera."""

import multiprocessing as mp
import os
from pathlib import Path
import sys
import traceback

import numpy as np


class BackgroundRenderError(RuntimeError):
    pass


def uses_layered_rendering(scene_config):
    return bool(
        scene_config and scene_config.get("enabled") and scene_config.get("occlusion_proxy")
        and not scene_config.get("show_colliders")
    )


def occlusion_proxy_path(scene_config):
    return Path(__file__).resolve().parents[3] / scene_config["occlusion_proxy"]["asset_path"]


def _render_background(connection, pixels, model_path, scene_config, shape, model_loader):
    from .render_worker import create_camera_app

    sys.argv[:] = sys.argv[:1]
    try:
        app = create_camera_app()

        import mujoco

        from .isaac_visual import IsaacVisual

        # Replay applies legacy camera calibration in both render processes.
        model = (
            mujoco.MjModel.from_binary_path(model_path)
            if model_loader is None else model_loader(Path(model_path))
        )
        visual = IsaacVisual(app, model, Path(model_path).parent, scene_config, background_only=True)
        if (*visual.puppet.capture_size[::-1], 3) != shape:
            raise ValueError("Isaac background and foreground resolutions differ")
        target = np.frombuffer(pixels, dtype=np.uint8).reshape(shape)
        connection.send(None)
        while True:
            try:
                sequence, position, rotation = connection.recv()
            except EOFError:
                break
            visual.puppet.set_pose(position, rotation)
            target[:] = visual.capture_rgb()[0]
            try:
                connection.send(sequence)
            except BrokenPipeError:
                break
        visual.close()
    except Exception:
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)  # Preserve failures despite Kit fast shutdown.
    finally:
        connection.close()
    app.close()


class BackgroundRenderer:
    def __init__(self, model_path, scene_config, capture_size, *, model_loader=None):
        context = mp.get_context("spawn")
        shape = (*capture_size[::-1], 3)
        pixels = context.RawArray("B", int(np.prod(shape)))
        self.rgb = np.frombuffer(pixels, dtype=np.uint8).reshape(shape)
        self.connection, child = context.Pipe()
        self.process = context.Process(
            target=_render_background,
            args=(child, pixels, str(model_path), scene_config, shape, model_loader),
            name="Isaac NuRec background",
        )
        try:
            try:
                self.process.start()
            finally:
                child.close()
            # The first NuRec frame may have to compile ray-tracing shaders.
            self._wait(None, 240)
        except BaseException:
            self.close()
            raise

    def submit(self, sequence, position, rotation):
        try:
            self.connection.send((sequence, position, rotation))
        except OSError as exc:
            raise BackgroundRenderError("Isaac background pipe closed before capture") from exc

    def receive(self, sequence):
        self._wait(sequence, 5)
        # The receiver finishes the composition before sending the next request.
        return self.rgb

    def _wait(self, expected, timeout):
        try:
            if not self.connection.poll(timeout):
                raise BackgroundRenderError(f"Isaac background timed out after {timeout} seconds")
            actual = self.connection.recv()
        except (EOFError, OSError) as exc:
            raise BackgroundRenderError("Isaac background process exited during capture") from exc
        if actual != expected:
            raise BackgroundRenderError(f"Isaac background returned sequence {actual}, expected {expected}")

    def close(self):
        self.connection.close()
        if self.process.pid is not None:
            self.process.join(timeout=1)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=1)
            if self.process.is_alive():
                self.process.kill()
                self.process.join()
        self.process.close()
