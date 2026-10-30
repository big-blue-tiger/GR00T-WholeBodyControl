"""Nonblocking, latest-state IPC and lifecycle for the camera process.

Only qpos crosses this boundary. No renderer or Isaac module is imported here.
"""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

import msgpack
import mujoco
import numpy as np
import zmq


class StateRenderProcess:
    def __init__(self, model, camera_configs, backend, fps, camera_port,
                 comparison_camera_port, headless):
        self.process = None
        self.context = None
        self.socket = None
        self._closed = False
        self._failed = False
        self.sequence = 0
        self.directory = tempfile.TemporaryDirectory(prefix="sonic-render-")
        try:
            directory = Path(self.directory.name)
            self.endpoint = f"ipc://{directory / 'state.sock'}"
            # MJB embeds meshes/assets and the exact compiled physics model.
            model_path = directory / "model.mjb"
            mujoco.mj_saveModel(model, str(model_path))
            options = dict(
                endpoint=self.endpoint, model_path=str(model_path), backend=backend,
                fps=fps, camera_port=camera_port,
                comparison_camera_port=comparison_camera_port, headless=headless,
                camera_configs=camera_configs, parent_pid=os.getpid(),
            )
            config_path = directory / "render.json"
            config_path.write_text(json.dumps(options))
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.PUB)
            self.socket.setsockopt(zmq.LINGER, 0)
            self.socket.setsockopt(zmq.CONFLATE, 1)
            self.socket.bind(self.endpoint)
            environment = os.environ.copy()
            if backend in ("isaac", "both"):
                # Set PRIME variables before interpreter/native loader startup.
                environment.setdefault("__NV_PRIME_RENDER_OFFLOAD", "1")
                environment.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")
                environment.setdefault("__VK_LAYER_NV_optimus", "NVIDIA_only")
            self.process = subprocess.Popen(
                [sys.executable, "-m", "gear_sonic.utils.mujoco_sim.render_worker", str(config_path)],
                start_new_session=True,
                env=environment,
            )
            print(f"[Render] {backend} worker PID {self.process.pid}; physics does not wait for startup.", flush=True)
        except BaseException:
            # Also clean up if Ctrl-C/SIGTERM interrupts worker creation.
            self.stop()
            raise

    def publish(self, data):
        if self._failed or self._closed:
            return
        self.sequence += 1
        packet = msgpack.packb(dict(
            sequence=self.sequence, sim_time=float(data.time), timestamp=time.time(),
            qpos=np.asarray(data.qpos, dtype="<f8").tobytes(),
        ), use_bin_type=True)
        try:
            self.socket.send(packet, flags=zmq.NOBLOCK)
        except zmq.Again:
            pass
        except zmq.ZMQError as exc:
            self._failed = True
            print(f"[Render] State IPC failed: {exc}; MuJoCo/DDS continue.", flush=True)

    def check_health(self):
        if not self._failed and self.process is not None and self.process.poll() is not None:
            self._failed = True
            print(f"[Render] Worker exited ({self.process.returncode}); MuJoCo/DDS continue. "
                  "No automatic restart or backend switch.", flush=True)

    def stop(self):
        if self._closed:
            return
        self._closed = True
        if self.process is not None:
            try:
                # Also reap a worker deliberately paused with SIGSTOP during validation.
                os.killpg(self.process.pid, signal.SIGTERM)
                os.killpg(self.process.pid, signal.SIGCONT)
                self.process.wait(timeout=5)
            except ProcessLookupError:
                self.process.wait()
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait()
        if self.socket is not None:
            self.socket.close(linger=0)
        if self.context is not None:
            self.context.term()
        self.directory.cleanup()
