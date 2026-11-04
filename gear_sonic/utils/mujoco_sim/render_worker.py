"""Camera subprocess entry point. Isaac must initialize before native imports."""

import json
import os
from pathlib import Path
import signal
import sys
import time


def create_camera_app():
    """Bootstrap only the pinned Isaac 5.0 camera experience, before native imports."""
    from isaacsim import SimulationApp

    class CameraSimulationApp(SimulationApp):
        def _wait_for_viewport(self):
            # Isaac 5.0 waits for an unused default viewport before the camera
            # exists. Explicit RGB warm-up below provides the readiness check.
            pass

    # Settings adapted from SIMPLE's split renderer (MIT, SIMPLE Authors 2025-2026).
    return CameraSimulationApp({
        "headless": True, "width": 640, "height": 480,
        "renderer": "RaytracedLighting", "multi_gpu": False,
        # NVIDIA's default fast exit avoids Kit 107's Python UI destructor crash.
        "fast_shutdown": True,
        "anti_aliasing": 0,
        "create_new_stage": False,
        "disable_viewport_updates": True,
        "extra_args": ["--/exts/omni.kit.renderer.core/present/enabled=false",
                       "--/rtx/ecoMode/enabled=false",
                       "--ext-folder", os.path.join(os.environ["ISAAC_PATH"], "extscache")],
    }, experience=str(Path(__file__).with_name("isaac_renderer.kit")))


def run(options):
    # Kit forwards unknown Python argv to its own launcher. The private JSON
    # transport config has already been read; it is not a USD file to open.
    sys.argv[:] = sys.argv[:1]
    # This process only uses MuJoCo offscreen. EGL also works without DISPLAY;
    # leave the parent's GLFW viewer and any explicit user choice untouched.
    os.environ.setdefault("MUJOCO_GL", "egl")
    app = None
    isaac = None
    renderers = {}
    servers = {}
    context = None
    socket = None
    stopping = False
    failed = False
    preview_created = False
    preview_title = "Isaac Sim RTX - ego_view"

    def preview_is_open():
        if not preview_created:
            return True
        import cv2
        try:
            key = cv2.waitKey(1) & 0xFF
            return key not in (27, ord("q"), ord("Q")) and cv2.getWindowProperty(
                preview_title, cv2.WND_PROP_VISIBLE
            ) >= 1
        except cv2.error:
            return False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        if options["backend"] in ("isaac", "both"):
            # Kit stays offscreen in both modes. A small RGB preview below keeps
            # the renderer independent of Kit's native GUI/present scheduling.
            app = create_camera_app()

        import msgpack
        import mujoco
        import numpy as np
        import zmq
        from gear_sonic.camera.sensor_server import ImageMessageSchema, SensorServer

        model = mujoco.MjModel.from_binary_path(options["model_path"])
        data = mujoco.MjData(model)
        if app is not None:
            from gear_sonic.utils.mujoco_sim.isaac_visual import IsaacVisual

            isaac = IsaacVisual(app, model, os.path.dirname(options["model_path"]),
                                scene_config=options["isaac_scene"],
                                object_asset_path=options["isaac_object_asset"])
        if options["backend"] in ("mujoco", "both"):
            for name, config in options["camera_configs"].items():
                renderers[name] = mujoco.Renderer(model, height=config["height"], width=config["width"])

        ports = {options["backend"]: options["camera_port"]}
        if options["backend"] == "both":
            ports = {"isaac": options["camera_port"], "mujoco": options["comparison_camera_port"]}
        for name, port in ports.items():
            servers[name] = SensorServer()
            servers[name].start_server(port)

        def publish(name, camera_images, timestamp):
            message = ImageMessageSchema(
                images=camera_images,
                timestamps={key: timestamp for key in camera_images},
            )
            servers[name].send_message(message.serialize())

        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.CONFLATE, 1)
        socket.setsockopt(zmq.SUBSCRIBE, b"")
        socket.connect(options["endpoint"])
        print("[Render] Ready; receiving latest MuJoCo snapshots.", flush=True)
        deadline = 0.0
        last_sequence = -1
        frames = 0
        report_start = time.monotonic()
        latency_sum = 0.0
        while not stopping and os.getppid() == options["parent_pid"]:
            if app is not None and (not app.is_running() or not preview_is_open()):
                break
            # app.update() renders the full scene even with no new state.
            # Only advance Kit inside the synchronized camera capture; the
            # OpenCV preview is serviced by preview_is_open() above.
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(0.005, remaining))
                continue
            if not socket.poll(timeout=5):
                continue
            packet = msgpack.unpackb(socket.recv(), raw=False)
            # After SIGSTOP/startup, IPC can still have bytes in the kernel even
            # with CONFLATE. Give the ZMQ I/O thread one short quiet interval to
            # drain that burst before selecting a snapshot for this frame.
            drain_deadline = time.monotonic() + 0.005
            while time.monotonic() < drain_deadline and socket.poll(timeout=1):
                packet = msgpack.unpackb(socket.recv(), raw=False)
            if packet["sequence"] <= last_sequence:
                continue
            last_sequence = packet["sequence"]
            qpos = np.frombuffer(packet["qpos"], dtype="<f8")
            if qpos.shape != (model.nq,) or not np.isfinite(qpos).all():
                raise ValueError("Invalid MuJoCo state snapshot")
            data.qpos[:] = qpos
            for name, width in (("mocap_pos", 3), ("mocap_quat", 4)):
                values = np.frombuffer(packet[name], dtype="<f8").reshape(model.nmocap, width)
                if not np.isfinite(values).all():
                    raise ValueError(f"Invalid {name} snapshot")
                getattr(data, name)[:] = values
            data.time = packet["sim_time"]
            # Forward kinematics only. Never mj_step, World.step or timeline.play.
            mujoco.mj_forward(model, data)
            started = time.monotonic()
            if isaac is not None:
                isaac_image = isaac.render(data)
                # Send immediately, before the comparison renderer starts.
                publish("isaac", {"ego_view": isaac_image}, packet["timestamp"])
            if renderers:
                camera_images = {}
                for name, renderer in renderers.items():
                    config = options["camera_configs"][name]
                    renderer.update_scene(data, camera=config.get("mjcf_name", name))
                    camera_images[name] = renderer.render()
                publish("mujoco", camera_images, packet["timestamp"])
            if isaac is not None and not options["headless"]:
                import cv2
                if not preview_created:
                    cv2.namedWindow(preview_title, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(preview_title, 640, 480)
                    preview_created = True
                cv2.imshow(preview_title, np.ascontiguousarray(isaac_image[..., ::-1]))
            frames += 1
            latency_sum += time.time() - packet["timestamp"]
            now = time.monotonic()
            if now - report_start >= 5.0:
                print(f"[Render] {frames / (now - report_start):.1f} FPS; "
                      f"mean sample-to-publish {latency_sum / frames * 1000:.1f} ms; "
                      f"sequence {last_sequence}, sim {data.time:.3f}s", flush=True)
                frames, latency_sum, report_start = 0, 0.0, now
            deadline = started + 1.0 / options["fps"]
    except Exception:
        # Kit disables its stderr/log bridge during close(); print the cause
        # before cleanup so a failed renderer always leaves a useful error.
        import traceback
        failed = True
        traceback.print_exc()
        raise
    finally:
        if preview_created:
            import cv2
            try:
                cv2.destroyWindow(preview_title)
                cv2.waitKey(1)
            except cv2.error:
                pass  # The user may already have destroyed the preview window.
        if socket is not None:
            socket.close(linger=0)
        if context is not None:
            context.term()
        for server in servers.values():
            server.stop_server()
        for renderer in renderers.values():
            renderer.close()
        try:
            if isaac is not None:
                isaac.close()
        except Exception:
            import traceback
            failed = True
            traceback.print_exc()
        finally:
            if app is not None:
                if failed:
                    # Kit fast shutdown otherwise forces status 0, masking the
                    # exception. The OS releases this isolated process's GPU
                    # objects; the parent reaps it and removes the IPC directory.
                    sys.stdout.flush()
                    sys.stderr.flush()
                    os._exit(1)
                app.close()


if __name__ == "__main__":
    with open(sys.argv[1]) as stream:
        run(json.load(stream))
