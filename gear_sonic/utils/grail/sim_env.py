"""GRAIL scene plugged into the existing MuJoCo/DDS simulator."""

from pathlib import Path
from threading import Event

import mujoco

from gear_sonic.utils.mujoco_sim.base_sim import DefaultEnv

from .assets import prepared_object_keys
from .runtime import GrailRuntime
from .scene import build_scene


class GrailEnv(DefaultEnv):
    def __init__(self, *args, **kwargs):
        self.reset_requested = Event()
        super().__init__(*args, **kwargs)
        self.torque_limit = self.mj_model.actuator_ctrlrange[:, 1].copy()
        self.grail = GrailRuntime(self)

    def viewer_key_callback(self, key):
        import glfw

        if key == glfw.KEY_BACKSPACE:
            # Viewer callbacks can run on its UI thread. Reset only inside the
            # physics loop, so snapshots never mix two episodes.
            self.reset_requested.set()

    def sim_step(self):
        if self.reset_requested.is_set():
            self.reset_requested.clear()
            self.reset()
        super().sim_step()

    def compute_torques(self):
        monitor = self.grail.reward_value
        if monitor is None:
            return super().compute_torques()
        bridge = self.unitree_bridge
        with bridge.low_cmd_lock, bridge.left_hand_cmd_lock, bridge.right_hand_cmd_lock:
            torques = super().compute_torques()
            monitor.capture_target()
        return torques

    def load_model(self, xml_path):
        cache_dir = self.config["GRAIL_CACHE_DIR"]
        self.scene_inputs = dict(
            robot_scene=str(Path(xml_path).resolve()),
            scene_config=str(Path(self.config["GRAIL_SCENE_CONFIG"]).resolve()),
            dataset_dir=str(Path(self.config["GRAIL_DATASET_DIR"]).resolve()),
            cache_dir=str(Path(cache_dir).resolve()) if cache_dir is not None else None,
        )
        object_key = self.config["GRAIL_OBJECT_KEY"]
        model, self.grail_scene, self.grail_manifest = build_scene(
            object_key=object_key, **self.scene_inputs
        )
        self.object_keys = []
        if self.config.get("GRAIL_CYCLE_OBJECTS"):
            self.object_keys = prepared_object_keys(
                self.scene_inputs["dataset_dir"], self.scene_inputs["cache_dir"]
            )
        self.grail_object_cycle = (
            dict(object_key=object_key, scene_inputs=self.scene_inputs)
            if len(self.object_keys) > 1 else None
        )
        return model

    def prepare_next_scene(self):
        current = self.object_keys.index(self.grail_manifest["object_key"])
        for key in self.object_keys[current + 1:] + self.object_keys[:current]:
            try:
                return build_scene(object_key=key, **self.scene_inputs)
            except (OSError, EOFError, ValueError, KeyError, TypeError) as exc:
                print(f"[GRAIL] Skipping object {key}: {exc}", flush=True)
        return None

    def prepare_scene_data(self, model):
        model.opt.timestep = self.sim_dt
        data = mujoco.MjData(model)
        if self.viewer is not None:
            # MuJoCo 3.17's passive Handle has no public model reload method.
            viewer = self.viewer._get_sim()
            if viewer is not None:
                mujoco.mj_forward(model, data)
                viewer.load(model, data, "")
        return data

    def replace_scene(self, model, config, manifest, data):
        # Object assets only modify the task body. Robot control indices are
        # identical across the scenes built from the same robot MJCF.
        self.mj_model, self.mj_data = model, data
        self.grail_scene, self.grail_manifest = config, manifest
        if self.image_publish_process is not None:
            self.image_publish_process.object_key = manifest["object_key"]
        if self.offscreen and not self.isolated_rendering:
            for renderer in self.renderers.values():
                renderer.close()
            self.init_renderers()

    def update_viewer(self):
        # sync takes the viewer lock and can block until a pending load finishes.
        if self.grail.scene_data_future is None:
            super().update_viewer()

    def reset(self):
        if hasattr(self, "grail"):
            monitor = self.grail.reward_value
            if monitor is not None and self.grail.phase == "running" and getattr(self, "fall", False):
                monitor.finish(self.grail.episode, self.grail.step_count + 1, timeout=False)
            self.grail.event("reset_requested")
            self.grail.reset()
        else:
            super().reset()
