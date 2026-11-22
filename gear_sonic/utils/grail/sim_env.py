"""GRAIL scene plugged into the existing MuJoCo/DDS simulator."""

from threading import Event

from gear_sonic.utils.mujoco_sim.base_sim import DefaultEnv

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
        model, self.grail_scene, self.grail_manifest = build_scene(
            xml_path,
            self.config["GRAIL_SCENE_CONFIG"],
            self.config["GRAIL_DATASET_DIR"],
            self.config["GRAIL_OBJECT_KEY"],
            self.config["GRAIL_CACHE_DIR"],
        )
        return model

    def reset(self):
        if hasattr(self, "grail"):
            monitor = self.grail.reward_value
            if monitor is not None and self.grail.phase == "running" and getattr(self, "fall", False):
                monitor.finish(self.grail.episode, self.grail.step_count + 1, timeout=False)
            self.grail.event("reset_requested")
            self.grail.reset()
        else:
            super().reset()
