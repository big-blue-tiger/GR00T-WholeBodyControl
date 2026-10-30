"""Entry point for running a MuJoCo simulation loop with the G1 robot model.

Parses a YAML-based WBC config via tyro CLI, instantiates the G1 robot model,
and launches the simulator (optionally with offscreen image publishing).
"""

import math
import signal
from typing import Dict

import tyro

from gear_sonic.utils.mujoco_sim.simulator_factory import SimulatorFactory, init_channel
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
from gear_sonic.data.robot_model.instantiation.g1 import (
    instantiate_g1_robot_model,
)
from gear_sonic.data.robot_model.robot_model import RobotModel

ArgsConfig = SimLoopConfig


class SimWrapper:
    def __init__(self, robot_model: RobotModel, env_name: str, config: Dict[str, any], **kwargs):
        self.robot_model = robot_model
        self.config = config

        init_channel(config=self.config)

        # Create simulator using factory
        self.sim = SimulatorFactory.create_simulator(
            config=self.config,
            env_name=env_name,
            **kwargs,
        )


def main(config: ArgsConfig):
    # Service/shell termination must follow the same worker/IPC cleanup as Ctrl-C.
    def terminate(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    if not math.isfinite(config.render_fps) or config.render_fps <= 0:
        raise ValueError("render_fps must be finite and positive")
    if not 1 <= config.camera_port <= 65535 or not 1 <= config.comparison_camera_port <= 65535:
        raise ValueError("Camera ports must be between 1 and 65535")
    if config.render_backend == "both" and config.camera_port == config.comparison_camera_port:
        raise ValueError("Both renderers need distinct camera ports")
    if config.render_backend != "mujoco":
        config.enable_image_publish = True
    wbc_config = config.load_wbc_yaml()
    # NOTE: we will override the interface to local if it is not specified
    wbc_config["ENV_NAME"] = config.env_name

    wbc_config.update(
        RENDER_BACKEND=config.render_backend,
        ISAAC_HEADLESS=config.isaac_headless,
        RENDER_FPS=config.render_fps,
        COMPARISON_CAMERA_PORT=config.comparison_camera_port,
    )
    if config.enable_image_publish and config.render_backend == "mujoco":
        assert (
            config.enable_offscreen
        ), "enable_offscreen must be True when enable_image_publish is True"

    robot_model = instantiate_g1_robot_model()

    sim_wrapper = SimWrapper(
        robot_model=robot_model,
        env_name=config.env_name,
        config=wbc_config,
        onscreen=wbc_config.get("ENABLE_ONSCREEN", True),
        offscreen=wbc_config.get("ENABLE_OFFSCREEN", False),
        enable_image_publish=config.enable_image_publish,
    )
    # Physics stays here; cameras run in a fresh, independent interpreter.
    SimulatorFactory.start_simulator(
        sim_wrapper.sim,
        as_thread=False,
        enable_image_publish=config.enable_image_publish,
        mp_start_method=config.mp_start_method,
        camera_port=config.camera_port,
    )


if __name__ == "__main__":
    config = tyro.cli(ArgsConfig)
    main(config)
