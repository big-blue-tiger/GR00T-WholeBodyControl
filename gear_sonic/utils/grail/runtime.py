"""Simulation-side task, episode, actor IPC and existing C++ v4 bridge."""

import json
import pickle
import time
import uuid

import msgpack
import mujoco
import numpy as np
import zmq

from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message

from .observations import Observations
from .plan import CarryPlannerConfig
from .episode import reset_scene
from .task import CarryTask
from .transport import pack, socket, unpack


class GrailRuntime:
    def __init__(self, env):
        self.env, self.model, self.data = env, env.mj_model, env.mj_data
        self.cfg, self.manifest = env.grail_scene, env.grail_manifest
        self.rng = np.random.default_rng(env.config["GRAIL_SEED"])
        self.planner = CarryPlannerConfig(**self.cfg["planner"])
        self.observations = Observations(self.model, self.cfg, self.manifest["bps"])
        with open(env.config["GRAIL_INITIAL_POSE"], "rb") as stream:
            self.initial = pickle.load(stream)
        for key, size in (
            ("dof", 29),
            ("hand_dof_pos", 14),
            ("root_rot", 4),
            ("root_trans_offset", 3),
        ):
            if (
                np.asarray(self.initial[key]).shape != (size,)
                or not np.isfinite(self.initial[key]).all()
            ):
                raise ValueError(f"Invalid initial pose field {key}")
        self.context = zmq.Context()
        self.states = socket(
            self.context, zmq.PUB, env.config["GRAIL_OBSERVATION_ENDPOINT"], bind=True
        )
        self.actions = socket(self.context, zmq.SUB, env.config["GRAIL_ACTION_ENDPOINT"])
        self.decoder = socket(self.context, zmq.PUB, "tcp://127.0.0.1:5556", bind=True)
        self.debug = socket(self.context, zmq.SUB, "tcp://127.0.0.1:5557", topic=b"g1_debug")
        self.log = (
            open(env.config["GRAIL_LOG"], "a", buffering=1) if env.config.get("GRAIL_LOG") else None
        )
        self.session = uuid.uuid4().hex
        self.episode = 0
        self.sequence = 0
        self.last_report = time.monotonic()
        self.reset()

    def event(self, kind, **details):
        entry = dict(
            event=kind, episode_id=self.episode_id, sim_time=float(self.data.time), **details
        )
        if self.log:
            self.log.write(json.dumps(entry) + "\n")
        print("[GRAIL] " + json.dumps(entry), flush=True)

    def reset(self):
        self.episode += 1
        self.episode_id = f"{self.session}:{self.episode}"
        self.plan, details = reset_scene(
            self.model, self.data, self.cfg, self.manifest, self.initial, self.rng, self.planner
        )
        self.task = CarryTask(self.plan, self.planner.pickup_settle_s)
        self.observations.reset()
        self.phase = "flushing"
        self.reset_at = time.monotonic()
        self.next_tick = 0.0
        self.last_wait_tick = 0.0
        self.debug_count = 0
        self.last_debug_index = None
        self.last_action_sequence = -1
        self.pending = None
        self.active = None
        self.sent = {}
        self.arm_at = None
        self.step_count = 0
        self.reported_events = set()
        self.event("reset", **details)

    def send_decoder(self, token, left, right):
        self.decoder.send(
            pack_pose_message(
                dict(
                    token_state=np.asarray(token, dtype=np.float32).reshape(1, 64),
                    left_hand_joints=np.asarray(left, dtype=np.float32).reshape(1, 7),
                    right_hand_joints=np.asarray(right, dtype=np.float32).reshape(1, 7),
                    frame_index=np.array([self.sequence], dtype=np.int64),
                ),
                version=4,
            ),
            zmq.NOBLOCK,
        )

    def initial_hands(self):
        return [
            self.data.qpos[getattr(self.env, side + "_hand_qpos_adr")].copy()
            for side in ("left", "right")
        ]

    def poll_action(self):
        try:
            result = unpack(self.actions.recv(zmq.NOBLOCK))
        except zmq.Again:
            return
        now = time.monotonic()
        if (
            result["episode_id"] != self.episode_id
            or result["sequence"] <= self.last_action_sequence
            or result["sequence"] not in self.sent
            or now - self.sent[result["sequence"]] > 0.15
        ):
            return
        if self.pending is not None and result["sequence"] <= self.pending["sequence"]:
            return
        for name, size in (("action", 66), ("token", 64), ("left", 7), ("right", 7)):
            if result[name].shape != (size,):
                raise ValueError(f"Invalid actor {name} shape")
        if (
            np.max(np.abs(result["token"] * 16 - np.round(result["token"] * 16))) > 1e-6
            or np.any(result["token"] < -1)
            or np.any(result["token"] > 15 / 16)
        ):
            raise ValueError("Actor did not send FSQ codes")
        self.pending = result

    def poll_debug(self, expected):
        try:
            raw = self.debug.recv(zmq.NOBLOCK)
        except zmq.Again:
            return
        packet = msgpack.unpackb(raw[len(b"g1_debug") :], raw=False)
        index = packet["index"]
        token = np.asarray(packet["token_state"])
        # Feedback proves the current static scene reached C++ and it consumed
        # the replacement token. Observe ten consecutive policy log samples.
        matching = token.shape == (64,) and np.allclose(token, expected, atol=1e-6)
        matching &= np.allclose(packet["body_q"], self.data.qpos[self.env.body_qpos_adr], atol=2e-3)
        quat = np.asarray(packet["base_quat"])
        matching &= abs(float(quat @ self.data.qpos[3:7])) > 1 - 1e-5
        matching &= np.allclose(packet["body_dq"], 0, atol=2e-3)
        matching &= np.allclose(packet["base_ang_vel"], 0, atol=2e-3)
        for side in ("left", "right"):
            expected_hand = self.data.qpos[getattr(self.env, side + "_hand_qpos_adr")]
            matching &= np.allclose(packet[side + "_hand_q"], expected_hand, atol=2e-3)
        contiguous = self.last_debug_index is None or index == self.last_debug_index + 1
        self.debug_count = (
            self.debug_count + 1 if matching and contiguous else (1 if matching else 0)
        )
        self.last_debug_index = index

    def publish_observation(self):
        self.sequence += 1
        now = time.monotonic()
        self.sent[self.sequence] = now
        self.sent = {key: value for key, value in self.sent.items() if now - value < 0.2}
        self.states.send(
            pack(
                dict(
                    episode_id=self.episode_id,
                    sequence=self.sequence,
                    sim_time=float(self.data.time),
                    observation_time=now,
                    phase=self.phase,
                    release_requested=bool(self.task.release_requested),
                ),
                proprio=self.observations.proprio(self.data),
                task=self.observations.privileged(self.data, self.task),
            ),
            zmq.NOBLOCK,
        )

    def before_step(self):
        now = time.monotonic()
        self.poll_action()
        if self.phase != "running":
            # Read the 50 Hz C++ feedback at the 200 Hz physics cadence so
            # wall-clock jitter cannot make a ten-sample barrier impossible.
            if self.phase == "flushing":
                self.poll_debug(np.zeros(64))
            elif self.phase == "arming":
                self.poll_debug(self.active["token"])
            if now - self.last_wait_tick < 0.02:
                return False
            self.last_wait_tick = now
            if self.phase == "flushing":
                left, right = self.initial_hands()
                self.send_decoder(np.zeros(64), left, right)
                if (
                    now - self.reset_at >= 1
                    and self.debug_count >= 10
                    and self.env.unitree_bridge.command_received_at > self.reset_at
                ):
                    self.phase = "first_action"
                    self.observations.reset()
                    self.pending = None
                    self.sent.clear()
                    self.event("history_flushed", policy_samples=self.debug_count)
            elif self.phase == "first_action":
                left, right = self.initial_hands()
                self.send_decoder(np.zeros(64), left, right)
                if self.pending is not None:
                    self.active = self.pending
                    self.phase = "arming"
                    self.arm_at = now
                    self.debug_count, self.last_debug_index = 0, None
                    self.event("first_action", inference_s=self.active["inference_s"])
            elif self.phase == "arming":
                self.send_decoder(self.active["token"], self.active["left"], self.active["right"])
                if (
                    self.debug_count >= 2
                    and self.env.unitree_bridge.command_received_at > self.arm_at
                ):
                    self.observations.last_action[:] = self.active["action"]
                    self.last_action_sequence = self.active["sequence"]
                    self.pending = None
                    self.phase = "running"
                    self.next_tick = 0.02
                    self.event("running")
            if self.phase == "first_action":
                # Reset history rather than accumulating repeated wait reads.
                self.observations.history = None
                self.publish_observation()
            return False
        if self.data.time + 1e-9 >= 25:
            self.event("episode_end", task_events=self.task.events)
            self.reset()
            return False
        policy_tick = self.data.time + 1e-9 >= self.next_tick
        if policy_tick:
            # mj_step integrates qpos/qvel after computing its derived body
            # poses. Refresh them so joint, frame and contact observations all
            # describe the same current MuJoCo state.
            mujoco.mj_forward(self.model, self.data)
            self.step_count += 1
            forces, table = self.observations.contacts(self.data)
            self.task.update(self.step_count, forces.sum(), table)
            for name, elapsed in self.task.events.items():
                if name not in self.reported_events:
                    self.event(name, task_seconds=elapsed)
                    self.reported_events.add(name)
            # This observation describes the interval just simulated. Its last
            # action must not be replaced by a newly arriving, unapplied action.
            self.publish_observation()
            self.next_tick += 0.02
        accepted = False
        if self.pending is not None:
            # Forward at the next 200 Hz physics tick, without waiting another
            # 20 ms policy period. C++ still consumes at its unchanged 50 Hz.
            if now - self.sent.get(self.pending["sequence"], -float("inf")) <= 0.15:
                self.active = self.pending
                self.last_action_sequence = self.active["sequence"]
                accepted = True
            self.pending = None
        if accepted or policy_tick:
            self.send_decoder(self.active["token"], self.active["left"], self.active["right"])
            self.observations.last_action[:] = self.active["action"]
        if now - self.last_report >= 5:
            self.event(
                "timing",
                action_age_ms=1000 * (now - self.active["observation_time"]),
                phase=self.phase,
                hand_force=float(self.observations.contacts(self.data)[0].sum()),
                robot_position=self.data.xpos[self.observations.pelvis].tolist(),
                object_position=self.data.xpos[self.observations.obj].tolist(),
            )
            self.last_report = now
        return True

    def close(self):
        for stream in (self.states, self.actions, self.decoder, self.debug):
            stream.close(0)
        self.context.term()
        if self.log:
            self.log.close()
