"""GRAIL reference-free carry clock and release requests."""

import math

import numpy as np

from .plan import PLAN_FPS

RELEASE_ENTER_DISTANCE_M = 0.20
RELEASE_EXIT_DISTANCE_M = 0.30
RELEASE_DWELL_S = 0.5
CONTACT_STABLE_STEPS = 3


class CarryTask:
    def __init__(self, plan, settle_s=2.0, dt=0.02, threshold=0.1, stable_steps=CONTACT_STABLE_STEPS,
                 *, release_by_distance=True):
        self.plan = plan
        self.dt = dt
        self.settle_steps = math.ceil(settle_s / dt - 1e-9)
        self.threshold, self.stable_steps = threshold, stable_steps
        self.release_by_distance = release_by_distance
        self.release_dwell_steps = math.ceil(RELEASE_DWELL_S / dt - 1e-9)
        self._release_enter_step = None
        self._release_requested = False
        self.steps, self.start_step = 0, -1
        self.pickup_count = self.release_count = self.lost_contact_count = 0
        self.picked_up = self.released = self.dropped = self.carrying = False
        self.events = {}

    @property
    def playing(self):
        return self.start_step >= 0 and self.steps >= self.start_step

    @property
    def frame(self):
        return min(
            max(0, math.floor((self.steps - self.start_step) * self.dt * PLAN_FPS + 1e-6)),
            self.plan.hold_end_frame,
        )

    @property
    def release_requested(self):
        if self.release_by_distance:
            return self._release_requested
        # The GR00T task log and older rollouts use the end of the carry plan.
        return self.playing and self.frame >= self.plan.hold_end_frame

    def _update_release_request(self, object_position):
        distance = np.linalg.norm(object_position - self.plan.final_position)
        if not (distance <= RELEASE_EXIT_DISTANCE_M):
            self._release_enter_step = None
            return
        if self._release_enter_step is None and distance < RELEASE_ENTER_DISTANCE_M:
            self._release_enter_step = self.steps
        if self._release_enter_step is not None:
            self._release_requested = self.steps - self._release_enter_step >= self.release_dwell_steps

    def read(self, base_position):
        sparse = self.plan.sparse[self.frame].copy()
        sparse[0] = min(np.linalg.norm(self.plan.base_goal_xy - np.asarray(base_position)[:2]), 0.15)
        return sparse if self.playing and not self.dropped else np.zeros(2, dtype=np.float32)

    def command(self, base_position):
        return self.read(base_position) if self.carrying else np.zeros(2, dtype=np.float32)

    def update(self, steps, hand_force, table_force, object_position):
        if steps == self.steps:
            return
        request = self.release_requested
        contact = hand_force >= self.threshold
        self.pickup_count = (
            self.pickup_count + 1 if contact and table_force < self.threshold else 0
        )
        self.picked_up |= self.pickup_count >= self.stable_steps
        self.release_count = (
            self.release_count + 1 if self.picked_up and request and not contact else 0
        )
        self.released |= self.release_count >= self.stable_steps
        self.lost_contact_count = (
            self.lost_contact_count + 1 if self.picked_up and not contact else 0
        )
        self.dropped |= self.lost_contact_count >= self.stable_steps
        self.carrying = self.picked_up and contact
        if self.picked_up and self.start_step < 0:
            self.start_step = steps + self.settle_steps
        self.steps = steps
        if self.release_by_distance and self.picked_up and not request:
            self._update_release_request(object_position)
        for name, happened in dict(
            picked_up=self.picked_up,
            playback=self.playing,
            drop_requested=self.release_requested,
            released=self.released,
            dropped=self.dropped,
        ).items():
            if happened and name not in self.events:
                self.events[name] = steps * self.dt
