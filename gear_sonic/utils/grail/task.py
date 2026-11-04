"""Offline carry clock, ported from GRAIL reference_free_task (no success test)."""

import math

import numpy as np

from .plan import PLAN_FPS


class CarryTask:
    def __init__(self, plan, settle_s=1.0, dt=0.02, threshold=0.1, stable_steps=3):
        self.plan = plan
        self.dt = dt
        self.settle_steps = math.ceil(settle_s / dt - 1e-9)
        self.threshold, self.stable_steps = threshold, stable_steps
        self.steps, self.start_step = 0, -1
        self.pickup_count = self.release_count = self.lost_contact_count = 0
        self.picked_up = self.released = self.dropped = self.carrying = False
        self.events = {}

    @property
    def playing(self):
        return self.start_step >= 0 and self.steps >= self.start_step

    def read(self):
        frame = min(
            max(0, math.floor((self.steps - self.start_step) * self.dt * PLAN_FPS + 1e-6)),
            self.plan.release_start_frame,
        )
        sparse = (
            self.plan.sparse[frame].copy() if self.playing and not self.dropped else np.zeros(2)
        )
        drop = float(self.plan.drop[frame, 0]) if self.playing else 0.0
        return sparse, drop

    def command(self):
        sparse, drop = self.read()
        return sparse if self.carrying and drop < 0.5 else np.zeros(2), drop

    def update(self, steps, hand_force, table_force):
        if steps == self.steps:
            return
        _, drop = self.read()  # Previously observed request, before advancing clock.
        contact = hand_force >= self.threshold
        self.pickup_count = (
            self.pickup_count + 1 if contact and table_force < self.threshold and drop < 0.5 else 0
        )
        self.picked_up |= self.pickup_count >= self.stable_steps
        self.release_count = (
            self.release_count + 1 if self.picked_up and drop >= 0.5 and not contact else 0
        )
        self.released |= self.release_count >= self.stable_steps
        self.lost_contact_count = (
            self.lost_contact_count + 1 if self.picked_up and drop < 0.5 and not contact else 0
        )
        self.dropped |= self.lost_contact_count >= self.stable_steps
        self.carrying = self.picked_up and contact and drop < 0.5 and not self.released
        if self.picked_up and self.start_step < 0:
            self.start_step = steps + self.settle_steps
        self.steps = steps
        for name, happened in dict(
            picked_up=self.picked_up,
            playback=self.playing,
            drop_requested=self.read()[1] >= 0.5,
            released=self.released,
            dropped=self.dropped,
        ).items():
            if happened and name not in self.events:
                self.events[name] = steps * self.dt
