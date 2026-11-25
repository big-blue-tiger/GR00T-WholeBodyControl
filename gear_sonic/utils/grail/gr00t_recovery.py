"""Detect the first suggested handover from the TD Advantage of a rollout."""

from collections import deque
from dataclasses import dataclass
import math

from .offline_data import SUCCESS_DISTANCE_M


@dataclass(frozen=True)
class RecoveryConfig:
    td_threshold: float = -0.35
    window_steps: int = 50
    confirmation_steps: int = 5

    def __post_init__(self):
        if not math.isfinite(self.td_threshold) or self.td_threshold >= 0:
            raise ValueError("recovery TD threshold must be finite and negative")
        if self.window_steps < 2 or self.confirmation_steps < 1:
            raise ValueError("invalid recovery TD window or confirmation length")


class RecoveryGate:
    def __init__(self, config):
        self.config = config
        self.window = deque(maxlen=config.window_steps)
        self.below_threshold_steps = 0
        self.release_started = False

    def update(self, td_advantage, *, release_requested, target_distance_m):
        cfg = self.config
        # The GR00T policy can reach and release at the bin ahead of the GRAIL clock.
        self.release_started |= bool(release_requested or target_distance_m < SUCCESS_DISTANCE_M)
        if self.release_started:
            return None
        if not math.isfinite(td_advantage):
            self.window.clear()
            self.below_threshold_steps = 0
            return None
        self.window.append(td_advantage)
        if len(self.window) < cfg.window_steps:
            return None
        mean = math.fsum(self.window) / cfg.window_steps
        self.below_threshold_steps = self.below_threshold_steps + 1 if mean <= cfg.td_threshold else 0
        if self.below_threshold_steps >= cfg.confirmation_steps:
            return dict(reason="td_advantage", mean_td_advantage=mean)
        return None
