"""Policy action -> physical joint position targets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from biped.config import RobotSpec


@dataclass(frozen=True)
class ActionParams:
    home: np.ndarray
    scale: float
    max_step: float          # rad per control step (rate limit)
    lower: np.ndarray        # soft limits
    upper: np.ndarray
    head_map: np.ndarray     # (n, 4): joint offsets = head_map @ command[3:7]
    rate_limit: bool = True
    clip_to_soft_limits: bool = True
    head_command_offset: bool = True

    @classmethod
    def from_spec(cls, spec: RobotSpec, rate_limit=True, clip_to_soft_limits=True, head_command_offset=True):
        lower, upper = spec.soft_limits()
        head_map = np.zeros((spec.num_joints, 4))
        for k, i in enumerate(spec.indices("head")):
            head_map[i, k] = 1.0
        return cls(spec.home, float(spec.control["action_scale"]),
                   float(spec.actuator["max_velocity"] * spec.control["policy_dt"]),
                   lower, upper, head_map, rate_limit, clip_to_soft_limits, head_command_offset)


def process_action(xp, p: ActionParams, action, command, prev_targets):
    """target = home + scale * action (+ head command offsets), rate-limited and clipped.

    Array-namespace generic (numpy / jax.numpy). Head offsets match the
    upstream runtime so the policy sees identical behaviour in sim and on hardware.
    """
    action = xp.clip(action, -1.0 / p.scale, 1.0 / p.scale)
    targets = p.home + p.scale * action
    if p.head_command_offset:
        targets = targets + p.head_map @ command[3:7]
    if p.rate_limit:
        targets = xp.clip(targets, prev_targets - p.max_step, prev_targets + p.max_step)
    if p.clip_to_soft_limits:
        targets = xp.clip(targets, p.lower, p.upper)
    return targets


class ActionProcessor:
    """Stateful numpy wrapper around `process_action` (keeps the previous targets)."""

    def __init__(self, spec: RobotSpec, rate_limit: bool = True, clip_to_soft_limits: bool = True,
                 head_command_offset: bool = True):
        self.params = ActionParams.from_spec(spec, rate_limit, clip_to_soft_limits, head_command_offset)
        self.home = spec.home
        self.prev_targets = self.home.copy()

    def reset(self, targets: np.ndarray | None = None) -> None:
        self.prev_targets = (self.home if targets is None else targets).copy()

    def __call__(self, action: np.ndarray, command: np.ndarray | None = None) -> np.ndarray:
        command = np.zeros(7) if command is None else command
        targets = process_action(np, self.params, np.asarray(action, dtype=np.float64), command, self.prev_targets)
        self.prev_targets = targets.copy()
        return targets
