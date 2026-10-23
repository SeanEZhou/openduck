"""Reward terms. Each returns an unscaled scalar; weights live in configs/train_*.yaml.

Positive weights are rewards, negative weights are costs. Every term is
logged separately so curricula can be debugged.

All terms are branch-free and array-namespace generic (`c.xp` is numpy for
the CPU env, jax.numpy for the MJX env), so both simulators optimize exactly
the same objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from biped.control.reference_motion import ReferenceMotion


@dataclass
class RewardContext:
    command: Any               # (7,)
    joint_pos: Any             # true q
    joint_vel: Any
    home: Any
    leg_idx: Any
    action: Any
    last_action: Any
    torques: Any
    gravity: Any               # torso-frame gravity direction (true)
    local_linvel: Any          # IMU-frame
    gyro: Any                  # IMU-frame angular velocity (true)
    base_height: Any
    home_height: float
    base_qvel: Any             # free-joint qvel (world lin, local ang)
    feet_contact: Any
    feet_linvel: Any           # (2, 3) world
    feet_air_time: Any         # (2,) before reset on touchdown
    first_contact: Any         # (2,) bool
    soft_lower: Any
    soft_upper: Any
    ref_frame: Any | None
    terminated: Any
    xp: Any = np
    head_idx: Any = None       # indices of head joints
    head_target: Any = None    # commanded head joint positions (home + command[3:7])
    recovering: Any = 0.0      # 1.0 within the post-push recovery window, else 0.0
    ref_joint_idx: Any = None  # robot joint order -> reference-frame joint index (default: full robot)
    heading_error: Any = 0.0   # wrapped (yaw - integrated commanded yaw), rad
    ref_offset: Any = 0.0      # this robot's home minus the reference gaits' home (re-postured variants)

    @property
    def moving(self):
        """1.0 when a non-zero velocity command is active, else 0.0."""
        xp = self.xp
        return (xp.linalg.norm(self.command[:3]) > 0.01).astype(xp.float32)


def alive(c: RewardContext, **_):
    return c.xp.ones(())


def termination(c: RewardContext, **_):
    return c.xp.asarray(c.terminated, dtype=c.xp.float32)


def tracking_lin_vel(c: RewardContext, sigma: float = 0.01, y_tol: float = 0.1, **_):
    xp = c.xp
    ex = (c.command[0] - c.local_linvel[0]) ** 2
    ey = xp.maximum(xp.abs(c.local_linvel[1] - c.command[1]) - y_tol, 0.0) ** 2
    return xp.exp(-(ex + ey) / sigma)


def tracking_ang_vel(c: RewardContext, sigma: float = 0.01, **_):
    return c.xp.exp(-((c.command[2] - c.gyro[2]) ** 2) / sigma)


def push_recovery(c: RewardContext, sigma: float = 0.25, **_):
    """After a push, reward shedding the excess velocity. Unlike tracking_lin_vel (sigma 0.01,
    flat at zero for errors above ~0.2 m/s), the wide kernel still has a gradient right after a
    0.5-1 m/s kick. Zero outside the post-push recovery window."""
    xp = c.xp
    return xp.exp(-xp.sum((c.local_linvel[:2] - c.command[:2]) ** 2) / sigma) * c.recovering


def heading(c: RewardContext, **_):
    """Accumulated heading error vs. the integral of the yaw-rate command. Unlike yaw-rate
    tracking, a small constant turning bias grows into a large cost."""
    return c.heading_error**2


def orientation(c: RewardContext, **_):
    return c.xp.sum(c.gravity[:2] ** 2)


def base_height(c: RewardContext, **_):
    return (c.base_height - c.home_height) ** 2


def lin_vel_z(c: RewardContext, **_):
    return c.base_qvel[2] ** 2


def ang_vel_xy(c: RewardContext, **_):
    return c.xp.sum(c.gyro[:2] ** 2)


def base_lin_vel_xy(c: RewardContext, **_):
    return c.xp.sum(c.base_qvel[:2] ** 2)


def pose(c: RewardContext, **_):
    i = c.leg_idx
    return c.xp.sum(c.xp.abs(c.joint_pos[i] - c.home[i]))


def stand_still(c: RewardContext, **_):
    xp = c.xp
    return (xp.sum(xp.abs(c.joint_pos - c.home)) + xp.sum(xp.abs(c.joint_vel))) * (1.0 - c.moving)


def torques(c: RewardContext, **_):
    return c.xp.sum(c.torques**2)


def action_rate(c: RewardContext, **_):
    return c.xp.sum((c.action - c.last_action) ** 2)


def joint_vel(c: RewardContext, **_):
    return c.xp.sum(c.joint_vel**2)


def joint_limits(c: RewardContext, **_):
    xp = c.xp
    below = xp.maximum(c.soft_lower - c.joint_pos, 0.0)
    above = xp.maximum(c.joint_pos - c.soft_upper, 0.0)
    return xp.sum(below + above)


def feet_slip(c: RewardContext, **_):
    xp = c.xp
    v = xp.sum(c.feet_linvel[:, :2] ** 2, axis=1)
    return xp.sum(v * c.feet_contact)


def feet_air_time(c: RewardContext, target: float = 0.2, **_):
    return c.xp.sum((c.feet_air_time - target) * c.first_contact) * c.moving


def double_support(c: RewardContext, **_):
    """Reward keeping both feet on the ground (standing task)."""
    return c.xp.all(c.feet_contact).astype(c.xp.float32)


def head_tracking(c: RewardContext, **_):
    """Head joints should follow the commanded head pose (not be used as a balance mass)."""
    return c.xp.sum((c.joint_pos[c.head_idx] - c.head_target) ** 2)


def imitation(c: RewardContext, **_):
    """Open Duck / Disney BDX style imitation of polynomial reference gaits (legs only)."""
    xp = c.xp
    if c.ref_frame is None:
        return xp.zeros(())
    f, i = c.ref_frame, c.leg_idx
    ri = {} if c.ref_joint_idx is None else {"idx": c.ref_joint_idx}
    ref_q = (ReferenceMotion.joint_pos(f, **ri) + c.ref_offset)[i]
    ref_qd = ReferenceMotion.joint_vel(f, **ri)[i]
    lin, ang = ReferenceMotion.base_linvel(f), ReferenceMotion.base_angvel(f)
    v, w = c.base_qvel[:3], c.base_qvel[3:]
    r = (
        xp.exp(-8.0 * xp.sum((v[:2] - lin[:2]) ** 2))
        + xp.exp(-8.0 * (v[2] - lin[2]) ** 2)
        + 0.5 * xp.exp(-2.0 * xp.sum((w[:2] - ang[:2]) ** 2))
        + 0.5 * xp.exp(-2.0 * (w[2] - ang[2]) ** 2)
        - 15.0 * xp.sum((c.joint_pos[i] - ref_q) ** 2)
        - 1e-3 * xp.sum((c.joint_vel[i] - ref_qd) ** 2)
        + xp.sum(c.feet_contact == ReferenceMotion.contacts(f))
    )
    return r * c.moving


REWARD_FUNCTIONS = {
    f.__name__: f
    for f in (
        alive, termination, tracking_lin_vel, tracking_ang_vel, push_recovery, heading, orientation, base_height,
        lin_vel_z, ang_vel_xy, base_lin_vel_xy, pose, stand_still, torques, action_rate,
        joint_vel, joint_limits, feet_slip, feet_air_time, double_support, head_tracking, imitation,
    )
}


def compute_rewards(c: RewardContext, scales: dict[str, float], params: dict,
                    suspend_after_push: tuple[str, ...] = ()) -> dict:
    """Weighted terms. Terms in `suspend_after_push` are zeroed during the post-push recovery
    window, so e.g. stand-still / pose costs don't forbid a recovery step."""
    out = {}
    for name, w in scales.items():
        if w == 0:
            continue
        v = w * REWARD_FUNCTIONS[name](c, **params.get(name, {}))
        out[name] = v * (1.0 - c.recovering) if name in suspend_after_push else v
    return out
