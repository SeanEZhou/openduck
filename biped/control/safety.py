"""Runtime safety checks shared by simulated deployment rehearsal and hardware."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from biped.config import RobotSpec
from biped.hardware.interface import RobotState


@dataclass
class SafetyLimits:
    max_tilt_rad: float = 1.0          # torso tilt from upright before stopping
    joint_limit_margin: float = 0.05   # rad beyond hard limits counted as a violation
    max_joint_vel: float = 12.0        # rad/s, sensor sanity bound
    max_loop_overrun: float = 0.010    # s beyond the control period


class SafetyMonitor:
    """Returns a reason string when the controller must stop, else None."""

    def __init__(self, spec: RobotSpec, limits: SafetyLimits | None = None):
        self.limits = limits or SafetyLimits()
        self.lower, self.upper = spec.lower, spec.upper
        self.period = spec.control["policy_dt"]

    def check(self, state: RobotState, loop_time: float | None = None) -> str | None:
        lim = self.limits
        values = np.concatenate([state.joint_pos, state.joint_vel, state.gyro, state.gravity])
        if not np.all(np.isfinite(values)):
            return "non-finite sensor reading"
        tilt = np.arccos(np.clip(-state.gravity[2] / max(np.linalg.norm(state.gravity), 1e-9), -1, 1))
        if tilt > lim.max_tilt_rad:
            return f"tilt {np.degrees(tilt):.0f} deg exceeds limit"
        m = lim.joint_limit_margin
        bad = (state.joint_pos < self.lower - m) | (state.joint_pos > self.upper + m)
        if bad.any():
            return f"joint limit violated at indices {np.flatnonzero(bad).tolist()}"
        if np.abs(state.joint_vel).max() > lim.max_joint_vel:
            return "joint velocity out of range"
        if loop_time is not None and loop_time > self.period + lim.max_loop_overrun:
            return f"control loop overrun {loop_time * 1e3:.1f} ms"
        return None
