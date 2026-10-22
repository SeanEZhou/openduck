"""Polynomial gait reference motions (converted from upstream; see scripts/import_reference_motion.py).

Each frame is 40-D, upstream layout:
    [0:16]  joint positions (5 left leg, 4 head, 2 antennas, 5 right leg)
    [16:32] joint velocities (same order)
    [32:34] foot contacts [left, right]
    [34:37] base linear velocity, [37:40] base angular velocity
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Joint order of the 16-joint upstream reference frames.
REF_JOINTS = (
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll", "left_antenna", "right_antenna",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
)
# Indices of the full robot's 14 actuated joints (antennas dropped).
_JOINTS_IN_REF = np.r_[0:9, 11:16]


def ref_indices(joint_names) -> np.ndarray:
    """Indices into a reference frame's joint block for the given robot joint order."""
    return np.array([REF_JOINTS.index(n) for n in joint_names])


class ReferenceMotion:
    def __init__(self, path: str | Path):
        z = np.load(path)
        self.coeffs = z["coeffs"]
        self.dxs, self.dys, self.dthetas = z["dxs"], z["dys"], z["dthetas"]
        self.period = float(z["period"])
        self.fps = float(z["fps"])
        self.steps_per_period = int(self.period * self.fps)

    def _arrays(self, xp):
        # Not cached: under jit, xp.asarray of a numpy constant is free and caching would leak tracers.
        return tuple(xp.asarray(a) for a in (self.coeffs, self.dxs, self.dys, self.dthetas))

    def frame(self, cmd, i, xp=np):
        """Reference frame for velocity command cmd[:3] at phase step i (nearest grid point)."""
        coeffs, dxs, dys, dths = self._arrays(xp)
        ix = xp.argmin(xp.abs(dxs - cmd[0]))
        iy = xp.argmin(xp.abs(dys - cmd[1]))
        it = xp.argmin(xp.abs(dths - cmd[2]))
        t = (i % self.steps_per_period) / self.steps_per_period
        c = coeffs[ix, iy, it]  # (dim, order), lowest order first
        return c @ (t ** xp.arange(c.shape[1]))

    @staticmethod
    def joint_pos(frame, idx=_JOINTS_IN_REF):
        return frame[0:16][idx]

    @staticmethod
    def joint_vel(frame, idx=_JOINTS_IN_REF):
        return frame[16:32][idx]

    @staticmethod
    def contacts(frame):
        return frame[32:34] > 0.5

    @staticmethod
    def base_linvel(frame):
        return frame[34:37]

    @staticmethod
    def base_angvel(frame):
        return frame[37:40]
