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

# Indices of our 14 actuated joints inside the 16-joint reference (antennas at 9, 10 dropped).
_JOINTS_IN_REF = np.r_[0:9, 11:16]


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
    def joint_pos(frame):
        return frame[0:16][_JOINTS_IN_REF]

    @staticmethod
    def joint_vel(frame):
        return frame[16:32][_JOINTS_IN_REF]

    @staticmethod
    def contacts(frame):
        return frame[32:34] > 0.5

    @staticmethod
    def base_linvel(frame):
        return frame[34:37]

    @staticmethod
    def base_angvel(frame):
        return frame[37:40]
