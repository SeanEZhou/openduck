"""Robot I/O abstraction shared by simulation and hardware backends.

Only quantities measurable on the real robot appear in `RobotState`. The
policy runtime (biped.control) talks exclusively to this interface, so the
exact same observation/action code runs in MuJoCo and on the Open Duck.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class RobotState:
    joint_pos: np.ndarray      # (n,) rad, in config joint order
    joint_vel: np.ndarray      # (n,) rad/s
    gyro: np.ndarray           # (3,) rad/s, IMU frame
    accelerometer: np.ndarray  # (3,) m/s^2, IMU frame (specific force)
    gravity: np.ndarray        # (3,) unit gravity direction in IMU frame (from IMU fusion)
    feet_contact: np.ndarray   # (2,) bool, [left, right]
    timestamp: float           # s


class RobotInterface(ABC):
    """Position-controlled robot: read measured state, write joint targets."""

    @abstractmethod
    def read_state(self) -> RobotState: ...

    @abstractmethod
    def write_targets(self, joint_targets: np.ndarray) -> None:
        """Command joint position targets (rad, config joint order)."""

    @abstractmethod
    def step(self) -> None:
        """Advance one control period (sim: integrate physics; hw: wait for the tick)."""

    def enable(self) -> None:
        """Enable torque."""

    def disable(self) -> None:
        """Disable torque / go limp. Must be safe to call at any time."""
