"""Deployable actor observation, matching the Open Duck Mini v2 runtime layout.

Only quantities available on the real robot are used: servo positions and
velocities, BNO055 gyro/accelerometer, foot contact switches, the operator
command and the controller's own history. Layout for n joints (n = 14):

    gyro(3) accel(3) command(7) q-q_home(n) 0.05*qd(n)
    a[t-1](n) a[t-2](n) a[t-3](n) motor_targets(n) contacts(2) phase(2)

= 101 dims for n = 14, identical to the upstream ONNX policies' input.

`build_observation` and `gait_phase` are array-namespace generic (`xp` is
numpy or jax.numpy) so the CPU env, the MJX env and the hardware runtime
share one implementation.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from biped.config import RobotSpec
from biped.hardware.interface import RobotState

COMMAND_DIM = 7  # vx, vy, wz, neck_pitch, head_pitch, head_yaw, head_roll
JOINT_VEL_SCALE = 0.05


def observation_layout(n: int) -> "OrderedDict[str, slice]":
    sizes = [
        ("gyro", 3),
        ("accelerometer", 3),
        ("command", COMMAND_DIM),
        ("joint_pos", n),
        ("joint_vel", n),
        ("last_action", n),
        ("last_last_action", n),
        ("last_last_last_action", n),
        ("motor_targets", n),
        ("feet_contact", 2),
        ("phase", 2),
    ]
    out, i = OrderedDict(), 0
    for name, k in sizes:
        out[name] = slice(i, i + k)
        i += k
    return out


def build_observation(xp, gyro, accelerometer, command, joint_pos, joint_vel, home,
                      action_history, motor_targets, feet_contact, phase):
    return xp.concatenate(
        [
            gyro,
            accelerometer,
            command,
            joint_pos - home,
            joint_vel * JOINT_VEL_SCALE,
            action_history[0],
            action_history[1],
            action_history[2],
            motor_targets,
            feet_contact.astype(xp.float32),
            phase,
        ]
    ).astype(xp.float32)


def gait_phase(xp, i, started, steps_per_period: int):
    """[cos, sin] of the gait clock; [0, 0] before the first tick (upstream convention)."""
    a = 2 * np.pi * i / steps_per_period
    return xp.where(started, xp.stack([xp.cos(a), xp.sin(a)]), xp.zeros(2))


class ObservationBuilder:
    def __init__(self, spec: RobotSpec):
        self.home = spec.home
        self.n = spec.num_joints
        self.layout = observation_layout(self.n)
        self.dim = list(self.layout.values())[-1].stop

    def __call__(
        self,
        state: RobotState,
        command: np.ndarray,
        action_history: np.ndarray,  # (3, n): a[t-1], a[t-2], a[t-3]
        motor_targets: np.ndarray,
        phase: np.ndarray,
    ) -> np.ndarray:
        return build_observation(np, state.gyro, state.accelerometer, command, state.joint_pos,
                                 state.joint_vel, self.home, action_history, motor_targets,
                                 state.feet_contact, phase)


class GaitPhase:
    """Free-running gait clock, advanced once per control step."""

    def __init__(self, steps_per_period: int):
        self.steps_per_period = steps_per_period
        self.i = 0
        self.started = False

    def reset(self) -> None:
        self.i = 0
        self.started = False

    def advance(self) -> None:
        self.i = (self.i + 1) % self.steps_per_period
        self.started = True

    @property
    def value(self) -> np.ndarray:
        return gait_phase(np, self.i, self.started, self.steps_per_period)
