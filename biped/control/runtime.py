"""Controller-side state machine shared by training, evaluation and hardware deployment.

One control tick, identical in simulation and on the robot:

    obs = runtime.observe(state, command)   # uses a[t-1..t-3], last targets, phase
    action = policy(obs)
    targets = runtime.act(action, command)  # updates history
    robot.write_targets(targets); robot.step()
"""

from __future__ import annotations

import numpy as np

from biped.config import RobotSpec
from biped.control.actions import ActionProcessor
from biped.control.observations import GaitPhase, ObservationBuilder
from biped.hardware.interface import RobotState


class PolicyRuntime:
    def __init__(self, spec: RobotSpec, steps_per_period: int, **action_kwargs):
        self.spec = spec
        self.obs_builder = ObservationBuilder(spec)
        self.actions = ActionProcessor(spec, **action_kwargs)
        self.phase = GaitPhase(steps_per_period)
        n = spec.num_joints
        self.action_history = np.zeros((3, n))
        self.motor_targets = spec.home.copy()

    @property
    def obs_dim(self) -> int:
        return self.obs_builder.dim

    def reset(self) -> None:
        self.action_history[:] = 0.0
        self.motor_targets = self.spec.home.copy()
        self.actions.reset(self.motor_targets)
        self.phase.reset()

    def observe(self, state: RobotState, command: np.ndarray) -> np.ndarray:
        obs = self.obs_builder(state, command, self.action_history, self.motor_targets, self.phase.value)
        self.phase.advance()
        return obs

    def act(self, action: np.ndarray, command: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        self.action_history = np.roll(self.action_history, 1, axis=0)
        self.action_history[0] = action
        self.motor_targets = self.actions(action, command)
        return self.motor_targets
