"""Per-episode domain randomization of the MuJoCo model, driven by configs/randomization.yaml."""

from __future__ import annotations

from typing import Any

import numpy as np

from biped.hardware.mujoco import MujocoRobot


class DomainRandomizer:
    """Samples physical parameters around the nominal model at each reset.

    Also samples a per-joint calibration offset (servo zero error), applied by
    the env as: commanded = target + offset, measured = q - offset.
    """

    def __init__(self, robot: MujocoRobot, cfg: dict[str, Any]):
        self.robot = robot
        self.cfg = cfg or {}
        self.enabled = bool(self.cfg.get("enable", False))
        m = robot.model
        self._nominal = {
            "geom_friction": m.geom_friction.copy(),
            "body_mass": m.body_mass.copy(),
            "body_ipos": m.body_ipos.copy(),
            "dof_damping": m.dof_damping.copy(),
            "dof_frictionloss": m.dof_frictionloss.copy(),
            "dof_armature": m.dof_armature.copy(),
            "actuator_gainprm": m.actuator_gainprm.copy(),
            "actuator_biasprm": m.actuator_biasprm.copy(),
            "actuator_forcerange": m.actuator_forcerange.copy(),
        }
        self.joint_offset = np.zeros(robot.spec.num_joints)
        self.sample: dict[str, Any] = {}

    def _u(self, rng: np.random.Generator, key: str, size=None):
        lo, hi = self.cfg[key]
        return rng.uniform(lo, hi, size)

    def restore(self) -> None:
        m = self.robot.model
        for k, v in self._nominal.items():
            getattr(m, k)[:] = v
        self.joint_offset[:] = 0.0
        self.sample = {}

    def randomize(self, rng: np.random.Generator) -> None:
        self.restore()
        if not self.enabled:
            return
        r, m, c = self.robot, self.robot.model, self.cfg
        dofs, acts, n = r.qvel_adr, r.actuator_ids, r.spec.num_joints
        s = self.sample

        if "floor_friction" in c:
            s["floor_friction"] = self._u(rng, "floor_friction")
            m.geom_friction[r.floor_geom, 0] = s["floor_friction"]
        if "body_mass_scale" in c:
            m.body_mass[:] = self._nominal["body_mass"] * self._u(rng, "body_mass_scale", m.nbody)
        if "trunk_added_mass" in c:
            s["trunk_added_mass"] = self._u(rng, "trunk_added_mass")
            m.body_mass[r.root_body] += s["trunk_added_mass"]
        if "trunk_com_offset" in c:
            m.body_ipos[r.root_body] += self._u(rng, "trunk_com_offset", 3)
        for key, field in (
            ("joint_damping_scale", "dof_damping"),
            ("joint_frictionloss_scale", "dof_frictionloss"),
            ("joint_armature_scale", "dof_armature"),
        ):
            if key in c:
                getattr(m, field)[dofs] = self._nominal[field][dofs] * self._u(rng, key, n)
        if "kp_scale" in c:
            kp = self._nominal["actuator_gainprm"][acts, 0] * self._u(rng, "kp_scale", n)
            m.actuator_gainprm[acts, 0] = kp
            m.actuator_biasprm[acts, 1] = -kp
        if "torque_limit_scale" in c:
            m.actuator_forcerange[acts] = self._nominal["actuator_forcerange"][acts] * self._u(
                rng, "torque_limit_scale", n
            )[:, None]
        if "joint_offset" in c:
            self.joint_offset[:] = self._u(rng, "joint_offset", n)
