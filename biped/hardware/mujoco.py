"""MuJoCo backend for RobotInterface, plus privileged (sim-only) accessors.

Privileged accessors (`base_*`, `feet_pos`, ...) are for rewards, termination
and evaluation only. They must never feed the deployed actor's observation.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from biped.config import RobotSpec
from biped.hardware.interface import RobotInterface, RobotState


class MujocoRobot(RobotInterface):
    def __init__(self, spec: RobotSpec, scene: str | Path | None = None):
        self.spec = spec
        self.model = mujoco.MjModel.from_xml_path(str(scene or spec.model_path("scene")))
        self.data = mujoco.MjData(self.model)
        m = self.model

        sim_dt = spec.control["sim_dt"]
        if not np.isclose(m.opt.timestep, sim_dt):
            raise ValueError(f"MJCF timestep {m.opt.timestep} != robot.yaml sim_dt {sim_dt}")
        self.n_substeps = int(round(spec.control["policy_dt"] / sim_dt))

        names = spec.joint_names
        self.actuator_ids = np.array([m.actuator(n).id for n in names])
        joint_ids = [m.actuator_trnid[a, 0] for a in self.actuator_ids]
        for n, jid in zip(names, joint_ids):
            if m.joint(jid).name != n:
                raise ValueError(f"actuator {n} drives joint {m.joint(jid).name}")
        self.qpos_adr = np.array([m.jnt_qposadr[j] for j in joint_ids])
        self.qvel_adr = np.array([m.jnt_dofadr[j] for j in joint_ids])

        free = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        self.base_qpos_adr = m.jnt_qposadr[free[0]]
        self.base_qvel_adr = m.jnt_dofadr[free[0]]

        md = spec.model
        self.root_body = m.body(md["root_body"]).id
        self.imu_site = m.site(md["imu_site"]).id
        self.foot_sites = np.array([m.site(s).id for s in md["foot_sites"]])
        self.foot_geoms = np.array([m.geom(g).id for g in md["foot_geoms"]])
        self.floor_geom = m.geom(md["floor_geom"]).id
        self.home_key = m.keyframe(md["home_keyframe"]).id

        self._sensor = {m.sensor(i).name: (m.sensor_adr[i], m.sensor_dim[i]) for i in range(m.nsensor)}
        self._contact_cache = None

    # ------------------------------------------------------------------ reset
    def reset_to_home(self) -> None:
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key)
        self.forward()

    def settled_home_qpos(self, seconds: float = 1.0) -> np.ndarray:
        """qpos after holding the home pose on the floor (the keyframe starts ~1 cm in the ground)."""
        self.reset_to_home()
        for _ in range(int(seconds / self.spec.control["policy_dt"])):
            self.step()
        q = self.data.qpos.copy()
        self.reset_to_home()
        return q

    def min_foot_site_z(self, qpos: np.ndarray) -> float:
        """Height of the lower foot site for a given qpos (kinematics only; data.qpos is restored)."""
        saved = self.data.qpos.copy()
        self.data.qpos[:] = qpos
        mujoco.mj_kinematics(self.model, self.data)
        z = float(self.data.site_xpos[self.foot_sites, 2].min())
        self.data.qpos[:] = saved
        mujoco.mj_kinematics(self.model, self.data)
        return z

    def set_joint_pos(self, q: np.ndarray) -> None:
        self.data.qpos[self.qpos_adr] = q

    def set_base_pose(self, pos: np.ndarray | None = None, quat: np.ndarray | None = None) -> None:
        a = self.base_qpos_adr
        if pos is not None:
            self.data.qpos[a : a + 3] = pos
        if quat is not None:
            self.data.qpos[a + 3 : a + 7] = quat

    def set_base_vel(self, qvel6: np.ndarray) -> None:
        a = self.base_qvel_adr
        self.data.qvel[a : a + 6] = qvel6

    def forward(self) -> None:
        mujoco.mj_forward(self.model, self.data)
        self._contact_cache = None

    # ------------------------------------------------------- RobotInterface
    def read_state(self) -> RobotState:
        return RobotState(
            joint_pos=self.data.qpos[self.qpos_adr].copy(),
            joint_vel=self.data.qvel[self.qvel_adr].copy(),
            gyro=self.sensor("gyro").copy(),
            accelerometer=self.sensor("accelerometer").copy(),
            gravity=self.gravity_in_imu(),
            feet_contact=self.feet_contact(),
            timestamp=self.data.time,
        )

    def write_targets(self, joint_targets: np.ndarray) -> None:
        self.data.ctrl[self.actuator_ids] = joint_targets

    def step(self) -> None:
        mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
        self._contact_cache = None

    # ------------------------------------------------------------- sensors
    def sensor(self, name: str) -> np.ndarray:
        adr, dim = self._sensor[name]
        return self.data.sensordata[adr : adr + dim]

    def gravity_in_imu(self) -> np.ndarray:
        return self.data.site_xmat[self.imu_site].reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])

    def feet_contact(self) -> np.ndarray:
        if self._contact_cache is None:
            c = np.zeros(2, dtype=bool)
            n = self.data.ncon
            if n:
                g = self.data.contact.geom[:n]
                other = np.where(g[:, 0] == self.floor_geom, g[:, 1], np.where(g[:, 1] == self.floor_geom, g[:, 0], -1))
                c[:] = np.isin(self.foot_geoms, other)
            self._contact_cache = c
        return self._contact_cache.copy()

    # --------------------------------------------------------- privileged
    @property
    def base_pos(self) -> np.ndarray:
        a = self.base_qpos_adr
        return self.data.qpos[a : a + 3]

    @property
    def base_quat(self) -> np.ndarray:
        a = self.base_qpos_adr
        return self.data.qpos[a + 3 : a + 7]

    @property
    def base_qvel(self) -> np.ndarray:
        a = self.base_qvel_adr
        return self.data.qvel[a : a + 6]

    def local_linvel(self) -> np.ndarray:
        """Linear velocity of the IMU site expressed in the IMU frame."""
        return self.sensor("local_linvel")

    def global_linvel(self) -> np.ndarray:
        return self.sensor("global_linvel")

    def global_angvel(self) -> np.ndarray:
        return self.sensor("global_angvel")

    def feet_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.foot_sites]

    def feet_linvel(self) -> np.ndarray:
        return np.stack([self.sensor("left_foot_global_linvel"), self.sensor("right_foot_global_linvel")])

    def actuator_force(self) -> np.ndarray:
        return self.data.actuator_force[self.actuator_ids]

    def has_nan(self) -> bool:
        return bool(np.isnan(self.data.qpos).any() or np.isnan(self.data.qvel).any())
