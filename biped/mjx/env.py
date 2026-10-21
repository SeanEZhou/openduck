"""MJX (GPU, JAX) twin of biped.envs.locomotion.DuckEnv.

Same YAML configs, same observation/action/reward code (the array-generic
functions in biped.control and biped.envs.rewards), same noise, latency, push
and termination semantics. Policies trained here are validated in the CPU
MuJoCo env and the deployment loop (sim-to-sim check), which use the same
contract code with numpy.

Differences from the CPU env, by design:
- Model-level domain randomization is sampled once per parallel env (vmapped
  model, as in MuJoCo Playground); joint calibration offsets are re-sampled
  every episode.
- The critic gets a privileged observation ("privileged_state"); the actor
  only gets "state" (the deployable 101-D observation).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from biped.config import RobotSpec, load_robot, load_yaml
from biped.control.actions import ActionParams, process_action
from biped.control.observations import build_observation, gait_phase
from biped.control.reference_motion import ReferenceMotion
from biped.envs.locomotion import HEAD_COMMANDS, per_joint_noise
from biped.envs.rewards import RewardContext, compute_rewards
from biped.hardware.mujoco import MujocoRobot


def _rng(r) -> tuple[float, float]:
    return (float(r[0]), float(r[1])) if r is not None else (0.0, 0.0)


class DuckMjxEnv(mjx_env.MjxEnv):
    def __init__(self, env_cfg: dict[str, Any], spec: RobotSpec | None = None):
        spec = spec or load_robot()
        super().__init__(config_dict.create(ctrl_dt=spec.control["policy_dt"], sim_dt=spec.control["sim_dt"]))
        self.cfg, self.spec = env_cfg, spec
        cpu = MujocoRobot(spec)
        self._mj_model = cpu.model
        self._mjx_model = mjx.put_model(cpu.model)
        self._xml_path = str(spec.model_path("scene"))
        self.n = spec.num_joints

        settled = cpu.settled_home_qpos()
        self._init_qpos = jp.asarray(settled)
        self.home_height = float(settled[cpu.base_qpos_adr + 2])
        self.foot_z_ref = cpu.min_foot_site_z(settled)
        self.foot_sites = np.asarray(cpu.foot_sites)
        self.qpos_adr, self.qvel_adr = np.asarray(cpu.qpos_adr), np.asarray(cpu.qvel_adr)
        self.act_ids = np.asarray(cpu.actuator_ids)
        self.bq, self.bv = int(cpu.base_qpos_adr), int(cpu.base_qvel_adr)
        self.imu_site, self.root_body = int(cpu.imu_site), int(cpu.root_body)
        self.foot_geoms, self.floor_geom = [int(g) for g in cpu.foot_geoms], int(cpu.floor_geom)
        self._sensor = {k: (int(a), int(d)) for k, (a, d) in cpu._sensor.items()}

        self.ref = ReferenceMotion(spec.model_path("reference_motion"))
        self.P = self.ref.steps_per_period
        ap = ActionParams.from_spec(spec, **env_cfg.get("action", {}))
        self.ap = dataclasses.replace(ap, home=jp.asarray(ap.home), lower=jp.asarray(ap.lower),
                                      upper=jp.asarray(ap.upper), head_map=jp.asarray(ap.head_map))
        self.home = jp.asarray(spec.home)
        self.hard_lower, self.hard_upper = jp.asarray(spec.lower), jp.asarray(spec.upper)
        self.soft_lower, self.soft_upper = (jp.asarray(x) for x in spec.soft_limits())
        self.leg_idx = np.asarray(spec.leg_indices)

        self.noise = env_cfg.get("noise", {})
        self.noise_level = float(self.noise.get("level", 0.0))
        self.joint_noise = jp.asarray(per_joint_noise(spec, self.noise.get("joint_pos", {})))
        self.action_delay = self._delay_range("action_delay")
        self.imu_delay = self._delay_range("imu_delay")

        rand = env_cfg.get("randomization", "nominal")
        self.rand_cfg = load_yaml("configs/randomization.yaml")[rand] if isinstance(rand, str) else (rand or {})
        jo = self.rand_cfg.get("joint_offset") if self.rand_cfg.get("enable") else None
        self.joint_offset_range = _rng(jo)

    # ------------------------------------------------------------ MjxEnv API
    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self.n

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    # --------------------------------------------------------------- helpers
    def _delay_range(self, key):
        lo, hi = self.noise.get(key, [0, 0]) if self.noise.get("level", 0.0) > 0 else (0, 0)
        return int(lo), int(hi)

    def _sensor_data(self, data, name):
        a, d = self._sensor[name]
        return data.sensordata[a : a + d]

    def _uniform(self, key, scale, shape):
        return (2 * jax.random.uniform(key, shape) - 1) * self.noise_level * jp.asarray(scale)

    def _gravity(self, data):
        return data.site_xmat[self.imu_site].reshape(3, 3).T @ jp.array([0.0, 0.0, -1.0])

    def _feet_contact(self, data):
        c = data._impl.contact
        g1, g2, active = c.geom[:, 0], c.geom[:, 1], c.dist < 0
        f = self.floor_geom
        return jp.stack([
            jp.any(active & (((g1 == f) & (g2 == fg)) | ((g2 == f) & (g1 == fg)))) for fg in self.foot_geoms
        ])

    def sample_command(self, key):
        c = self.cfg.get("commands", {})
        k = jax.random.split(key, 6)
        vel = jp.array([
            jax.random.uniform(k[0], minval=_rng(c.get("lin_vel_x"))[0], maxval=_rng(c.get("lin_vel_x"))[1]),
            jax.random.uniform(k[1], minval=_rng(c.get("lin_vel_y"))[0], maxval=_rng(c.get("lin_vel_y"))[1]),
            jax.random.uniform(k[2], minval=_rng(c.get("ang_vel_yaw"))[0], maxval=_rng(c.get("ang_vel_yaw"))[1]),
        ])
        vel = vel * (jax.random.uniform(k[3]) >= c.get("zero_prob", 0.0))
        head_cfg = c.get("head", {})
        lo = jp.array([_rng(head_cfg.get(h))[0] for h in HEAD_COMMANDS])
        hi = jp.array([_rng(head_cfg.get(h))[1] for h in HEAD_COMMANDS])
        head = jax.random.uniform(k[4], (4,), minval=lo, maxval=hi)
        head = head * (jax.random.uniform(k[5]) >= c.get("head_zero_prob", 1.0))
        return jp.concatenate([vel, head])

    def _sample_push(self, key, step):
        push = self.cfg.get("push", {})
        if not push.get("enable"):
            return jp.asarray(-1, dtype=jp.int32)
        lo, hi = push["interval_range"]
        return (step + jp.round(jax.random.uniform(key, minval=lo, maxval=hi) / self.dt)).astype(jp.int32)

    def _measure(self, data, info, key, init=False):
        """Sensor readings the robot would report: calibration offset, noise, IMU latency."""
        k = jax.random.split(key, 5)
        q = data.qpos[self.qpos_adr] - info["joint_offset"] + self._uniform(k[0], self.joint_noise, (self.n,))
        qd = data.qvel[self.qvel_adr] + self._uniform(k[1], self.noise.get("joint_vel", 0.0), (self.n,))
        gyro = self._sensor_data(data, "gyro") + self._uniform(k[2], self.noise.get("gyro", 0.0), (3,))
        acc = self._sensor_data(data, "accelerometer") + self._uniform(k[3], self.noise.get("accelerometer", 0.0), (3,))
        imu = jp.concatenate([gyro, acc])
        if init:
            queue = jp.tile(imu, (self.imu_delay[1] + 1, 1))
        else:
            queue = jp.roll(info["imu_queue"], 1, axis=0).at[0].set(imu)
        d = jax.random.randint(k[4], (), self.imu_delay[0], self.imu_delay[1] + 1)
        return q, qd, queue[d, :3], queue[d, 3:], self._feet_contact(data), queue

    def _observe(self, data, info, key, init=False):
        q, qd, gyro, acc, contact, queue = self._measure(data, info, key, init)
        phase = gait_phase(jp, info["phase_i"], info["phase_started"], self.P)
        obs = build_observation(jp, gyro, acc, info["command"], q, qd, self.home, info["action_history"],
                                info["motor_targets"], contact, phase)
        info = {**info, "imu_queue": queue, "phase_i": (info["phase_i"] + 1) % self.P,
                "phase_started": jp.asarray(True)}
        priv = jp.concatenate([
            obs,
            self._sensor_data(data, "gyro"), self._sensor_data(data, "accelerometer"), self._gravity(data),
            self._sensor_data(data, "local_linvel"), data.qvel[self.bv : self.bv + 6],
            data.qpos[self.qpos_adr] - self.home, data.qvel[self.qvel_adr],
            data.qpos[self.bq + 2 : self.bq + 3], data.actuator_force[self.act_ids],
            self._feet_contact(data).astype(jp.float32),
            self._sensor_data(data, "left_foot_global_linvel"), self._sensor_data(data, "right_foot_global_linvel"),
            info["feet_air_time"], info["joint_offset"],
        ])
        return {"state": obs, "privileged_state": priv}, info

    def _terminated(self, data):
        t = self.cfg.get("termination", {})
        up_z = -self._gravity(data)[2]
        nan = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        return nan | (up_z < np.cos(t.get("max_tilt", np.pi / 2))) | (data.qpos[self.bq + 2] < t.get("min_height", 0.0))

    # ------------------------------------------------------------ reset/step
    def reset(self, rng: jax.Array) -> mjx_env.State:
        k = jax.random.split(rng, 10)
        init = self.cfg.get("init", {})
        joint_offset = jax.random.uniform(k[0], (self.n,), minval=self.joint_offset_range[0],
                                          maxval=self.joint_offset_range[1])
        s0, s1 = init.get("joint_scale", [1.0, 1.0])
        q = jp.clip(self.home * jax.random.uniform(k[1], (self.n,), minval=s0, maxval=s1),
                    self.hard_lower, self.hard_upper)
        xy = jax.random.uniform(k[2], (2,), minval=-1, maxval=1) * init.get("base_xy", 0.0)
        yaw = jax.random.uniform(k[3], minval=-1, maxval=1) * init.get("yaw", 0.0)
        qpos = self._init_qpos.at[self.qpos_adr].set(q)
        qpos = qpos.at[self.bq : self.bq + 3].set(jp.array([xy[0], xy[1], self.home_height]))
        qpos = qpos.at[self.bq + 3 : self.bq + 7].set(jp.array([jp.cos(yaw / 2), 0.0, 0.0, jp.sin(yaw / 2)]))
        qvel = jp.zeros(self.mjx_model.nv).at[self.bv : self.bv + 6].set(
            jax.random.uniform(k[4], (6,), minval=-1, maxval=1) * init.get("base_vel", 0.0))
        ctrl = jp.zeros(self.mjx_model.nu).at[self.act_ids].set(q + joint_offset)
        data = mjx.make_data(self.mjx_model).replace(qpos=qpos, qvel=qvel, ctrl=ctrl)
        # Rest the lower foot on the ground (see DuckEnv.reset).
        foot_z = mjx.kinematics(self.mjx_model, data).site_xpos[self.foot_sites, 2]
        data = data.replace(qpos=qpos.at[self.bq + 2].add(self.foot_z_ref - foot_z.min()))
        data = mjx.forward(self.mjx_model, data)

        step0 = jp.asarray(0, dtype=jp.int32)
        info = {
            "rng": k[5],
            "step": step0,
            "command": self.sample_command(k[6]),
            "action_history": jp.zeros((3, self.n)),
            "motor_targets": self.home,
            "phase_i": jp.asarray(0, dtype=jp.int32),
            "phase_started": jp.asarray(False),
            "target_queue": jp.tile(self.home, (self.action_delay[1] + 1, 1)),
            "imu_queue": jp.zeros((self.imu_delay[1] + 1, 6)),
            "feet_air_time": jp.zeros(2),
            "last_contact": jp.zeros(2, dtype=bool),
            "last_action": jp.zeros(self.n),
            "next_push": self._sample_push(k[7], step0),
            "joint_offset": joint_offset,
        }
        obs, info = self._observe(data, info, k[8], init=True)
        metrics = {f"reward/{name}": jp.zeros(()) for name, w in self.cfg["rewards"]["scales"].items() if w}
        return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        info = dict(state.info)
        k = jax.random.split(info["rng"], 8)
        info["rng"] = k[0]
        cmd = info["command"]

        targets = process_action(jp, self.ap, action, cmd, info["motor_targets"])
        info["action_history"] = jp.roll(info["action_history"], 1, axis=0).at[0].set(action)
        info["motor_targets"] = targets
        queue = jp.roll(info["target_queue"], 1, axis=0).at[0].set(targets)
        info["target_queue"] = queue
        d = jax.random.randint(k[1], (), self.action_delay[0], self.action_delay[1] + 1)
        ctrl = state.data.ctrl.at[self.act_ids].set(queue[d] + info["joint_offset"])

        data = state.data
        push = self.cfg.get("push", {})
        if push.get("enable"):
            do_push = info["step"] == info["next_push"]
            theta = jax.random.uniform(k[2], maxval=2 * jp.pi)
            mag = jax.random.uniform(k[3], minval=push["magnitude_range"][0], maxval=push["magnitude_range"][1])
            kick = mag * jp.array([jp.cos(theta), jp.sin(theta)]) * do_push
            data = data.replace(qvel=data.qvel.at[self.bv : self.bv + 2].add(kick))
            info["next_push"] = jp.where(do_push, self._sample_push(k[4], info["step"]), info["next_push"])

        data = mjx_env.step(self.mjx_model, data, ctrl, self.n_substeps)
        info["step"] = info["step"] + 1

        contact = self._feet_contact(data)
        first_contact = (info["feet_air_time"] > 0) & (contact | info["last_contact"])
        air = info["feet_air_time"] + self.dt
        terminated = self._terminated(data)
        ref_frame = self.ref.frame(cmd, info["phase_i"], jp) if self.cfg.get("use_reference_motion") else None

        ctx = RewardContext(
            command=cmd,
            joint_pos=data.qpos[self.qpos_adr],
            joint_vel=data.qvel[self.qvel_adr],
            home=self.home,
            leg_idx=self.leg_idx,
            action=action,
            last_action=info["last_action"],
            torques=data.actuator_force[self.act_ids],
            gravity=self._gravity(data),
            local_linvel=self._sensor_data(data, "local_linvel"),
            gyro=self._sensor_data(data, "gyro"),
            base_height=data.qpos[self.bq + 2],
            home_height=self.home_height,
            base_qvel=data.qvel[self.bv : self.bv + 6],
            feet_contact=contact,
            feet_linvel=jp.stack([self._sensor_data(data, "left_foot_global_linvel"),
                                  self._sensor_data(data, "right_foot_global_linvel")]),
            feet_air_time=air,
            first_contact=first_contact,
            soft_lower=self.soft_lower,
            soft_upper=self.soft_upper,
            ref_frame=ref_frame,
            terminated=terminated,
            xp=jp,
        )
        rc = self.cfg["rewards"]
        terms = compute_rewards(ctx, rc["scales"], rc.get("params", {}))
        reward = sum(terms.values()) * self.dt
        if rc.get("clip_min") is not None:
            reward = jp.maximum(reward, rc["clip_min"])
        reward = jp.where(jp.isfinite(reward), reward, 0.0)

        info["feet_air_time"] = air * ~contact
        info["last_contact"] = contact
        info["last_action"] = action

        rs = self.cfg.get("commands", {}).get("resample_steps")
        if rs:
            info["command"] = jp.where(info["step"] % rs == 0, self.sample_command(k[5]), cmd)

        obs, info = self._observe(data, info, k[6])
        metrics = {**state.metrics, **{f"reward/{name}": v for name, v in terms.items()}}
        return state.replace(data=data, obs=obs, reward=reward, done=terminated.astype(jp.float32),
                             metrics=metrics, info=info)


def make_mjx_env(config_path: str, overrides: dict | None = None) -> DuckMjxEnv:
    from biped.config import deep_update, load_config

    cfg = load_config(config_path)["env"]
    if overrides:
        cfg = deep_update(cfg, overrides)
    return DuckMjxEnv(cfg)
