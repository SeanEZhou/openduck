"""Gymnasium environment for Open Duck Mini v2 standing and walking.

The actor only ever sees `PolicyRuntime.observe(noisy_measured_state)`, which
is exactly what the hardware runtime computes. Everything privileged
(true base velocity, height, contacts from the simulator, reference motion)
is used only for rewards and termination.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from biped.config import RobotSpec, load_robot, load_yaml
from biped.control.reference_motion import ReferenceMotion
from biped.control.runtime import PolicyRuntime
from biped.envs.randomization import DomainRandomizer
from biped.envs.rewards import RewardContext, compute_rewards
from biped.hardware.interface import RobotState
from biped.hardware.mujoco import MujocoRobot

HEAD_COMMANDS = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")


def per_joint_noise(spec: RobotSpec, scales: dict[str, float]) -> np.ndarray:
    """Per-joint noise amplitude from {"hip": .., "knee": .., "ankle": .., "head": ..}."""
    out = np.zeros(spec.num_joints)
    for i, j in enumerate(spec.joints):
        for key, v in scales.items():
            if key in j.name or (key == "head" and j.group == "head"):
                out[i] = v
    return out


def _range(r) -> tuple[float, float]:
    return (float(r[0]), float(r[1])) if r is not None else (0.0, 0.0)


class DuckEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, config: dict[str, Any], spec: RobotSpec | None = None, render_mode: str | None = None):
        self.cfg = config
        self.spec = spec or load_robot()
        self.robot = MujocoRobot(self.spec)
        self.ref = ReferenceMotion(self.spec.model_path("reference_motion"))
        self.runtime = PolicyRuntime(self.spec, self.ref.steps_per_period, **config.get("action", {}))
        self.dt = self.spec.control["policy_dt"]
        self.n = self.spec.num_joints

        rand_cfg = config.get("randomization", "nominal")
        if isinstance(rand_cfg, str):
            rand_cfg = load_yaml("configs/randomization.yaml")[rand_cfg]
        self.randomizer = DomainRandomizer(self.robot, rand_cfg)

        self.noise = config.get("noise", {})
        self._joint_noise = per_joint_noise(self.spec, self.noise.get("joint_pos", {}))
        self.soft_lower, self.soft_upper = self.spec.soft_limits()
        self.leg_idx = self.spec.leg_indices

        settled = self.robot.settled_home_qpos()
        self.home_height = float(settled[self.robot.base_qpos_adr + 2])
        self.foot_z_ref = self.robot.min_foot_site_z(settled)

        hi = np.full(self.runtime.obs_dim, np.inf, dtype=np.float32)
        self.observation_space = gym.spaces.Box(-hi, hi, dtype=np.float32)
        a = float(config.get("action_bound", 4.0))
        self.action_space = gym.spaces.Box(-a, a, shape=(self.n,), dtype=np.float32)

        self.render_mode = render_mode
        self._renderer = None
        self.np_random = np.random.default_rng()
        self.command = np.zeros(7)

    # ---------------------------------------------------------------- utils
    def _uniform_noise(self, scale, shape) -> np.ndarray:
        level = self.noise.get("level", 0.0)
        return (2 * self.np_random.random(shape) - 1) * level * np.asarray(scale)

    def sample_command(self) -> np.ndarray:
        c = self.cfg.get("commands", {})
        rng = self.np_random
        cmd = np.zeros(7)
        if rng.random() >= c.get("zero_prob", 0.0):
            cmd[0] = rng.uniform(*_range(c.get("lin_vel_x")))
            cmd[1] = rng.uniform(*_range(c.get("lin_vel_y")))
            cmd[2] = rng.uniform(*_range(c.get("ang_vel_yaw")))
        if rng.random() >= c.get("head_zero_prob", 1.0):
            head = c.get("head", {})
            for k, name in enumerate(HEAD_COMMANDS):
                cmd[3 + k] = rng.uniform(*_range(head.get(name)))
        return cmd

    # ------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        rng, init = self.np_random, self.cfg.get("init", {})
        r = self.robot

        self.randomizer.randomize(rng)
        r.reset_to_home()
        q = self.spec.home * rng.uniform(*init.get("joint_scale", [1.0, 1.0]), self.n)
        q = np.clip(q, self.spec.lower, self.spec.upper)
        r.set_joint_pos(q)
        xy = rng.uniform(-1, 1, 2) * init.get("base_xy", 0.0)
        yaw = rng.uniform(-1, 1) * init.get("yaw", 0.0)
        r.set_base_pose(
            pos=np.array([xy[0], xy[1], self.home_height]),
            quat=np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]),
        )
        # Rest the lower foot on the ground: perturbed joint angles change the leg length, and a
        # fixed base height would start the feet up to ~1 cm inside the floor (violent ejection).
        r.data.qpos[r.base_qpos_adr + 2] += self.foot_z_ref - r.min_foot_site_z(r.data.qpos)
        r.set_base_vel(rng.uniform(-1, 1, 6) * init.get("base_vel", 0.0))
        r.data.ctrl[r.actuator_ids] = q + self.randomizer.joint_offset
        r.forward()

        self.runtime.reset()
        self.command = self.sample_command()
        self.step_count = 0
        self.feet_air_time = np.zeros(2)
        self.last_contact = np.zeros(2, dtype=bool)
        self.last_action = np.zeros(self.n)
        self.target_queue = [self.runtime.motor_targets.copy()] * (self._max_delay("action_delay") + 1)
        self.imu_queue = []
        push = self.cfg.get("push", {})
        self.next_push = self._sample_push_interval() if push.get("enable") else -1
        self.episode_terms: dict[str, float] = {}

        obs = self.runtime.observe(self._measure(), self.command)
        return obs, {}

    def _max_delay(self, key: str) -> int:
        return int(self.noise.get(key, [0, 0])[1]) if self.noise.get("level", 0.0) > 0 else 0

    def _sample_delay(self, key: str) -> int:
        lo, hi = self.noise.get(key, [0, 0]) if self.noise.get("level", 0.0) > 0 else (0, 0)
        return int(self.np_random.integers(lo, hi + 1))

    def _sample_push_interval(self) -> int:
        lo, hi = self.cfg["push"]["interval_range"]
        return self.step_count + int(round(self.np_random.uniform(lo, hi) / self.dt))

    def _measure(self) -> RobotState:
        """What the robot's sensors would report: offsets, noise and IMU latency applied."""
        s = self.robot.read_state()
        s.joint_pos = s.joint_pos - self.randomizer.joint_offset
        s.joint_pos = s.joint_pos + self._uniform_noise(self._joint_noise, self.n)
        s.joint_vel = s.joint_vel + self._uniform_noise(self.noise.get("joint_vel", 0.0), self.n)
        gyro = s.gyro + self._uniform_noise(self.noise.get("gyro", 0.0), 3)
        acc = s.accelerometer + self._uniform_noise(self.noise.get("accelerometer", 0.0), 3)
        self.imu_queue.insert(0, (gyro, acc))
        del self.imu_queue[self._max_delay("imu_delay") + 1 :]
        d = min(self._sample_delay("imu_delay"), len(self.imu_queue) - 1)
        s.gyro, s.accelerometer = self.imu_queue[d]
        s.gravity = s.gravity + self._uniform_noise(self.noise.get("gravity", 0.0), 3)
        return s

    def step(self, action: np.ndarray):
        r = self.robot
        action = np.asarray(action, dtype=np.float64)
        targets = self.runtime.act(action, self.command)

        # Actuation latency: apply targets issued `d` control steps ago.
        self.target_queue.insert(0, targets.copy())
        self.target_queue.pop()
        d = self._sample_delay("action_delay")
        r.write_targets(self.target_queue[d] + self.randomizer.joint_offset)

        if self.step_count == self.next_push:
            lo, hi = self.cfg["push"]["magnitude_range"]
            theta = self.np_random.uniform(0, 2 * np.pi)
            kick = self.np_random.uniform(lo, hi) * np.array([np.cos(theta), np.sin(theta)])
            r.data.qvel[r.base_qvel_adr : r.base_qvel_adr + 2] += kick
            self.next_push = self._sample_push_interval()

        r.step()
        self.step_count += 1

        contact = r.feet_contact()
        first_contact = (self.feet_air_time > 0) & (contact | self.last_contact)
        self.feet_air_time += self.dt
        air_time_at_touchdown = self.feet_air_time.copy()

        terminated = self._terminated()
        self.ref_i = self.runtime.phase.i
        ref_frame = self.ref.frame(self.command, self.ref_i) if self.cfg.get("use_reference_motion") else None

        ctx = RewardContext(
            command=self.command,
            joint_pos=r.data.qpos[r.qpos_adr],
            joint_vel=r.data.qvel[r.qvel_adr],
            home=self.spec.home,
            leg_idx=self.leg_idx,
            action=action,
            last_action=self.last_action,
            torques=r.actuator_force(),
            gravity=r.gravity_in_imu(),
            local_linvel=r.local_linvel(),
            gyro=r.sensor("gyro"),
            base_height=float(r.base_pos[2]),
            home_height=self.home_height,
            base_qvel=r.base_qvel,
            feet_contact=contact,
            feet_linvel=r.feet_linvel(),
            feet_air_time=air_time_at_touchdown,
            first_contact=first_contact,
            soft_lower=self.soft_lower,
            soft_upper=self.soft_upper,
            ref_frame=ref_frame,
            terminated=terminated,
        )
        rc = self.cfg["rewards"]
        terms = {k: float(v) for k, v in compute_rewards(ctx, rc["scales"], rc.get("params", {})).items()}
        reward = sum(terms.values()) * self.dt
        if rc.get("clip_min") is not None:
            reward = max(reward, rc["clip_min"])
        for k, v in terms.items():
            self.episode_terms[k] = self.episode_terms.get(k, 0.0) + v * self.dt

        self.feet_air_time *= ~contact
        self.last_contact = contact
        self.last_action = action

        c = self.cfg.get("commands", {})
        if c.get("resample_steps") and self.step_count % c["resample_steps"] == 0:
            self.command = self.sample_command()

        truncated = self.step_count >= self.cfg.get("episode_length", 1000)
        obs = self.runtime.observe(self._measure(), self.command)
        info = {"reward_terms": terms}
        if terminated or truncated:
            info["episode_terms"] = dict(self.episode_terms)
        return obs, float(reward), terminated, truncated, info

    def _terminated(self) -> bool:
        r, t = self.robot, self.cfg.get("termination", {})
        if r.has_nan():
            return True
        up_z = -r.gravity_in_imu()[2]  # cos(tilt)
        if up_z < np.cos(t.get("max_tilt", np.pi / 2)):
            return True
        return float(r.base_pos[2]) < t.get("min_height", 0.0)

    # ------------------------------------------------------------- render
    def render(self):
        import mujoco

        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.robot.model, 480, 640)
            self._cam = mujoco.MjvCamera()
            self._cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self._cam.trackbodyid = self.robot.root_body
            self._cam.distance, self._cam.elevation, self._cam.azimuth = 0.9, -15, 135
        self._renderer.update_scene(self.robot.data, self._cam)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def make_env(config_path: str, overrides: dict | None = None, **kwargs) -> DuckEnv:
    from biped.config import deep_update, load_config

    cfg = load_config(config_path)["env"]
    if overrides:
        cfg = deep_update(cfg, overrides)
    return DuckEnv(cfg, **kwargs)
