"""Quantitative rollout metrics, shared by sim evaluation and (later) hardware logs."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from biped.envs.locomotion import DuckEnv


@dataclass
class EpisodeMetrics:
    command: list[float]
    duration_s: float
    fell: bool
    distance_m: float           # planar displacement of the base
    vx_error: float             # RMS, body-frame, after 1 s warm-up
    vy_error: float
    yaw_rate_error: float
    mean_tilt_deg: float
    joint_tracking_rms: float   # |measured - commanded| over leg joints, rad
    action_rate_rms: float

    def as_dict(self) -> dict:
        return asdict(self)


def rollout(env: DuckEnv, policy, command: np.ndarray | None = None, seed: int = 0,
            max_steps: int | None = None, frames: list | None = None) -> EpisodeMetrics:
    obs, _ = env.reset(seed=seed)
    if command is not None:
        env.command = np.asarray(command, dtype=np.float64)
        env.runtime.reset()
        obs = env.runtime.observe(env._measure(), env.command)
    r = env.robot
    start = r.base_pos[:2].copy()
    max_steps = max_steps or env.cfg.get("episode_length", 1000)
    warmup = int(1.0 / env.dt)
    v_err, tilt, track, arate = [], [], [], []
    prev_a = np.zeros(env.n)
    fell = False
    t = 0
    for t in range(1, max_steps + 1):
        a = policy(obs)
        obs, _, terminated, truncated, _ = env.step(a)
        legs = env.leg_idx
        track.append(np.abs(r.data.qpos[r.qpos_adr][legs] - env.runtime.motor_targets[legs]).mean())
        arate.append(np.sum((np.asarray(a) - prev_a) ** 2))
        prev_a = np.asarray(a)
        tilt.append(np.degrees(np.arccos(np.clip(-r.gravity_in_imu()[2], -1, 1))))
        if t > warmup:
            lv, w = r.local_linvel(), r.sensor("gyro")
            v_err.append([lv[0] - env.command[0], lv[1] - env.command[1], w[2] - env.command[2]])
        if frames is not None:
            frames.append(env.render())
        if terminated:
            fell = True
            break
        if truncated and t >= max_steps:
            break
    v = np.sqrt(np.mean(np.square(v_err), axis=0)) if v_err else np.full(3, np.nan)
    return EpisodeMetrics(
        command=[float(x) for x in env.command[:3]],
        duration_s=t * env.dt,
        fell=fell,
        distance_m=float(np.linalg.norm(r.base_pos[:2] - start)),
        vx_error=float(v[0]),
        vy_error=float(v[1]),
        yaw_rate_error=float(v[2]),
        mean_tilt_deg=float(np.mean(tilt)),
        joint_tracking_rms=float(np.sqrt(np.mean(np.square(track)))),
        action_rate_rms=float(np.sqrt(np.mean(arate))),
    )


def summarize(results: list[EpisodeMetrics]) -> dict:
    keys = ["duration_s", "distance_m", "vx_error", "vy_error", "yaw_rate_error",
            "mean_tilt_deg", "joint_tracking_rms", "action_rate_rms"]
    out = {"episodes": len(results), "fall_rate": float(np.mean([m.fell for m in results]))}
    for k in keys:
        out[k] = float(np.nanmean([getattr(m, k) for m in results]))
    return out
