"""Behaviour diagnostics for a walking/standing policy (CPU MuJoCo, nominal model).

    python scripts/diagnose.py --policy results/mjx_walking_v5_seed0/policy.onnx [--config configs/eval_legs.yaml]

Reports, as JSON:
- push reaction: after a 0.6 m/s kick from behind (forward push) at t = 1 s, the time to the
  first foot lift-off, number of steps in the next 2 s, total forward foot travel, peak leg joint
  speed, and whether it fell;
- heading drift: yaw change and lateral offset over 10 s of straight walking (0.1 and 0.15 m/s);
- mirror asymmetry: on states visited while walking straight, how far the policy's action differs
  from the mirror image of its action on the mirrored observation (0 = perfectly left/right symmetric).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from biped.control.observations import observation_layout  # noqa: E402
from biped.envs import make_env  # noqa: E402
from biped.envs.locomotion import quat_yaw  # noqa: E402
from biped.policy import load_policy  # noqa: E402


def start(env, cmd):
    env.reset(seed=0)
    env.command = np.asarray(cmd, dtype=float)
    env.runtime.reset()
    return env.runtime.observe(env._measure(), env.command)


def push_reaction(env, policy, mag=0.6, deg=0.0, push_t=1.0, window=2.0):
    obs = start(env, np.zeros(7))
    r, dt = env.robot, env.dt
    kp, kn = int(push_t / dt), int((push_t + window) / dt)
    first_lift, steps, prev_contact, fell = None, 0, None, False
    foot_x0, peak_vel = None, 0.0
    for k in range(kn):
        if k == kp:
            env.apply_push(mag * np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg))]))
            foot_x0 = r.feet_pos()[:, 0].copy()
        obs, _, term, _, _ = env.step(policy(obs))
        if k >= kp:
            c = r.feet_contact()
            if prev_contact is not None:
                lifts = prev_contact & ~c
                if lifts.any() and first_lift is None:
                    first_lift = (k - kp) * dt
                steps += int((~prev_contact & c).sum())  # touchdowns
            prev_contact = c
            peak_vel = max(peak_vel, float(np.abs(r.data.qvel[r.qvel_adr][env.leg_idx]).max()))
        if term:
            fell = True
            break
    travel = float(np.abs(r.feet_pos()[:, 0] - foot_x0).sum()) if foot_x0 is not None else 0.0
    return {"first_liftoff_s": first_lift, "touchdowns_2s": steps, "foot_travel_m": round(travel, 3),
            "peak_leg_joint_vel": round(peak_vel, 2), "fell": fell}


def drift(env, policy, vx, seconds=10.0):
    obs = start(env, np.r_[vx, 0, 0, 0, 0, 0, 0])
    r = env.robot
    y0, p0 = quat_yaw(np, r.base_quat.copy()), r.base_pos[:2].copy()
    visited = []
    for _ in range(int(seconds / env.dt)):
        visited.append(obs)
        obs, _, term, _, _ = env.step(policy(obs))
        if term:
            break
    d = r.base_pos[:2] - p0
    yaw = float(np.degrees((quat_yaw(np, r.base_quat) - y0 + np.pi) % (2 * np.pi) - np.pi))
    return {"heading_change_deg": round(yaw, 1), "lateral_m": round(float(d[1]), 3),
            "forward_m": round(float(d[0]), 3)}, np.array(visited)


def mirror_obs_fn(env):
    """Mirror an observation across the robot's sagittal plane (y -> -y)."""
    spec, n = env.spec, env.n
    perm, sign = spec.mirror_permutation()
    lay = observation_layout(n)
    head = spec.indices("head")

    def mirror_joint_vec(v):
        return sign * v[perm]

    def m(o):
        o = o.copy()
        o[lay["gyro"]] = o[lay["gyro"]] * [-1, 1, -1]            # angular velocity is a pseudovector
        o[lay["accelerometer"]] = o[lay["accelerometer"]] * [1, -1, 1]
        c = o[lay["command"]].copy()
        c[1], c[2] = -c[1], -c[2]
        c[5], c[6] = -c[5], -c[6]                                 # head yaw, head roll
        o[lay["command"]] = c
        for key in ("joint_pos", "joint_vel", "last_action", "last_last_action", "last_last_last_action"):
            o[lay[key]] = mirror_joint_vec(o[lay[key]])
        # motor targets are absolute angles: mirror them about the mirrored home pose
        home = spec.home
        o[lay["motor_targets"]] = mirror_joint_vec(o[lay["motor_targets"]] - home) + home
        o[lay["feet_contact"]] = o[lay["feet_contact"]][::-1]
        o[lay["phase"]] = -o[lay["phase"]]                        # swapping legs = half a gait period
        return o.astype(np.float32)

    _ = head
    return m, mirror_joint_vec


def mirror_asymmetry(env, policy, visited):
    m, mj = mirror_obs_fn(env)
    a = np.array([policy(o) for o in visited])
    am = np.array([mj(np.asarray(policy(m(o)))) for o in visited])
    legs = env.leg_idx
    per_joint = np.sqrt(np.mean((a - am)[:, legs] ** 2, axis=0))
    return {"rms_action_mismatch": round(float(np.sqrt(np.mean((a - am)[:, legs] ** 2))), 3),
            "action_rms": round(float(np.sqrt(np.mean(a[:, legs] ** 2))), 3),
            "worst_joint": env.spec.joint_names[legs[int(np.argmax(per_joint))]]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True)
    p.add_argument("--config", default="configs/eval_baseline.yaml")
    p.add_argument("--out")
    a = p.parse_args()
    env = make_env(a.config, {"episode_length": 10**6, "commands": {"resample_steps": 0}, "push": {"enable": False}})
    pol = load_policy(a.policy)
    res = {"policy": a.policy, "push_reaction": {}}
    for deg, name in ((0, "from_behind"), (90, "from_right"), (180, "from_front")):
        res["push_reaction"][name] = push_reaction(env, pol, deg=deg)
    d1, visited = drift(env, pol, 0.1)
    d2, visited2 = drift(env, pol, 0.15)
    res["drift_0.10"], res["drift_0.15"] = d1, d2
    res["mirror_asymmetry"] = mirror_asymmetry(env, pol, np.concatenate([visited[::5], visited2[::5]]))
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
