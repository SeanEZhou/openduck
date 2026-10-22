"""Push-recovery benchmark: success rate vs push magnitude (CPU MuJoCo, nominal model).

A planar base-velocity kick of fixed magnitude is applied at t = 3 s in N evenly
spaced directions; success = no fall in the following 3 s.

    python scripts/push_recovery.py --policy results/mjx_walking_seed0/policy.onnx \
        --out results/mjx_walking_seed0/push_recovery.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from biped.envs import make_env  # noqa: E402
from biped.policy import load_policy  # noqa: E402


def trial(env, policy, cmd, magnitude, theta, push_t=3.0, after=3.0, seed=0):
    obs, _ = env.reset(seed=seed)
    env.command = cmd
    env.runtime.reset()
    obs = env.runtime.observe(env._measure(), cmd)
    r = env.robot
    k_push = int(push_t / env.dt)
    for k in range(int((push_t + after) / env.dt)):
        if k == k_push:
            r.data.qvel[r.base_qvel_adr : r.base_qvel_adr + 2] += magnitude * np.array([np.cos(theta), np.sin(theta)])
        obs, _, term, _, _ = env.step(policy(obs))
        if term:
            return k >= k_push, False  # (reached the push, survived)
    return True, True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True)
    p.add_argument("--config", default="configs/eval_baseline.yaml")
    p.add_argument("--magnitudes", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8, 1.0])
    p.add_argument("--directions", type=int, default=8)
    p.add_argument("--cmd", type=float, nargs=3, default=[0, 0, 0])
    p.add_argument("--latency", action="store_true",
                   help="0-2 step actuation and IMU latency (as on hardware), no sensor noise")
    p.add_argument("--randomize", action="store_true",
                   help="sample physical parameters per trial (configs/randomization.yaml: randomized)")
    p.add_argument("--out")
    a = p.parse_args()

    ov = {"episode_length": 10**6, "commands": {"resample_steps": 0}, "push": {"enable": False}}
    if a.latency:
        ov["noise"] = {"level": 1e-9, "action_delay": [0, 2], "imu_delay": [0, 2]}
    if a.randomize:
        ov["randomization"] = "randomized"
    env = make_env(a.config, ov)
    policy = load_policy(a.policy)
    cmd = np.r_[a.cmd, 0, 0, 0, 0]
    res = {}
    for mag in a.magnitudes:
        ok = []
        for i in range(a.directions):
            reached, survived = trial(env, policy, cmd, mag, 2 * np.pi * i / a.directions, seed=i)
            if not reached:
                raise RuntimeError("fell before the push; benchmark invalid for this policy/command")
            ok.append(survived)
        res[f"{mag:.1f}"] = float(np.mean(ok))
        print(f"push {mag:.1f} m/s: success {res[f'{mag:.1f}']:.2f}")
    if a.out:
        Path(a.out).write_text(json.dumps({"policy": a.policy, "cmd": a.cmd, "latency": a.latency, "randomized": a.randomize,
                                           "success_by_magnitude": res}, indent=2))


if __name__ == "__main__":
    main()
