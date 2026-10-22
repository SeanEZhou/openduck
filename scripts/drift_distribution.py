"""Heading drift while walking straight, over randomized-physics episodes (more representative
of hardware than a single nominal-model episode, whose drift is one sample of a wide spread).

    python scripts/drift_distribution.py --policy results/mjx_walking_v5_seed0/policy.onnx --out results/mjx_walking_v5_seed0/drift.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from biped.envs import make_env  # noqa: E402
from biped.envs.locomotion import quat_yaw  # noqa: E402
from biped.policy import load_policy  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True)
    p.add_argument("--config", default="configs/eval_baseline.yaml")
    p.add_argument("--vx", type=float, default=0.15)
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--out")
    a = p.parse_args()
    env = make_env(a.config, {"episode_length": 10**6, "commands": {"resample_steps": 0}, "push": {"enable": False},
                              "randomization": "randomized", "noise": {"level": 1.0}})
    pol, r = load_policy(a.policy), env.robot
    drift, dist, falls = [], [], 0
    for s in range(a.episodes):
        env.reset(seed=100 + s)
        env.command = np.r_[a.vx, 0, 0, 0, 0, 0, 0]
        env.runtime.reset()
        obs = env.runtime.observe(env._measure(), env.command)
        y0, x0 = quat_yaw(np, r.base_quat.copy()), r.base_pos[:2].copy()
        fell = False
        for _ in range(int(10.0 / env.dt)):
            obs, _, term, _, _ = env.step(pol(obs))
            if term:
                fell = True
                break
        if fell:
            falls += 1
            continue
        drift.append(float(np.degrees((quat_yaw(np, r.base_quat) - y0 + np.pi) % (2 * np.pi) - np.pi)))
        dist.append(float(np.linalg.norm(r.base_pos[:2] - x0)))
    d = np.array(drift)
    res = {"policy": a.policy, "vx": a.vx, "episodes": a.episodes, "falls": falls,
           "mean_deg": round(float(d.mean()), 1), "std_deg": round(float(d.std()), 1),
           "mean_abs_deg": round(float(np.abs(d).mean()), 1), "distance_m": round(float(np.mean(dist)), 2)}
    print(json.dumps(res))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
