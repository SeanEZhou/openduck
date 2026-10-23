"""Fall rate on hard commands (turn in place, forward 0.15 m/s) with one realism factor at a time.

    python scripts/realism_ablation.py <name> <policy.onnx>
"""
import sys, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from biped.envs import make_env
from biped.evaluation.benchmarks import rollout
from biped.policy import load_policy
from biped.config import load_yaml
R = load_yaml("configs/randomization.yaml")["randomized"]
name, path = sys.argv[1], sys.argv[2]
pol = load_policy(path)
conds = {
    "nominal": {},
    "latency only": {"noise": {"level": 1e-9, "action_delay": [0, 2], "imu_delay": [0, 2]}},
    "sensor noise only": {"noise": {"level": 1.0, "action_delay": [0, 0], "imu_delay": [0, 0]}},
    "calibration offsets only": {"randomization": {"enable": True, "joint_offset": R["joint_offset"]}},
    "trunk mass/CoM only": {"randomization": {"enable": True, "trunk_added_mass": R["trunk_added_mass"], "trunk_com_offset": R["trunk_com_offset"]}},
    "damping+torque only": {"randomization": {"enable": True, "joint_damping_scale": R["joint_damping_scale"], "torque_limit_scale": R["torque_limit_scale"]}},
    "all (realistic)": {"randomization": "randomized", "noise": {"level": 1.0}},
}
cmds = [np.r_[0, 0, 0.5, 0, 0, 0, 0], np.r_[0, 0, -0.5, 0, 0, 0, 0], np.r_[0.15, 0, 0, 0, 0, 0, 0]]
row = []
for cname, ov in conds.items():
    env = make_env("configs/eval_baseline.yaml", {"episode_length": 750, "commands": {"resample_steps": 0}, "push": {"enable": False}, **ov})
    ms = [rollout(env, pol, c, seed=s, max_steps=750) for c in cmds for s in range(10)]
    row.append(f"{cname}: {np.mean([m.fell for m in ms]):.2f}")
print(f"{name:10s} falls (turn ±0.5 rad/s, forward 0.15 m/s; 30 episodes) | " + " | ".join(row))
