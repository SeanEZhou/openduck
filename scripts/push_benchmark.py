"""Pooled push-survival benchmark in two conditions, used to compare push robustness.

    python scripts/push_benchmark.py --policy results/<run>/policy.onnx [--out results/<run>/push_bench.json]

Standing (zero command) robot, kicked at t = 3 s; success = no fall in the next 3 s
(the trial in scripts/push_recovery.py). Trials: 64 directions x 0.4 / 0.6 / 0.8 m/s.
- nominal:   nominal physics, 0-40 ms actuation and IMU latency (the seed-study protocol);
- realistic: plus randomized physics and sensor noise, re-sampled per trial.
Both report the mean survival and its standard error.
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

MAGNITUDES = (0.4, 0.6, 0.8)
CONDITIONS = {
    "nominal": {"noise": {"level": 1e-9, "action_delay": [0, 2], "imu_delay": [0, 2]}},
    "realistic": {"randomization": "randomized", "noise": {"level": 1.0}},
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True)
    p.add_argument("--config", default="configs/eval_baseline.yaml")
    p.add_argument("--directions", type=int, default=64)
    p.add_argument("--cmd", type=float, nargs=3, default=[0, 0, 0])
    p.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out")
    a = p.parse_args()

    # each (condition, magnitude) is split into 4 chunks of directions, run in parallel
    chunks, per = 4, a.directions // 4
    jobs = []
    for cond in a.conditions:
        for mag in MAGNITUDES:
            for c in range(chunks):
                jobs.append((a.policy, a.config, cond, mag, per, a.cmd, c * per))
    res = {cond: {f"{m:.1f}": [] for m in MAGNITUDES} for cond in a.conditions}
    with ProcessPoolExecutor(a.workers) as ex:
        for cond, mag, ok in ex.map(_run_chunk, jobs):
            res[cond][f"{mag:.1f}"] += ok

    summary = {"policy": a.policy, "cmd": a.cmd}
    for cond, by_mag in res.items():
        allok = np.concatenate([np.asarray(v, float) for v in by_mag.values()])
        summary[cond] = {"pooled": round(float(allok.mean()), 3),
                         "se": round(float(allok.std() / np.sqrt(len(allok))), 3),
                         "trials": int(len(allok)),
                         "by_magnitude": {m: round(float(np.mean(v)), 3) for m, v in by_mag.items()}}
        s = summary[cond]
        print(f"{cond:9s} pooled {s['pooled']:.2f} ± {s['se']:.2f}  " +
              "  ".join(f"{m}: {v:.2f}" for m, v in s["by_magnitude"].items()), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(summary, indent=2))


def _run_chunk(job):
    """One chunk of directions; directions are interleaved over the full circle."""
    policy_path, config, cond, mag, per, cmd, off = job
    from biped.envs import make_env
    from biped.policy import load_policy
    from push_recovery import trial

    env = make_env(config, {"episode_length": 10**6, "commands": {"resample_steps": 0}, "push": {"enable": False},
                           **CONDITIONS[cond]})
    pol = load_policy(policy_path)
    n = per * 4
    ok = []
    for j in range(per):
        i = off + j
        _, s = trial(env, pol, np.r_[cmd, 0, 0, 0, 0], mag, 2 * np.pi * i / n, seed=i)
        ok.append(s)
    return cond, mag, ok


if __name__ == "__main__":
    main()
