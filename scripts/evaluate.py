"""Evaluate a policy (ONNX file or SB3 run directory) on a set of commands.

Examples:
    python scripts/evaluate.py --policy third_party/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx \
        --config configs/eval_baseline.yaml --out results/baseline_upstream_onnx2.json
    python scripts/evaluate.py --policy results/standing_seed0 --config configs/train_standing.yaml \
        --commands stand --episodes 20 --video results/standing.mp4
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from biped.envs import make_env  # noqa: E402
from biped.evaluation.benchmarks import rollout, summarize  # noqa: E402
from biped.policy import load_policy  # noqa: E402

COMMAND_SETS = {
    "stand": [[0, 0, 0]],
    "walk": [[0, 0, 0], [0.1, 0, 0], [0.15, 0, 0], [-0.1, 0, 0], [0, 0.1, 0], [0, -0.1, 0],
             [0, 0, 0.5], [0, 0, -0.5], [0.1, 0, 0.3]],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True)
    p.add_argument("--config", default="configs/eval_baseline.yaml")
    p.add_argument("--commands", default="walk", choices=COMMAND_SETS)
    p.add_argument("--episodes", type=int, default=3, help="episodes per command")
    p.add_argument("--steps", type=int, default=750, help="max control steps per episode")
    p.add_argument("--randomize", action="store_true", help="enable DR, noise and pushes")
    p.add_argument("--out", help="write JSON results here")
    p.add_argument("--video", help="record the first episode to this .mp4")
    a = p.parse_args()

    overrides = {"episode_length": a.steps, "commands": {"resample_steps": 0}}
    if a.randomize:
        overrides.update({
            "randomization": "randomized",
            "noise": {"level": 1.0},
            "push": {"enable": True},
        })
    env = make_env(a.config, overrides, render_mode="rgb_array" if a.video else None)
    policy = load_policy(a.policy)

    per_command = []
    for cmd in COMMAND_SETS[a.commands]:
        full = np.zeros(7)
        full[:3] = cmd
        results = []
        for ep in range(a.episodes):
            frames = [] if (a.video and not per_command and ep == 0) else None
            results.append(rollout(env, policy, full, seed=ep, max_steps=a.steps, frames=frames))
            if frames:
                import mediapy

                mediapy.write_video(a.video, frames, fps=int(1 / env.dt))
                print(f"wrote {a.video}")
        s = summarize(results)
        s["command"] = cmd
        per_command.append(s)
        print(f"cmd {cmd}: fall {s['fall_rate']:.2f}  dur {s['duration_s']:.1f}s  "
              f"vx_err {s['vx_error']:.3f}  vy_err {s['vy_error']:.3f}  wz_err {s['yaw_rate_error']:.3f}  "
              f"tilt {s['mean_tilt_deg']:.1f}°  dist {s['distance_m']:.2f}m")

    out = {
        "policy": a.policy,
        "config": a.config,
        "randomized": a.randomize,
        "per_command": per_command,
        "overall_fall_rate": float(np.mean([s["fall_rate"] for s in per_command])),
    }
    print(f"overall fall rate {out['overall_fall_rate']:.2f}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=2))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
