"""Side-by-side table of evaluate.py JSON results (Markdown).

    python scripts/compare.py upstream=results/baseline_BEST_WALK_ONNX_randomized.json \
        ours=results/mjx_walking_seed0/eval_walk_randomized.json
"""

import json
import sys
from pathlib import Path


def main(pairs):
    runs = {}
    for p in pairs:
        name, path = p.split("=", 1)
        runs[name] = {tuple(c["command"]): c for c in json.loads(Path(path).read_text())["per_command"]}
    names = list(runs)
    commands = list(next(iter(runs.values())))
    metrics = [("fall_rate", "fall"), ("vx_error", "vx err"), ("vy_error", "vy err"), ("yaw_rate_error", "wz err")]
    head = "| command (vx, vy, wz) | " + " | ".join(f"{label} {n}" for _, label in metrics for n in names) + " |"
    print(head)
    print("|" + "---|" * (1 + len(metrics) * len(names)))
    for c in commands:
        cells = [f"{runs[n][c][k]:.2f}" if c in runs[n] else "–" for k, _ in metrics for n in names]
        print(f"| {c} | " + " | ".join(cells) + " |")
    means = [f"{sum(runs[n][c][k] for c in commands) / len(commands):.2f}" for k, _ in metrics for n in names]
    print("| **mean** | " + " | ".join(means) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])
