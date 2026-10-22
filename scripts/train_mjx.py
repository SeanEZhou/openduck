"""GPU training (MJX + Brax PPO), then automatic validation in the CPU MuJoCo env.

    env -u LD_LIBRARY_PATH .venv/bin/python scripts/train_mjx.py \
        --config configs/train_walking.yaml --run results/mjx_walking_seed0

(`env -u LD_LIBRARY_PATH` keeps a system CUDA in /usr/local/cuda from shadowing
the CUDA libraries JAX installs in the venv.)

Writes to the run directory: config snapshot, progress.jsonl, tensorboard
logs, params_<step>.pkl + policy_<step>.onnx at each eval, final params.pkl +
policy.onnx, and validation_cpu.json (the exported ONNX policy evaluated in
the CPU MuJoCo env: a sim-to-sim transfer check).
"""

import argparse
import functools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import jax  # noqa: E402
import yaml  # noqa: E402
from brax.training.agents.ppo import networks as ppo_networks  # noqa: E402
from brax.training.agents.ppo import train as ppo  # noqa: E402
from mujoco_playground._src import wrapper  # noqa: E402

from biped.config import deep_update, load_config  # noqa: E402
from biped.mjx.env import DuckMjxEnv  # noqa: E402
from biped.mjx.export import export_onnx, save_params  # noqa: E402
from biped.mjx.randomize import make_randomization_fn  # noqa: E402

DEFAULT_PPO = dict(
    num_timesteps=150_000_000, num_envs=8192, batch_size=256, num_minibatches=32, unroll_length=20,
    num_updates_per_batch=4, learning_rate=3e-4, entropy_cost=5e-3, discounting=0.97, clipping_epsilon=0.2,
    max_grad_norm=1.0, reward_scaling=1.0, normalize_observations=True, num_evals=15, num_resets_per_eval=0,
    full_reset=True,  # False = Playground default used upstream: reset to a cached first state, keep info
    policy_hidden_layer_sizes=[512, 256, 128], value_hidden_layer_sizes=[512, 256, 128],
)


def parse_overrides(pairs):
    out = {}
    for kv in pairs:
        key, val = kv.split("=", 1)
        cur = out
        parts = key.split(".")
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = yaml.safe_load(val)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--timesteps", type=float)
    p.add_argument("--num-envs", type=int)
    p.add_argument("--num-evals", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--set", nargs="*", default=[], help="env overrides, e.g. push.enable=false")
    p.add_argument("--validate", default=None, choices=["stand", "walk"], help="CPU validation command set")
    a = p.parse_args()

    cfg = load_config(a.config)
    cfg["env"] = deep_update(cfg["env"], parse_overrides(a.set))
    hp = {**DEFAULT_PPO, **cfg.get("brax_ppo", {})}
    if a.timesteps:
        hp["num_timesteps"] = int(a.timesteps)
    if a.num_envs:
        hp["num_envs"] = a.num_envs
    if a.num_evals:
        hp["num_evals"] = a.num_evals

    run = Path(a.run)
    run.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or None
    (run / "config.yaml").write_text(yaml.safe_dump({**cfg, "brax_ppo": hp, "source_config": a.config,
                                                     "seed": a.seed, "git_commit": commit}))

    jax.config.update("jax_compilation_cache_dir", str(ROOT / ".jax_cache"))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1)

    env = DuckMjxEnv(cfg["env"])
    eval_env = DuckMjxEnv(cfg["env"])
    rand_fn = make_randomization_fn(env, env.rand_cfg) if env.rand_cfg.get("enable") else None
    obs_sizes = {k: v[0] for k, v in env.observation_size.items()}

    full_reset = bool(hp.pop("full_reset"))
    net = functools.partial(ppo_networks.make_ppo_networks,
                            policy_hidden_layer_sizes=tuple(hp.pop("policy_hidden_layer_sizes")),
                            value_hidden_layer_sizes=tuple(hp.pop("value_hidden_layer_sizes")),
                            policy_obs_key="state", value_obs_key="privileged_state")

    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(str(run / "tb"))
    log = open(run / "progress.jsonl", "a")
    t0 = time.time()

    def progress(step, metrics):
        m = {k: float(v) for k, v in metrics.items()}
        for k, v in m.items():
            writer.add_scalar(k, v, step)
        log.write(json.dumps({"step": int(step), "wall_s": time.time() - t0, **m}) + "\n")
        log.flush()
        print(f"[{time.time() - t0:7.0f}s] step {step:>11,}  eval reward {m.get('eval/episode_reward', float('nan')):8.2f}"
              f"  ep len {m.get('eval/avg_episode_length', float('nan')):6.1f}", flush=True)

    inference = {}

    def on_params(step, make_policy, params):
        save_params(params, run / f"params_{step:011d}.pkl")
        inference["make_policy"] = make_policy
        export_onnx(params, env.action_size, run / f"policy_{step:011d}.onnx")

    _, params, _ = ppo.train(
        environment=env, eval_env=eval_env, seed=a.seed, network_factory=net, randomization_fn=rand_fn,
        wrap_env_fn=functools.partial(wrapper.wrap_for_brax_training, full_reset=full_reset),
        episode_length=cfg["env"]["episode_length"], progress_fn=progress, policy_params_fn=on_params, **hp,
    )
    save_params(params, run / "params.pkl")
    err = export_onnx(params, env.action_size, run / "policy.onnx",
                      inference_fn=inference["make_policy"](params, deterministic=True), obs_sizes=obs_sizes)
    print(f"exported {run / 'policy.onnx'} (max abs diff vs Brax inference {err:.2e}); "
          f"training took {(time.time() - t0) / 60:.1f} min")

    commands = a.validate or ("walk" if cfg["env"].get("use_reference_motion") else "stand")
    subprocess.run([sys.executable, "scripts/evaluate.py", "--policy", str(run / "policy.onnx"),
                    "--config", a.config, "--commands", commands, "--episodes", "5", "--randomize",
                    "--out", str(run / "validation_cpu.json")], check=True)


if __name__ == "__main__":
    main()
