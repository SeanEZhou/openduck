"""Evaluate a trained policy in the MJX env (batched), to measure the MJX -> CPU MuJoCo gap.

Uses the run's params.pkl with Brax's deterministic inference function (the exported
policy.onnx is verified against exactly this at export time).

    env -u LD_LIBRARY_PATH .venv/bin/python scripts/evaluate_mjx.py --run results/mjx_standing_seed0 \
        --config configs/train_standing.yaml --set push.enable=false noise.level=0 randomization=nominal
"""

import argparse
import functools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import jax  # noqa: E402
import jax.numpy as jp  # noqa: E402
import numpy as np  # noqa: E402
from brax.training.acme import running_statistics  # noqa: E402
from brax.training.agents.ppo import networks as ppo_networks  # noqa: E402

from biped.config import deep_update, load_config  # noqa: E402
from biped.mjx.env import DuckMjxEnv  # noqa: E402
from biped.mjx.export import load_params  # noqa: E402
from biped.mjx.randomize import make_randomization_fn  # noqa: E402
from train_mjx import parse_overrides  # noqa: E402


def make_policy(run: Path, obs_sizes, action_size):
    cfg = load_config(run / "config.yaml")
    hp = cfg["brax_ppo"]
    nets = ppo_networks.make_ppo_networks(
        obs_sizes, action_size, preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=tuple(hp.get("policy_hidden_layer_sizes", (512, 256, 128))),
        value_hidden_layer_sizes=tuple(hp.get("value_hidden_layer_sizes", (512, 256, 128))),
        policy_obs_key="state", value_obs_key="privileged_state")
    params = load_params(run / "params.pkl")
    return ppo_networks.make_inference_fn(nets)(params, deterministic=True)


def evaluate(env: DuckMjxEnv, policy, n_envs: int, steps: int, command=None, seed: int = 0, rand_cfg=None):
    keys = jax.random.split(jax.random.PRNGKey(seed), n_envs)
    if rand_cfg and rand_cfg.get("enable"):
        model, in_axes = make_randomization_fn(env, rand_cfg)(env.mjx_model, keys)
    else:
        model, in_axes = env.mjx_model, None

    def with_model(m, fn, *args):
        env.unwrapped._mjx_model = m
        return fn(*args)

    def reset(m, k):
        s = with_model(m, env.reset, k)
        if command is not None:
            s.info["command"] = jp.asarray(command, dtype=jp.float32)
        return s

    def step(m, s):
        a, _ = policy(s.obs, jax.random.PRNGKey(0))
        return with_model(m, env.step, s, a)

    axes = in_axes if in_axes is not None else None
    v_reset = jax.jit(jax.vmap(reset, in_axes=(axes, 0)))
    v_step = jax.vmap(step, in_axes=(axes, 0))

    @jax.jit
    def run(s):
        def body(carry, _):
            s, alive, t = carry
            s = v_step(model, s)
            alive = alive & (s.done < 0.5)
            return (s, alive, t + alive), None
        (s, alive, t), _ = jax.lax.scan(body, (s, jp.ones(n_envs, bool), jp.zeros(n_envs)), None, length=steps)
        return alive, t

    alive, t = run(v_reset(model, keys))
    return {"fall_rate": float(1 - alive.mean()), "mean_duration_s": float(t.mean() * env.dt), "n_envs": n_envs}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--cmd", type=float, nargs=3)
    p.add_argument("--n-envs", type=int, default=256)
    p.add_argument("--steps", type=int, default=750)
    a = p.parse_args()
    cfg = load_config(a.config)["env"]
    cfg = deep_update(cfg, {"commands": {"resample_steps": 0}, **parse_overrides(a.set)})
    env = DuckMjxEnv(cfg)
    policy = make_policy(Path(a.run), {k: v for k, v in env.observation_size.items()}, env.action_size)
    cmd = None if a.cmd is None else np.r_[a.cmd, 0, 0, 0, 0]
    res = evaluate(env, policy, a.n_envs, a.steps, cmd, rand_cfg=env.rand_cfg)
    print(json.dumps({"overrides": a.set, "cmd": a.cmd, **res}))


if __name__ == "__main__":
    main()
