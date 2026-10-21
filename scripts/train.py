"""Train a PPO policy (Stable-Baselines3) on a task config.

    python scripts/train.py --config configs/train_standing.yaml --run results/standing_seed0
    python scripts/train.py --config configs/train_walking.yaml --run results/walking_seed0 \
        --init results/standing_seed0      # optional warm start

Writes to the run directory: config snapshot, tensorboard logs, periodic
checkpoints (<step>_model.zip + <step>_vecnormalize.pkl), final_* and policy.onnx.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback, CallbackList  # noqa: E402
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize  # noqa: E402

from biped.config import deep_update, load_config  # noqa: E402
from biped.envs.locomotion import DuckEnv  # noqa: E402


def _env_factory(env_cfg: dict, seed: int):
    def make():
        env = DuckEnv(env_cfg)
        env.reset(seed=seed)
        return env

    return make


class TermLogger(BaseCallback):
    """Logs per-term episode returns from info["episode_terms"]."""

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            for k, v in info.get("episode_terms", {}).items():
                self.logger.record_mean(f"terms/{k}", v)
        return True


class Checkpoint(BaseCallback):
    def __init__(self, run: Path, every: int):
        super().__init__()
        self.run, self.every, self.next = run, every, every

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next:
            self.next += self.every
            tag = f"{self.num_timesteps // 1_000_000:04d}M"
            self.model.save(self.run / f"{tag}_model.zip")
            self.model.get_env().save(str(self.run / f"{tag}_vecnormalize.pkl"))
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--run", required=True, help="output directory")
    p.add_argument("--timesteps", type=float, help="override ppo.total_timesteps")
    p.add_argument("--seed", type=int, help="override ppo.seed")
    p.add_argument("--set", nargs="*", default=[], help="env overrides, e.g. randomization=nominal")
    p.add_argument("--init", help="warm start from a run directory (final_model.zip)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    cfg = load_config(a.config)
    for kv in a.set:
        key, val = kv.split("=", 1)
        node = {}
        cur = node
        parts = key.split(".")
        for part in parts[:-1]:
            cur[part] = {}
            cur = cur[part]
        cur[parts[-1]] = yaml.safe_load(val)
        cfg["env"] = deep_update(cfg["env"], node)
    ppo = cfg["ppo"]
    if a.timesteps:
        ppo["total_timesteps"] = int(a.timesteps)
    if a.seed is not None:
        ppo["seed"] = a.seed

    run = Path(a.run)
    run.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = ""
    (run / "config.yaml").write_text(yaml.safe_dump({**cfg, "source_config": a.config, "git_commit": commit or None}))
    shutil.copy("configs/robot.yaml", run / "robot.yaml")

    seed = int(ppo["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    venv = SubprocVecEnv([_env_factory(cfg["env"], seed * 1000 + i) for i in range(ppo["n_envs"])])
    venv = VecMonitor(venv)
    if a.init:
        venv = VecNormalize.load(str(Path(a.init) / "final_vecnormalize.pkl"), venv)
        venv.training = True
    else:
        venv = VecNormalize(venv, norm_obs=ppo.get("normalize_obs", True), norm_reward=True,
                            clip_obs=10.0, gamma=ppo["gamma"])

    act = {"elu": torch.nn.ELU, "tanh": torch.nn.Tanh, "relu": torch.nn.ReLU}[ppo.get("activation", "elu")]
    policy_kwargs = dict(net_arch=dict(ppo["net_arch"]), activation_fn=act, log_std_init=ppo["log_std_init"])
    common = dict(
        n_steps=ppo["n_steps"], batch_size=ppo["batch_size"], n_epochs=ppo["n_epochs"],
        learning_rate=ppo["learning_rate"], gamma=ppo["gamma"], gae_lambda=ppo["gae_lambda"],
        clip_range=ppo["clip_range"], ent_coef=ppo["ent_coef"], vf_coef=ppo["vf_coef"],
        max_grad_norm=ppo["max_grad_norm"], target_kl=ppo.get("target_kl"),
    )
    if a.init:
        model = PPO.load(Path(a.init) / "final_model.zip", env=venv, device=a.device,
                         tensorboard_log=str(run / "tb"), **common)
    else:
        model = PPO("MlpPolicy", venv, policy_kwargs=policy_kwargs, seed=seed, device=a.device,
                    tensorboard_log=str(run / "tb"), verbose=1, **common)

    every = max(int(ppo["total_timesteps"]) // 10, 1_000_000)
    model.learn(total_timesteps=int(ppo["total_timesteps"]),
                callback=CallbackList([TermLogger(), Checkpoint(run, every)]), tb_log_name="ppo")
    model.save(run / "final_model.zip")
    venv.save(str(run / "final_vecnormalize.pkl"))
    venv.close()

    from export_onnx import export

    export(run, run / "policy.onnx")
    print(f"done: {run}")


if __name__ == "__main__":
    main()
