"""Uniform `policy(obs) -> action` wrappers for ONNX files and SB3 checkpoints."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class OnnxPolicy:
    def __init__(self, path: str | Path):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        return self.session.run(None, {self.input_name: x})[0][0]


class SB3Policy:
    """Deterministic SB3 PPO policy with its VecNormalize observation statistics baked in."""

    def __init__(self, run_dir: str | Path, checkpoint: str = "final"):
        import pickle

        from stable_baselines3 import PPO

        run_dir = Path(run_dir)
        self.model = PPO.load(run_dir / f"{checkpoint}_model.zip", device="cpu")
        stats = run_dir / f"{checkpoint}_vecnormalize.pkl"
        self.obs_rms = None
        if stats.exists():
            with open(stats, "rb") as f:  # written by our own train.py
                vn = pickle.load(f)
            self.obs_rms, self.clip_obs, self.eps = vn.obs_rms, vn.clip_obs, vn.epsilon

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        if self.obs_rms is None:
            return obs
        z = (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.eps)
        return np.clip(z, -self.clip_obs, self.clip_obs)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        action, _ = self.model.predict(self.normalize(np.asarray(obs, dtype=np.float32)), deterministic=True)
        return action


def load_policy(path: str | Path):
    path = Path(path)
    if path.suffix == ".onnx":
        return OnnxPolicy(path)
    return SB3Policy(path)
