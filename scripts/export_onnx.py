"""Export an SB3 run's deterministic actor to ONNX with observation normalization baked in.

The result has the same interface as the upstream Open Duck policies
(input "obs" [1, 101] -> output "continuous_actions" [1, 14]), so it runs on
the upstream Raspberry Pi runtime as well as through biped.policy.OnnxPolicy.

    python scripts/export_onnx.py results/standing_seed0 [--checkpoint final]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from biped.policy import SB3Policy  # noqa: E402


class _Actor(torch.nn.Module):
    def __init__(self, pol: SB3Policy):
        super().__init__()
        p = pol.model.policy
        self.features = p.mlp_extractor.policy_net
        self.head = p.action_net
        rms = pol.obs_rms
        mean = rms.mean if rms is not None else 0.0
        std = np.sqrt(rms.var + pol.eps) if rms is not None else 1.0
        n = p.observation_space.shape[0]
        self.register_buffer("mean", torch.as_tensor(np.broadcast_to(mean, n), dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(np.broadcast_to(std, n), dtype=torch.float32))
        self.clip = float(pol.clip_obs) if rms is not None else 1e9
        low, high = p.action_space.low, p.action_space.high
        self.register_buffer("low", torch.as_tensor(low, dtype=torch.float32))
        self.register_buffer("high", torch.as_tensor(high, dtype=torch.float32))

    def forward(self, obs):
        z = torch.clamp((obs - self.mean) / self.std, -self.clip, self.clip)
        a = self.head(self.features(z))
        return torch.maximum(torch.minimum(a, self.high), self.low)


def export(run_dir, out, checkpoint: str = "final") -> None:
    pol = SB3Policy(run_dir, checkpoint)
    actor = _Actor(pol).eval()
    n = actor.mean.shape[0]
    torch.onnx.export(actor, torch.zeros(1, n), str(out), input_names=["obs"],
                      output_names=["continuous_actions"], opset_version=17, dynamo=False)

    # Verify against the SB3 policy.
    from biped.policy import OnnxPolicy

    onnx_pol = OnnxPolicy(out)
    x = np.random.default_rng(0).normal(size=n).astype(np.float32)
    err = np.abs(onnx_pol(x) - pol(x)).max()
    if err > 1e-4:
        raise RuntimeError(f"ONNX export mismatch {err}")
    print(f"exported {out} (max abs diff vs SB3 {err:.2e})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--checkpoint", default="final")
    p.add_argument("--out")
    a = p.parse_args()
    export(a.run_dir, a.out or Path(a.run_dir) / "policy.onnx", a.checkpoint)
