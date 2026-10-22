"""Typed access to the versioned YAML configs in configs/."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = REPO_ROOT / path
    with open(path) as f:
        return yaml.safe_load(f)


def deep_update(base: dict, override: dict) -> dict:
    """Recursively merge `override` into a copy of `base`."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, resolving an optional `base:` file it inherits from."""
    cfg = load_yaml(path)
    base = cfg.pop("base", None)
    return deep_update(load_config(base), cfg) if base else cfg


@dataclass(frozen=True)
class JointSpec:
    name: str
    group: str
    servo_id: int
    home: float
    range: tuple[float, float]


@dataclass(frozen=True)
class RobotSpec:
    name: str
    joints: tuple[JointSpec, ...]
    mirror: tuple[tuple[str, str, int], ...]
    model: dict[str, Any]
    actuator: dict[str, Any]
    control: dict[str, Any]
    source: dict[str, Any] = field(default_factory=dict)

    @property
    def joint_names(self) -> list[str]:
        return [j.name for j in self.joints]

    @property
    def num_joints(self) -> int:
        return len(self.joints)

    @property
    def home(self) -> np.ndarray:
        return np.array([j.home for j in self.joints])

    @property
    def lower(self) -> np.ndarray:
        return np.array([j.range[0] for j in self.joints])

    @property
    def upper(self) -> np.ndarray:
        return np.array([j.range[1] for j in self.joints])

    def soft_limits(self) -> tuple[np.ndarray, np.ndarray]:
        c = (self.lower + self.upper) / 2
        r = (self.upper - self.lower) * self.control["soft_limit_factor"] / 2
        return c - r, c + r

    def indices(self, group: str) -> np.ndarray:
        return np.array([i for i, j in enumerate(self.joints) if j.group == group], dtype=int)

    @property
    def leg_indices(self) -> np.ndarray:
        return np.array([i for i, j in enumerate(self.joints) if j.group != "head"])

    def model_path(self, key: str = "scene") -> Path:
        return REPO_ROOT / self.model[key]

    def mirror_permutation(self) -> tuple[np.ndarray, np.ndarray]:
        """(perm, sign) such that mirrored_q = sign * q[perm] (head joints map to themselves)."""
        idx = {n: i for i, n in enumerate(self.joint_names)}
        perm = np.arange(self.num_joints)
        sign = np.ones(self.num_joints)
        for left, right, s in self.mirror:
            perm[idx[left]], perm[idx[right]] = idx[right], idx[left]
            sign[idx[left]] = sign[idx[right]] = s
        # Head yaw and roll flip sign under a sagittal mirror; pitch joints do not.
        for n in ("head_yaw", "head_roll"):
            if n in idx:
                sign[idx[n]] = -1
        return perm, sign


def load_robot(path: str | Path = CONFIG_DIR / "robot.yaml") -> RobotSpec:
    raw = load_yaml(path)
    joints = tuple(
        JointSpec(j["name"], j["group"], int(j["servo_id"]), float(j["home"]), tuple(j["range"]))
        for j in raw["joints"]
    )
    mirror = tuple((m["left"], m["right"], int(m["sign"])) for m in raw.get("mirror", []))
    return RobotSpec(
        name=raw["name"],
        joints=joints,
        mirror=mirror,
        model=raw["model"],
        actuator=raw["actuator"],
        control=raw["control"],
        source=raw.get("source", {}),
    )
