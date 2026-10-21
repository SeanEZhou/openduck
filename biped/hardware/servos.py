"""Feetech STS3215 calibration transforms and bus backend (hardware bring-up, M4/M5).

Calibration convention (per joint):
    q_sim = sign * (ticks - zero_ticks) * 2*pi / ticks_per_rev
    ticks = zero_ticks + sign * q_sim * ticks_per_rev / (2*pi)

`zero_ticks` is the raw encoder reading when the joint is at q_sim = 0 in the
MJCF frame; `sign` is +1 if the servo's positive direction matches the MJCF
joint axis. Both are measured during calibration (scripts/calibrate_actuators.py)
and stored in configs/calibration.yaml, never edited by hand in code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from biped.config import RobotSpec


@dataclass
class JointCalibration:
    zero_ticks: float = 2048.0
    sign: int = 1


class ServoCalibration:
    def __init__(self, spec: RobotSpec, joints: dict[str, JointCalibration] | None = None):
        self.spec = spec
        self.ticks_per_rev = int(spec.actuator["encoder_ticks_per_rev"])
        joints = joints or {}
        self.zero = np.array([joints.get(n, JointCalibration()).zero_ticks for n in spec.joint_names])
        self.sign = np.array([joints.get(n, JointCalibration()).sign for n in spec.joint_names], dtype=float)
        if not np.all(np.isin(self.sign, (-1, 1))):
            raise ValueError("calibration signs must be +1 or -1")

    @classmethod
    def load(cls, spec: RobotSpec, path: str | Path) -> "ServoCalibration":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        joints = {n: JointCalibration(float(v["zero_ticks"]), int(v["sign"])) for n, v in raw.get("joints", {}).items()}
        unknown = set(joints) - set(spec.joint_names)
        if unknown:
            raise ValueError(f"calibration has unknown joints: {sorted(unknown)}")
        return cls(spec, joints)

    def save(self, path: str | Path) -> None:
        data = {"joints": {n: {"zero_ticks": float(z), "sign": int(s)}
                           for n, z, s in zip(self.spec.joint_names, self.zero, self.sign)}}
        Path(path).write_text(yaml.safe_dump(data, sort_keys=False))

    def ticks_to_rad(self, ticks: np.ndarray) -> np.ndarray:
        return self.sign * (np.asarray(ticks, dtype=float) - self.zero) * 2 * np.pi / self.ticks_per_rev

    def rad_to_ticks(self, q: np.ndarray) -> np.ndarray:
        return self.zero + self.sign * np.asarray(q, dtype=float) * self.ticks_per_rev / (2 * np.pi)

    def vel_ticks_to_rad(self, ticks_per_s: np.ndarray) -> np.ndarray:
        return self.sign * np.asarray(ticks_per_s, dtype=float) * 2 * np.pi / self.ticks_per_rev

    def servo_ids(self) -> list[int]:
        return [j.servo_id for j in self.spec.joints]
