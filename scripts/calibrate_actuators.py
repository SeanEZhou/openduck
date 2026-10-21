"""Actuator characterization: step and chirp responses of one joint (Track B, steps 3 and 6).

Runs on any RobotInterface backend. In sim it produces the *model's* response
for the same test, so hardware logs can be compared like-for-like and the
MJCF actuator parameters (kp, damping, frictionloss, armature) refit.

    python scripts/calibrate_actuators.py --joint left_knee --out results/act_sim_left_knee.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from biped.config import load_robot  # noqa: E402
from biped.hardware.mujoco import MujocoRobot  # noqa: E402


def fixed_base_step(robot: MujocoRobot, pin: np.ndarray) -> None:
    robot.step()
    a, v = robot.base_qpos_adr, robot.base_qvel_adr
    robot.data.qpos[a : a + 7] = pin
    robot.data.qvel[v : v + 6] = 0


def run(robot, spec, joint: str, amplitude: float, seconds: float):
    i = spec.joint_names.index(joint)
    dt = spec.control["policy_dt"]
    lo, hi = spec.soft_limits()
    # Step away from the nearer limit so the test target stays reachable.
    if spec.home[i] + amplitude > hi[i]:
        amplitude = -amplitude
    amplitude = float(np.clip(spec.home[i] + amplitude, lo[i], hi[i]) - spec.home[i])
    robot.reset_to_home()
    pin = robot.data.qpos[robot.base_qpos_adr : robot.base_qpos_adr + 7].copy()
    pin[2] += 0.15  # suspended, like a test stand
    n = int(seconds / dt)
    t = np.arange(n) * dt
    tests = {
        "step": np.where(t > 0.2, amplitude, 0.0),
        "chirp": amplitude * np.sin(2 * np.pi * (0.2 * t + 0.5 * (5.0 - 0.2) / seconds * t**2)),
    }
    out = {}
    for name, offset in tests.items():
        robot.reset_to_home()
        target, meas = [], []
        for k in range(n):
            q = spec.home.copy()
            q[i] += offset[k]
            robot.write_targets(q)
            fixed_base_step(robot, pin)
            target.append(q[i])
            meas.append(robot.read_state().joint_pos[i])
        target, meas = np.array(target), np.array(meas)
        rec = {"t": t.tolist(), "target": target.tolist(), "measured": meas.tolist()}
        if name == "step":
            rel = (meas - spec.home[i]) / amplitude
            k0 = int(0.2 / dt)
            rise = np.argmax(rel[k0:] > 0.9) * dt if np.any(rel[k0:] > 0.9) else float("nan")
            rec.update(rise_time_90=rise, steady_state_error=float(target[-1] - meas[-1]),
                       overshoot=float(max(rel.max() - 1.0, 0.0)))
        else:
            # Lag estimate via cross-correlation.
            x, y = target - target.mean(), meas - meas.mean()
            lags = np.arange(0, 15)
            c = [np.dot(x[: n - L], y[L:]) for L in lags]
            rec.update(lag_s=float(lags[int(np.argmax(c))] * dt),
                       rms_error=float(np.sqrt(np.mean((target - meas) ** 2))))
        out[name] = rec
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--joint", default="left_knee")
    p.add_argument("--amplitude", type=float, default=0.3)
    p.add_argument("--seconds", type=float, default=4.0)
    p.add_argument("--out")
    a = p.parse_args()
    spec = load_robot()
    res = run(MujocoRobot(spec), spec, a.joint, a.amplitude, a.seconds)
    s, c = res["step"], res["chirp"]
    print(f"{a.joint}: step rise(90%) {s['rise_time_90'] * 1e3:.0f} ms, ss error {s['steady_state_error']:.4f} rad, "
          f"overshoot {s['overshoot'] * 100:.0f}%; chirp lag {c['lag_s'] * 1e3:.0f} ms, rms {c['rms_error']:.3f} rad")
    if a.out:
        Path(a.out).write_text(json.dumps({"joint": a.joint, "backend": "sim", **res}))


if __name__ == "__main__":
    main()
