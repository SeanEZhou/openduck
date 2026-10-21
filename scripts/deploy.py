"""Run an exported policy through the deployment loop (Track A step 7 / M5).

The loop only uses RobotInterface + PolicyRuntime + SafetyMonitor: no
simulator internals, no privileged state. With --backend sim this is the
deployment rehearsal; --backend hardware is wired up during M4/M5.

    python scripts/deploy.py --policy results/standing_seed0/policy.onnx --seconds 20
    python scripts/deploy.py --policy third_party/Open_Duck_Mini/BEST_WALK_ONNX.onnx --cmd 0.1 0 0 --realtime
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from biped.config import load_robot  # noqa: E402
from biped.control.reference_motion import ReferenceMotion  # noqa: E402
from biped.control.runtime import PolicyRuntime  # noqa: E402
from biped.control.safety import SafetyMonitor  # noqa: E402
from biped.policy import OnnxPolicy  # noqa: E402


def make_backend(name, spec):
    if name == "sim":
        from biped.hardware.mujoco import MujocoRobot

        robot = MujocoRobot(spec)
        robot.reset_to_home()
        return robot
    raise NotImplementedError(
        "hardware backend: implement biped/hardware/open_duck.py (Feetech bus via rustypot, "
        "BNO055 IMU, foot switches) with ServoCalibration from configs/calibration.yaml. "
        "Gate: M4 calibration complete and robot on a test stand/tether."
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True, help="ONNX file (obs[1,101] -> actions[1,14])")
    p.add_argument("--backend", default="sim", choices=["sim", "hardware"])
    p.add_argument("--cmd", type=float, nargs=3, default=[0, 0, 0])
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--realtime", action="store_true", help="pace the loop at the control rate")
    p.add_argument("--log", help="write a JSON log of states/targets")
    a = p.parse_args()

    spec = load_robot()
    ref = ReferenceMotion(spec.model_path("reference_motion"))
    robot = make_backend(a.backend, spec)
    runtime = PolicyRuntime(spec, ref.steps_per_period)
    safety = SafetyMonitor(spec)
    policy = OnnxPolicy(a.policy)
    dt = spec.control["policy_dt"]
    cmd = np.zeros(7)
    cmd[:3] = a.cmd

    runtime.reset()
    log, stop_reason = [], None
    robot.enable()
    try:
        loop_time = None
        for k in range(int(a.seconds / dt)):
            t0 = time.perf_counter()
            state = robot.read_state()
            stop_reason = safety.check(state, loop_time if a.realtime else None)
            if stop_reason:
                break
            obs = runtime.observe(state, cmd)
            targets = runtime.act(policy(obs), cmd)
            robot.write_targets(targets)
            robot.step()
            if a.log:
                log.append({"t": k * dt, "q": state.joint_pos.tolist(), "target": targets.tolist(),
                            "gyro": state.gyro.tolist(), "contact": state.feet_contact.tolist()})
            if a.realtime:
                time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
            loop_time = time.perf_counter() - t0
    finally:
        robot.disable()

    duration = (k + (stop_reason is None)) * dt
    print(f"ran {duration:.2f}s; stop reason: {stop_reason or 'completed'}")
    if a.log:
        Path(a.log).write_text(json.dumps({"stop_reason": stop_reason, "steps": log}))
    sys.exit(1 if stop_reason else 0)


if __name__ == "__main__":
    main()
