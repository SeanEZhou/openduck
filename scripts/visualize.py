"""Interactive MuJoCo viewer: manual joint control or policy playback.

Manual joint control (use the viewer's Control panel sliders, one per actuator):
    python scripts/visualize.py joints [--fixed-base]

Policy playback with a fixed command (vx vy wz):
    python scripts/visualize.py policy --policy results/standing_seed0 --cmd 0 0 0
    python scripts/visualize.py policy --policy third_party/Open_Duck_Mini/BEST_WALK_ONNX.onnx --cmd 0.15 0 0

Headless (no display, e.g. WSL without WSLg): add --video out.mp4 --seconds 10.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mujoco  # noqa: E402

from biped.config import load_robot  # noqa: E402
from biped.envs import make_env  # noqa: E402
from biped.hardware.mujoco import MujocoRobot  # noqa: E402


def joints(a):
    spec = load_robot()
    robot = MujocoRobot(spec)
    robot.reset_to_home()
    m, d = robot.model, robot.data
    pin_q = d.qpos[robot.base_qpos_adr : robot.base_qpos_adr + 7].copy()
    pin_q[2] += 0.1 if a.fixed_base else 0.0

    import mujoco.viewer

    print("Joint order:", ", ".join(f"{i}:{n}" for i, n in enumerate(spec.joint_names)))
    print("Drag the sliders in the Control panel (right side) to move each joint.")
    with mujoco.viewer.launch_passive(m, d) as v:
        while v.is_running():
            t = time.time()
            mujoco.mj_step(m, d, nstep=robot.n_substeps)
            if a.fixed_base:
                d.qpos[robot.base_qpos_adr : robot.base_qpos_adr + 7] = pin_q
                d.qvel[robot.base_qvel_adr : robot.base_qvel_adr + 6] = 0
                mujoco.mj_forward(m, d)
            v.sync()
            time.sleep(max(0.0, spec.control["policy_dt"] - (time.time() - t)))


def policy(a):
    from biped.policy import load_policy

    env = make_env(a.config, {"episode_length": 10**9}, render_mode="rgb_array" if a.video else None)
    pol = load_policy(a.policy)
    cmd = np.zeros(7)
    cmd[:3] = a.cmd
    obs, _ = env.reset(seed=a.seed)
    env.command = cmd

    if a.video:
        import mediapy

        frames = []
        for _ in range(int(a.seconds / env.dt)):
            obs, _, term, _, _ = env.step(pol(obs))
            frames.append(env.render())
            if term:
                print("fell")
                break
        mediapy.write_video(a.video, frames, fps=int(1 / env.dt))
        print(f"wrote {a.video}")
        return

    import mujoco.viewer

    with mujoco.viewer.launch_passive(env.robot.model, env.robot.data) as v:
        while v.is_running():
            t = time.time()
            obs, _, term, _, _ = env.step(pol(obs))
            if term:
                obs, _ = env.reset()
                env.command = cmd
            v.sync()
            time.sleep(max(0.0, env.dt - (time.time() - t)))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)
    j = sub.add_parser("joints")
    j.add_argument("--fixed-base", action="store_true", help="hold the torso in the air")
    q = sub.add_parser("policy")
    q.add_argument("--policy", required=True)
    q.add_argument("--config", default="configs/eval_baseline.yaml")
    q.add_argument("--cmd", type=float, nargs=3, default=[0, 0, 0])
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--video")
    q.add_argument("--seconds", type=float, default=10.0)
    a = p.parse_args()
    {"joints": joints, "policy": policy}[a.mode](a)


if __name__ == "__main__":
    main()
