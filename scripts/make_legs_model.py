"""Generate the budget "legs-only" Open Duck variant from the upstream MJCF.

Removes the neck/head subtree (4 STS3215 servos, head shell, antennas, expression parts:
~582 g) and adds a small rigid mass on the trunk for the relocated Raspberry Pi Zero 2W,
SD card and wiring. The legs, trunk, battery, IMU and servo board are unchanged, so the
10 leg joints keep the upstream geometry, actuator model and joint conventions.

    python scripts/make_legs_model.py   # writes assets/open_duck_mini_v2_legs/scene_flat_terrain.xml
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
HEAD_JOINTS = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")


def build(electronics_mass: float, electronics_pos, hip_lean: float, ankle_offset: float) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(ROOT / "assets/open_duck_mini_v2/scene_flat_terrain.xml"))
    full = spec.compile()
    keep_q = [i for i in range(full.njnt) if full.joint(i).name not in HEAD_JOINTS]
    key = full.key("home")
    qpos = np.concatenate([key.qpos[full.jnt_qposadr[j] : full.jnt_qposadr[j] + (7 if j == 0 else 1)] for j in keep_q])
    ctrl = np.array([key.ctrl[a] for a in range(full.nu) if full.actuator(a).name not in HEAD_JOINTS])

    # Re-balanced home pose: without the head the CoM sits behind the ankles and the upstream home
    # pose tips backward. Lean the trunk forward at the hips (left/right hip pitch axes are mirrored)
    # and compensate at the ankles to keep the feet flat.
    leg_names = [full.actuator(a).name for a in range(full.nu) if full.actuator(a).name not in HEAD_JOINTS]
    delta = np.zeros(len(leg_names))
    delta[leg_names.index("left_hip_pitch")] = -hip_lean
    delta[leg_names.index("right_hip_pitch")] = +hip_lean
    delta[leg_names.index("left_ankle")] = ankle_offset
    delta[leg_names.index("right_ankle")] = ankle_offset
    qpos[7:] += delta
    ctrl += delta

    for a in [a for a in spec.actuators if a.name in HEAD_JOINTS]:
        spec.delete(a)
    spec.delete(spec.body("neck_pitch_assembly"))

    trunk = spec.body("trunk_assembly")
    pi = trunk.add_body(name="relocated_electronics", pos=list(electronics_pos))
    pi.explicitinertial = True
    pi.mass = electronics_mass
    pi.ipos = [0, 0, 0]
    pi.inertia = [1e-5, 1e-5, 1e-5]

    k = spec.keys[0]
    k.qpos = qpos.tolist()
    k.ctrl = ctrl.tolist()
    spec.meshdir = "../open_duck_mini_v2/assets"
    return spec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--electronics-mass", type=float, default=0.03, help="kg: Pi Zero 2W + SD + wiring + mount")
    p.add_argument("--hip-lean", type=float, default=0.1, help="rad, forward trunk lean at the hips")
    p.add_argument("--ankle-offset", type=float, default=0.1, help="rad, ankle compensation for flat feet")
    p.add_argument("--out", default="assets/open_duck_mini_v2_legs/scene_flat_terrain.xml")
    a = p.parse_args()
    # Where the neck mounted on the trunk top: keeps the electronics near the original head base.
    spec = build(a.electronics_mass, (0.0, 0.0, 0.09), a.hip_lean, a.ankle_offset)
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(spec.to_xml())
    m = mujoco.MjModel.from_xml_path(str(out))
    print(f"wrote {out}: nu={m.nu}, total mass {m.body_subtreemass[1]:.3f} kg, "
          f"actuators {[m.actuator(i).name for i in range(m.nu)]}")
    print("home pose (copy into configs/robot_legs.yaml):",
          dict(zip([m.actuator(i).name for i in range(m.nu)], np.round(m.key("home").ctrl, 3))))


if __name__ == "__main__":
    main()
