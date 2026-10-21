"""The MJCF must agree with configs/robot.yaml: joint order, limits, home pose, symmetry."""

import itertools

import mujoco
import numpy as np
import pytest

from biped.config import load_robot
from biped.hardware.mujoco import MujocoRobot


@pytest.fixture(scope="module")
def spec():
    return load_robot()


@pytest.fixture(scope="module")
def robot(spec):
    return MujocoRobot(spec)


def test_actuator_order_matches_config(spec, robot):
    m = robot.model
    assert m.nu == spec.num_joints
    assert [m.actuator(i).name for i in range(m.nu)] == spec.joint_names


def test_each_actuator_drives_its_named_joint(spec, robot):
    m = robot.model
    for name, aid in zip(spec.joint_names, robot.actuator_ids):
        assert m.joint(m.actuator_trnid[aid, 0]).name == name


def test_joint_limits_match_config(spec, robot):
    m = robot.model
    for j in spec.joints:
        lo, hi = m.jnt_range[m.joint(j.name).id]
        assert lo == pytest.approx(j.range[0], abs=1e-3)
        assert hi == pytest.approx(j.range[1], abs=1e-3)


def test_home_pose_matches_keyframe(spec, robot):
    robot.reset_to_home()
    np.testing.assert_allclose(robot.data.qpos[robot.qpos_adr], spec.home, atol=1e-6)
    np.testing.assert_allclose(robot.data.ctrl[robot.actuator_ids], spec.home, atol=1e-6)
    assert np.all(spec.home >= spec.lower) and np.all(spec.home <= spec.upper)


def test_actuator_params_match_config(spec, robot):
    m, a = robot.model, spec.actuator
    for aid, dof in zip(robot.actuator_ids, robot.qvel_adr):
        assert m.actuator_gainprm[aid, 0] == pytest.approx(a["kp"])
        assert m.actuator_forcerange[aid, 1] == pytest.approx(a["forcerange"])
        assert m.dof_damping[dof] == pytest.approx(a["damping"])
        assert m.dof_frictionloss[dof] == pytest.approx(a["frictionloss"])
        assert m.dof_armature[dof] == pytest.approx(a["armature"])


def test_servo_ids_unique(spec):
    ids = [j.servo_id for j in spec.joints]
    assert len(set(ids)) == len(ids)


def _mirror_error(robot, spec, signs, n=30, seed=0):
    m, d = robot.model, robot.data
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n):
        d.qpos[:] = 0
        d.qpos[robot.base_qpos_adr + 3] = 1
        for (left, right, _), s in zip(spec.mirror, signs):
            lo, hi = m.jnt_range[m.joint(left).id]
            v = rng.uniform(lo / 2, hi / 2)
            d.qpos[m.joint(left).qposadr[0]] = v
            d.qpos[m.joint(right).qposadr[0]] = s * v
        mujoco.mj_kinematics(m, d)
        pl, pr = d.site("left_foot").xpos, d.site("right_foot").xpos
        worst = max(worst, np.abs(pl - pr * np.array([1, -1, 1])).max())
    return worst


def test_mirror_signs_give_symmetric_feet(spec, robot):
    signs = [s for _, _, s in spec.mirror]
    # 2.4 mm residual exists even at q=0 (CAD foot site offset).
    assert _mirror_error(robot, spec, signs) < 3e-3


def test_mirror_signs_are_unique(spec, robot):
    signs = tuple(s for _, _, s in spec.mirror)
    for other in itertools.product((1, -1), repeat=len(signs)):
        if other != signs:
            assert _mirror_error(robot, spec, other, n=10) > 1e-2


def test_positive_joint_motion_directions(spec, robot):
    """Sign conventions the policy and calibration depend on (z-up, x-forward)."""
    m, d = robot.model, robot.data

    def foot(q_overrides):
        robot.reset_to_home()
        q = spec.home.copy()
        for name, dq in q_overrides.items():
            q[spec.joint_names.index(name)] += dq
        robot.set_joint_pos(q)
        mujoco.mj_kinematics(m, d)
        return d.site("left_foot").xpos.copy(), d.site("right_foot").xpos.copy()

    l0, r0 = foot({})
    # +x is forward. Left hip pitch positive swings the left foot forward; right is mirrored.
    l1, _ = foot({"left_hip_pitch": 0.2})
    _, r1 = foot({"right_hip_pitch": -0.2})
    assert l1[0] > l0[0] + 0.01
    assert r1[0] > r0[0] + 0.01
    # Hip roll moves the feet laterally in mirrored directions.
    l2, _ = foot({"left_hip_roll": 0.2})
    _, r2 = foot({"right_hip_roll": -0.2})
    assert np.sign(l2[1] - l0[1]) == -np.sign(r2[1] - r0[1])


def test_only_feet_collide_with_floor(spec, robot):
    m = robot.model
    colliding = [m.geom(i).name for i in range(m.ngeom)
                 if (m.geom_contype[i] & m.geom_conaffinity[robot.floor_geom])
                 or (m.geom_conaffinity[i] & m.geom_contype[robot.floor_geom])]
    colliding = [g for g in colliding if g != spec.model["floor_geom"]]
    assert sorted(colliding) == sorted(spec.model["foot_geoms"])
