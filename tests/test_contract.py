"""Observation / action contract shared by sim and hardware."""

import numpy as np
import pytest

from biped.config import load_robot
from biped.control.actions import ActionProcessor
from biped.control.observations import ObservationBuilder, observation_layout
from biped.control.reference_motion import ReferenceMotion
from biped.control.runtime import PolicyRuntime
from biped.control.safety import SafetyMonitor
from biped.hardware.interface import RobotState
from biped.hardware.servos import JointCalibration, ServoCalibration


@pytest.fixture(scope="module")
def spec():
    return load_robot()


def _state(spec, **kw):
    n = spec.num_joints
    s = dict(joint_pos=spec.home.copy(), joint_vel=np.zeros(n), gyro=np.zeros(3),
             accelerometer=np.array([0, 0, 9.81]), gravity=np.array([0, 0, -1.0]),
             feet_contact=np.array([True, True]), timestamp=0.0)
    s.update(kw)
    return RobotState(**s)


def test_obs_dim_matches_upstream_onnx(spec):
    assert ObservationBuilder(spec).dim == 101


def test_obs_layout_contents(spec):
    n = spec.num_joints
    lay = observation_layout(n)
    b = ObservationBuilder(spec)
    q = spec.home + np.arange(n) * 0.01
    qd = np.arange(n, dtype=float)
    hist = np.stack([np.full(n, 1.0), np.full(n, 2.0), np.full(n, 3.0)])
    cmd = np.arange(7, dtype=float)
    obs = b(_state(spec, joint_pos=q, joint_vel=qd, gyro=np.array([1, 2, 3.0])), cmd, hist,
            spec.home, np.array([0.5, -0.5]))
    np.testing.assert_allclose(obs[lay["gyro"]], [1, 2, 3])
    np.testing.assert_allclose(obs[lay["command"]], cmd)
    np.testing.assert_allclose(obs[lay["joint_pos"]], np.arange(n) * 0.01, atol=1e-6)
    np.testing.assert_allclose(obs[lay["joint_vel"]], qd * 0.05, atol=1e-6)
    np.testing.assert_allclose(obs[lay["last_action"]], 1.0)
    np.testing.assert_allclose(obs[lay["last_last_last_action"]], 3.0)
    np.testing.assert_allclose(obs[lay["motor_targets"]], spec.home, atol=1e-6)
    np.testing.assert_allclose(obs[lay["phase"]], [0.5, -0.5])


def test_zero_action_targets_home(spec):
    ap = ActionProcessor(spec)
    np.testing.assert_allclose(ap(np.zeros(spec.num_joints), np.zeros(7)), spec.home)


def test_action_scale_and_order(spec):
    ap = ActionProcessor(spec, rate_limit=False, clip_to_soft_limits=False)
    i = spec.joint_names.index("left_knee")
    a = np.zeros(spec.num_joints)
    a[i] = 0.4
    t = ap(a)
    assert t[i] == pytest.approx(spec.home[i] + 0.1)
    assert np.count_nonzero(t - spec.home) == 1


def test_rate_limit(spec):
    ap = ActionProcessor(spec, clip_to_soft_limits=False)
    t = ap(np.full(spec.num_joints, 4.0))
    max_step = spec.actuator["max_velocity"] * spec.control["policy_dt"]
    np.testing.assert_allclose(t - spec.home, max_step)


def test_soft_limit_clipping(spec):
    ap = ActionProcessor(spec, rate_limit=False)
    lo, hi = spec.soft_limits()
    t = ap(np.full(spec.num_joints, 4.0))
    assert np.all(t <= hi + 1e-12) and np.all(t >= lo - 1e-12)


def test_head_command_offsets_head_only(spec):
    ap = ActionProcessor(spec, rate_limit=False, clip_to_soft_limits=False)
    cmd = np.array([0, 0, 0, 0.1, 0.2, 0.3, -0.1])
    t = ap(np.zeros(spec.num_joints), cmd)
    head = spec.indices("head")
    np.testing.assert_allclose(t[head] - spec.home[head], cmd[3:])
    np.testing.assert_allclose(np.delete(t, head), np.delete(spec.home, head))


def test_runtime_history_follows_hardware_convention(spec):
    """Obs used to pick a[t+1] contains a[t] as last_action (as the upstream runtime does)."""
    rt = PolicyRuntime(spec, steps_per_period=27)
    rt.reset()
    lay = observation_layout(spec.num_joints)
    s = _state(spec)
    rt.observe(s, np.zeros(7))
    a1 = np.full(spec.num_joints, 0.1)
    rt.act(a1, np.zeros(7))
    obs = rt.observe(s, np.zeros(7))
    np.testing.assert_allclose(obs[lay["last_action"]], a1, atol=1e-6)
    np.testing.assert_allclose(obs[lay["last_last_action"]], 0.0)


def test_gait_phase(spec):
    rt = PolicyRuntime(spec, steps_per_period=4)
    rt.reset()
    lay = observation_layout(spec.num_joints)
    s = _state(spec)
    phases = [rt.observe(s, np.zeros(7))[lay["phase"]] for _ in range(6)]
    np.testing.assert_allclose(phases[0], [0, 0])
    np.testing.assert_allclose(phases[1], [0, 1], atol=1e-6)   # i=1 of 4 -> 90 deg
    np.testing.assert_allclose(phases[5], phases[1], atol=1e-6)


def test_mirror_permutation_is_involution(spec):
    perm, sign = spec.mirror_permutation()
    q = np.random.default_rng(0).normal(size=spec.num_joints)
    np.testing.assert_allclose((sign * (sign * q[perm])[perm]), q)
    mirrored_home = sign * spec.home[perm]
    legs = spec.leg_indices
    np.testing.assert_allclose(mirrored_home[legs], spec.home[legs], atol=0.02)


def test_reference_motion(spec):
    ref = ReferenceMotion(spec.model_path("reference_motion"))
    assert ref.steps_per_period == 27
    f = ref.frame(np.array([0.1, 0, 0]), 0)
    assert f.shape == (40,)
    q = ReferenceMotion.joint_pos(f)
    assert q.shape == (spec.num_joints,)
    # Reference gait stays near the nominal leg pose.
    legs = spec.leg_indices
    assert np.abs(q[legs] - spec.home[legs]).max() < 0.6


def test_safety_monitor(spec):
    mon = SafetyMonitor(spec)
    assert mon.check(_state(spec)) is None
    tilted = _state(spec, gravity=np.array([0.9, 0, -0.43]))
    assert "tilt" in mon.check(tilted)
    q = spec.home.copy()
    q[0] = spec.upper[0] + 0.2
    assert "joint limit" in mon.check(_state(spec, joint_pos=q))
    q = spec.home.copy()
    q[3] = np.nan
    assert "non-finite" in mon.check(_state(spec, joint_pos=q))
    assert "overrun" in mon.check(_state(spec), loop_time=0.05)


def test_servo_calibration_roundtrip(spec, tmp_path):
    cal = ServoCalibration(spec, {"left_knee": JointCalibration(1900.0, -1)})
    q = np.random.default_rng(1).uniform(spec.lower, spec.upper)
    np.testing.assert_allclose(cal.ticks_to_rad(cal.rad_to_ticks(q)), q)
    i = spec.joint_names.index("left_knee")
    assert cal.ticks_to_rad(cal.rad_to_ticks(q))[i] == pytest.approx(q[i])
    assert cal.rad_to_ticks(np.zeros(spec.num_joints))[i] == 1900.0
    t = cal.rad_to_ticks(np.full(spec.num_joints, 0.1))
    assert t[i] < 1900.0  # negative sign
    cal.save(tmp_path / "cal.yaml")
    cal2 = ServoCalibration.load(spec, tmp_path / "cal.yaml")
    np.testing.assert_allclose(cal2.zero, cal.zero)
    np.testing.assert_allclose(cal2.sign, cal.sign)
