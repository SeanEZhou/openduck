import numpy as np
import pytest

from biped.envs import make_env
from biped.policy import OnnxPolicy
from biped.config import REPO_ROOT


@pytest.fixture(scope="module")
def standing_env():
    return make_env("configs/train_standing.yaml")


def test_reset_step_shapes(standing_env):
    env = standing_env
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape == (101,)
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    assert obs.shape == (101,) and np.isfinite(r)
    assert set(info["reward_terms"]) == {k for k, v in env.cfg["rewards"]["scales"].items() if v}


def test_seeded_reset_is_deterministic(standing_env):
    env = standing_env
    o1, _ = env.reset(seed=123)
    a = np.random.default_rng(0).uniform(-0.5, 0.5, (20, env.n))
    traj1 = [env.step(x)[0] for x in a]
    o2, _ = env.reset(seed=123)
    traj2 = [env.step(x)[0] for x in a]
    np.testing.assert_array_equal(o1, o2)
    np.testing.assert_array_equal(np.array(traj1), np.array(traj2))


def test_nominal_robot_stands_with_zero_action():
    env = make_env("configs/eval_baseline.yaml", {"use_reference_motion": False})
    env.reset(seed=0)
    for _ in range(250):
        _, _, term, _, _ = env.step(np.zeros(env.n))
        assert not term
    assert env.robot.feet_contact().all()


def test_randomization_restores_nominal(standing_env):
    env = standing_env
    m = env.robot.model
    kp0 = env.randomizer._nominal["actuator_gainprm"][:, 0].copy()
    env.reset(seed=1)
    assert not np.allclose(m.actuator_gainprm[:, 0], kp0)
    env.randomizer.restore()
    np.testing.assert_array_equal(m.actuator_gainprm[:, 0], kp0)


def test_head_command_moves_head_in_sim():
    env = make_env("configs/eval_baseline.yaml", {"use_reference_motion": False})
    env.reset(seed=0)
    env.command = np.array([0, 0, 0, 0, 0, 0.5, 0])
    for _ in range(50):
        env.step(np.zeros(env.n))
    i = env.spec.joint_names.index("head_yaw")
    assert env.robot.data.qpos[env.robot.qpos_adr[i]] == pytest.approx(0.5, abs=0.1)


@pytest.mark.slow
def test_upstream_onnx_policy_walks_forward():
    """Reproduces the published Open Duck baseline in our env: validates the obs/action contract."""
    path = REPO_ROOT / "third_party/Open_Duck_Mini/BEST_WALK_ONNX.onnx"
    if not path.exists():
        pytest.skip("upstream repo not cloned")
    env = make_env("configs/eval_baseline.yaml", {"episode_length": 500})
    from biped.evaluation.benchmarks import rollout

    cmd = np.zeros(7)
    cmd[0] = 0.15
    m = rollout(env, OnnxPolicy(path), cmd, max_steps=500)
    assert not m.fell
    assert m.distance_m > 0.8  # ~0.12 m/s over 10 s


def test_reset_rests_feet_on_ground(standing_env):
    env = standing_env
    for seed in range(10):
        env.reset(seed=seed)
        z = env.robot.min_foot_site_z(env.robot.data.qpos)
        assert z == pytest.approx(env.foot_z_ref, abs=1e-6)


def test_push_recovery_window_suspends_terms():
    env = make_env("configs/train_walking_v5.yaml", {"randomization": "nominal", "noise": {"level": 0.0},
                                                    "push": {"enable": False}, "commands": {"resample_steps": 0}})
    env.reset(seed=0)
    env.command = np.zeros(7)
    for _ in range(5):
        _, _, _, _, info = env.step(np.zeros(env.n))
    assert info["reward_terms"]["stand_still"] != 0.0
    env.apply_push(np.array([0.3, 0.0]))
    _, _, _, _, info = env.step(np.zeros(env.n))
    assert info["reward_terms"]["stand_still"] == 0.0
    for _ in range(env.recovery_steps + 1):
        _, _, _, _, info = env.step(np.zeros(env.n))
    assert info["reward_terms"]["stand_still"] != 0.0


def test_head_tracking_cost_zero_when_head_on_target():
    env = make_env("configs/train_walking_v5.yaml", {"randomization": "nominal", "noise": {"level": 0.0},
                                                    "push": {"enable": False}, "commands": {"resample_steps": 0}})
    env.reset(seed=0)
    env.command = np.array([0, 0, 0, 0.3, 0, 0.5, 0])
    for _ in range(60):
        _, _, _, _, info = env.step(np.zeros(env.n))
    assert info["reward_terms"]["head_tracking"] > -5.0 * 0.01  # head servos reach the commanded pose


def test_legs_only_variant():
    """Budget variant: 10 leg joints, 77-D observation, imitation indexes the legs by name.

    Without the head the upstream home pose tips backward (CoM behind the ankles), so the variant uses
    a forward-leaning home pose; it must stand passively in it like the full robot does."""
    from biped.config import load_robot
    from biped.control.reference_motion import ReferenceMotion

    spec = load_robot("configs/robot_legs.yaml")
    assert spec.num_joints == 10 and len(spec.indices("head")) == 0
    env = make_env("configs/eval_baseline.yaml", {"robot": "configs/robot_legs.yaml"})
    obs, _ = env.reset(seed=0)
    assert obs.shape == (77,) and env.action_space.shape == (10,)
    assert env.robot.model.body_subtreemass[1] == pytest.approx(1.556, abs=0.01)
    np.testing.assert_allclose(env.robot.model.key("home").ctrl, spec.home, atol=1e-6)
    for _ in range(150):
        _, _, term, _, info = env.step(np.zeros(10))
        assert not term
    assert np.isfinite(info["reward_terms"]["imitation"])
    off = env.ref_offset
    assert off[spec.joint_names.index("left_hip_pitch")] == pytest.approx(-0.1, abs=1e-6)
    assert off[spec.joint_names.index("left_knee")] == 0.0
    # Reference joints map by name: right knee in the legs robot is reference index 14.
    assert env.ref_joint_idx[spec.joint_names.index("right_knee")] == 14
    full = ReferenceMotion.joint_pos(env.ref.frame(np.array([0.1, 0, 0]), 3))
    legs = ReferenceMotion.joint_pos(env.ref.frame(np.array([0.1, 0, 0]), 3), env.ref_joint_idx)
    np.testing.assert_allclose(legs, np.delete(full, [5, 6, 7, 8]))
