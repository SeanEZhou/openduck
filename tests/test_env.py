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
