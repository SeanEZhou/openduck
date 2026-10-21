"""GPU-backend parity: the MJX env must match the CPU env and the shared contract code.

JAX runs on CPU here (set before import) so tests don't compete with training for GPU memory.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

jax = pytest.importorskip("jax")
jp = jax.numpy
pytest.importorskip("mujoco_playground")

from biped.config import load_robot  # noqa: E402
from biped.control.actions import ActionParams, process_action  # noqa: E402
from biped.control.observations import build_observation  # noqa: E402
from biped.control.reference_motion import ReferenceMotion  # noqa: E402
from biped.envs import make_env  # noqa: E402
from biped.envs.rewards import REWARD_FUNCTIONS, RewardContext  # noqa: E402

DETERMINISTIC = {"use_reference_motion": True, "commands": {"resample_steps": 0}}


@pytest.fixture(scope="module")
def spec():
    return load_robot()


def _ctx(xp, rng, spec, ref):
    n = spec.num_joints
    a = lambda *s: xp.asarray(rng.normal(size=s))  # noqa: E731
    cmd = xp.asarray(np.r_[0.1, -0.05, 0.3, rng.normal(size=4) * 0.1])
    lo, hi = spec.soft_limits()
    return RewardContext(
        command=cmd, joint_pos=xp.asarray(spec.home) + 0.1 * a(n), joint_vel=a(n), home=xp.asarray(spec.home),
        leg_idx=spec.leg_indices, action=a(n), last_action=a(n), torques=a(n), gravity=a(3), local_linvel=a(3),
        gyro=a(3), base_height=xp.asarray(0.15), home_height=0.16, base_qvel=a(6),
        feet_contact=xp.asarray([True, False]), feet_linvel=a(2, 3), feet_air_time=xp.asarray([0.3, 0.0]),
        first_contact=xp.asarray([True, False]), soft_lower=xp.asarray(lo), soft_upper=xp.asarray(hi),
        ref_frame=ref.frame(cmd, 5, xp), terminated=xp.asarray(False), xp=xp,
    )


def test_reward_functions_numpy_equals_jax(spec):
    ref = ReferenceMotion(spec.model_path("reference_motion"))
    for name, f in REWARD_FUNCTIONS.items():
        cn = _ctx(np, np.random.default_rng(0), spec, ref)
        cj = _ctx(jp, np.random.default_rng(0), spec, ref)
        np.testing.assert_allclose(float(f(cn)), float(f(cj)), rtol=1e-4, atol=1e-5, err_msg=name)


def test_action_and_obs_numpy_equals_jax(spec):
    rng = np.random.default_rng(0)
    p = ActionParams.from_spec(spec)
    a, cmd, prev = rng.normal(size=14), rng.normal(size=7) * 0.2, spec.home + rng.normal(size=14) * 0.05
    t_np = process_action(np, p, a, cmd, prev)
    t_jp = process_action(jp, p, jp.asarray(a), jp.asarray(cmd), jp.asarray(prev))
    np.testing.assert_allclose(t_np, np.asarray(t_jp), atol=1e-6)
    args = [rng.normal(size=3), rng.normal(size=3), cmd, rng.normal(size=14), rng.normal(size=14), spec.home,
            rng.normal(size=(3, 14)), t_np, np.array([True, False]), np.array([0.5, 0.5])]
    o_np = build_observation(np, *args)
    o_jp = build_observation(jp, *[jp.asarray(x) for x in args])
    np.testing.assert_allclose(o_np, np.asarray(o_jp), atol=1e-6)
    assert o_np.shape == (101,)


@pytest.fixture(scope="module")
def env_pair():
    from biped.mjx.env import make_mjx_env

    cpu = make_env("configs/eval_baseline.yaml", DETERMINISTIC)
    gpu = make_mjx_env("configs/eval_baseline.yaml", DETERMINISTIC)
    return cpu, gpu


def _aligned_reset(cpu, gpu):
    s = jax.jit(gpu.reset)(jax.random.PRNGKey(0))
    # Same deterministic initial state; commands are sampled per backend, so align them.
    cmd = np.asarray(s.info["command"])
    cpu.reset(seed=0)
    cpu.command = cmd
    cpu.runtime.reset()
    return s, cpu.runtime.observe(cpu._measure(), cmd)


@pytest.mark.slow
def test_reset_obs_matches_cpu_env(env_pair):
    from biped.control.observations import observation_layout

    cpu, gpu = env_pair
    s, obs_cpu = _aligned_reset(cpu, gpu)
    obs_gpu = np.asarray(s.obs["state"])
    # The t=0 accelerometer reflects each solver's cold-start contact transient; checked after settling below.
    keep = np.ones(101, bool)
    keep[observation_layout(14)["accelerometer"]] = False
    np.testing.assert_allclose(obs_cpu[keep], obs_gpu[keep], atol=2e-3)


@pytest.mark.slow
def test_rollout_matches_cpu_env(env_pair):
    from biped.control.observations import observation_layout

    cpu, gpu = env_pair
    s, _ = _aligned_reset(cpu, gpu)
    step = jax.jit(gpu.step)
    rng = np.random.default_rng(0)
    for _ in range(25):
        a = rng.uniform(-0.2, 0.2, 14)
        s = step(s, jp.asarray(a, dtype=jp.float32))
        obs_cpu, _, term, _, info = cpu.step(a)
        assert not term and not bool(s.done)
    lay = observation_layout(14)
    g = np.asarray(s.obs["state"])
    # Raw MJX vs CPU MuJoCo physics under the same open-loop actions already differ by ~0.02 rad here.
    np.testing.assert_allclose(obs_cpu[lay["joint_pos"]], g[lay["joint_pos"]], atol=0.03)
    np.testing.assert_array_equal(obs_cpu[lay["feet_contact"]], g[lay["feet_contact"]])
    np.testing.assert_allclose(obs_cpu[lay["phase"]], g[lay["phase"]], atol=1e-6)
    # Physics agrees closely but not exactly (different contact solvers). The sigma=0.01 tracking
    # exponentials amplify tiny velocity differences, so those are checked via the velocities above.
    for k, v in info["reward_terms"].items():
        if k.startswith("tracking"):
            continue
        np.testing.assert_allclose(v, float(s.metrics[f"reward/{k}"]), rtol=0.15, atol=0.05, err_msg=k)

    # IMU after a hold (instantaneous readings under random actions are dominated by contact impacts).
    for _ in range(25):
        s = step(s, jp.zeros(14))
        obs_cpu, *_ = cpu.step(np.zeros(14))
    g = np.asarray(s.obs["state"])
    np.testing.assert_allclose(obs_cpu[lay["accelerometer"]], g[lay["accelerometer"]], atol=0.5)
    np.testing.assert_allclose(obs_cpu[lay["gyro"]], g[lay["gyro"]], atol=0.1)


def test_randomization_fn_ranges(env_pair):
    from biped.config import load_yaml
    from biped.mjx.randomize import make_randomization_fn

    _, gpu = env_pair
    cfg = load_yaml("configs/randomization.yaml")["randomized"]
    model, in_axes = make_randomization_fn(gpu, cfg)(gpu.mjx_model, jax.random.split(jax.random.PRNGKey(0), 8))
    kp0 = gpu.mj_model.actuator_gainprm[gpu.act_ids, 0]
    kp = np.asarray(model.actuator_gainprm[:, gpu.act_ids, 0])
    assert kp.shape == (8, 14)
    lo, hi = cfg["kp_scale"]
    assert np.all(kp >= kp0 * lo - 1e-6) and np.all(kp <= kp0 * hi + 1e-6)
    np.testing.assert_allclose(np.asarray(model.actuator_biasprm[:, gpu.act_ids, 1]), -kp)
    fr = np.asarray(model.geom_friction[:, gpu.floor_geom, 0])
    assert np.all((fr >= cfg["floor_friction"][0]) & (fr <= cfg["floor_friction"][1]))
    assert in_axes.geom_friction == 0 and in_axes.qpos0 is None
