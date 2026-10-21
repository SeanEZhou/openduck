"""Model-level domain randomization for the MJX env (one sample per parallel env).

Same keys and ranges as configs/randomization.yaml, interpreted exactly as
biped.envs.randomization.DomainRandomizer does on the CPU model.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jp
from mujoco import mjx

FIELDS = ("geom_friction", "body_mass", "body_ipos", "dof_damping", "dof_frictionloss", "dof_armature",
          "actuator_gainprm", "actuator_biasprm", "actuator_forcerange")


def make_randomization_fn(env, cfg: dict[str, Any]):
    """Returns randomization_fn(model, rng) -> (batched model, in_axes) for wrap_for_brax_training."""
    dofs, acts = jp.asarray(env.qvel_adr), jp.asarray(env.act_ids)
    n, floor, trunk = env.n, env.floor_geom, env.root_body

    def u(key, name, shape=()):
        lo, hi = cfg[name]
        return jax.random.uniform(key, shape, minval=lo, maxval=hi)

    def randomization_fn(model: mjx.Model, rng: jax.Array):
        @jax.vmap
        def sample(key):
            k = jax.random.split(key, 10)
            f = {name: getattr(model, name) for name in FIELDS}
            if "floor_friction" in cfg:
                f["geom_friction"] = f["geom_friction"].at[floor, 0].set(u(k[0], "floor_friction"))
            if "body_mass_scale" in cfg:
                f["body_mass"] = f["body_mass"] * u(k[1], "body_mass_scale", (model.nbody,))
            if "trunk_added_mass" in cfg:
                f["body_mass"] = f["body_mass"].at[trunk].add(u(k[2], "trunk_added_mass"))
            if "trunk_com_offset" in cfg:
                f["body_ipos"] = f["body_ipos"].at[trunk].add(u(k[3], "trunk_com_offset", (3,)))
            for i, (key_name, field) in enumerate((("joint_damping_scale", "dof_damping"),
                                                   ("joint_frictionloss_scale", "dof_frictionloss"),
                                                   ("joint_armature_scale", "dof_armature"))):
                if key_name in cfg:
                    f[field] = f[field].at[dofs].multiply(u(k[4 + i], key_name, (n,)))
            if "kp_scale" in cfg:
                kp = model.actuator_gainprm[acts, 0] * u(k[7], "kp_scale", (n,))
                f["actuator_gainprm"] = f["actuator_gainprm"].at[acts, 0].set(kp)
                f["actuator_biasprm"] = f["actuator_biasprm"].at[acts, 1].set(-kp)
            if "torque_limit_scale" in cfg:
                f["actuator_forcerange"] = f["actuator_forcerange"].at[acts].multiply(
                    u(k[8], "torque_limit_scale", (n,))[:, None])
            return f

        fields = sample(rng)
        in_axes = jax.tree_util.tree_map(lambda _: None, model).tree_replace({k: 0 for k in FIELDS})
        return model.tree_replace(fields), in_axes

    return randomization_fn
