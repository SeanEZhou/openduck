"""Export Brax PPO policy params to ONNX (obs[1,101] -> continuous_actions[1,14]).

Graph: z = (obs - mean) / std  ->  MLP (swish hidden, linear output)
       ->  tanh(first action_size outputs)
This is Brax's deterministic `NormalTanhDistribution.mode`; `export_onnx`
verifies the graph against Brax's own inference function.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _np_tree(x):
    import jax

    return jax.tree_util.tree_map(lambda a: np.asarray(a), x)


def save_params(params, path: str | Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(_np_tree(params), f)


def load_params(path: str | Path):
    with open(path, "rb") as f:  # written by our own save_params
        return pickle.load(f)


def _policy_layers(policy_params) -> list[tuple[np.ndarray, np.ndarray]]:
    p = policy_params["params"]
    names = sorted(p, key=lambda k: int(k.split("_")[-1]))
    return [(np.asarray(p[k]["kernel"], np.float32), np.asarray(p[k]["bias"], np.float32)) for k in names]


def build_onnx(params, action_size: int, obs_key: str = "state") -> onnx.ModelProto:
    normalizer, policy = params[0], params[1]
    mean = np.asarray(normalizer.mean[obs_key], np.float32)
    std = np.asarray(normalizer.std[obs_key], np.float32)
    layers = _policy_layers(policy)
    n_obs = mean.shape[0]

    inits = [numpy_helper.from_array(mean, "mean"), numpy_helper.from_array(std, "std")]
    nodes = [helper.make_node("Sub", ["obs", "mean"], ["c"]), helper.make_node("Div", ["c", "std"], ["h0"])]
    x = "h0"
    for i, (w, b) in enumerate(layers):
        inits += [numpy_helper.from_array(w, f"w{i}"), numpy_helper.from_array(b, f"b{i}")]
        y = f"lin{i}"
        nodes.append(helper.make_node("Gemm", [x, f"w{i}", f"b{i}"], [y]))
        if i < len(layers) - 1:
            nodes += [helper.make_node("Sigmoid", [y], [f"sig{i}"]), helper.make_node("Mul", [y, f"sig{i}"], [f"h{i + 1}"])]
            x = f"h{i + 1}"
        else:
            x = y
    inits += [numpy_helper.from_array(np.array([0], np.int64), "s0"),
              numpy_helper.from_array(np.array([action_size], np.int64), "s1"),
              numpy_helper.from_array(np.array([1], np.int64), "ax")]
    nodes += [helper.make_node("Slice", [x, "s0", "s1", "ax"], ["loc"]),
              helper.make_node("Tanh", ["loc"], ["continuous_actions"])]
    graph = helper.make_graph(
        nodes, "biped_policy",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, n_obs])],
        [helper.make_tensor_value_info("continuous_actions", TensorProto.FLOAT, [1, action_size])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def export_onnx(params, action_size: int, out: str | Path, inference_fn=None, obs_sizes=None) -> float:
    """Write the ONNX file; if `inference_fn` (Brax make_policy) is given, return max abs error vs it."""
    onnx.save(build_onnx(params, action_size), str(out))
    if inference_fn is None:
        return float("nan")
    import jax
    import jax.numpy as jp

    from biped.policy import OnnxPolicy

    pol = OnnxPolicy(out)
    rng = np.random.default_rng(0)
    err = 0.0
    for _ in range(5):
        # Sample from the training observation distribution (raw N(0,1) saturates tanh and hides errors).
        obs = {k: (np.asarray(params[0].mean[k]) + np.asarray(params[0].std[k]) * rng.normal(size=s)).astype(np.float32)
               for k, s in obs_sizes.items()}
        a_ref, _ = inference_fn({k: jp.asarray(v) for k, v in obs.items()}, jax.random.PRNGKey(0))
        err = max(err, float(np.abs(np.asarray(a_ref) - pol(obs["state"])).max()))
    if err > 1e-4:
        raise RuntimeError(f"ONNX export mismatch vs Brax inference: {err}")
    return err
