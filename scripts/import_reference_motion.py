"""Convert upstream Open Duck polynomial reference motions (pickle) into a plain .npz.

The upstream file is a pickle; we load it with an allowlisted unpickler (numpy
scalars/arrays only) so no arbitrary code can run, then store a dense array:

    coeffs[ix, iy, itheta, dim, order]  polynomial coefficients (lowest order first)
    dxs, dys, dthetas                   command grid
    period, fps                         gait period [s] and sampling rate

Usage:
    python scripts/import_reference_motion.py \
        third_party/Open_Duck_Playground/playground/open_duck_mini_v2/data/polynomial_coefficients.pkl \
        assets/open_duck_mini_v2/reference_motion.npz
"""

import argparse
import pickle

import numpy as np

_ALLOWED = {
    ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "scalar"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy", "dtype"),
    ("numpy", "ndarray"),
}


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in _ALLOWED:
            raise pickle.UnpicklingError(f"blocked global {module}.{name}")
        return super().find_class(module, name)


def convert(src: str, dst: str) -> None:
    with open(src, "rb") as f:
        data = _SafeUnpickler(f).load()

    keys = list(data.keys())
    grid = np.array([[float(v) for v in k.split("_")] for k in keys])
    dxs, dys, dthetas = (np.unique(grid[:, i]) for i in range(3))

    first = data[keys[0]]
    sample = [np.asarray(c, dtype=np.float64) for c in first["coefficients"].values()]
    dim, order = len(sample), max(len(c) for c in sample)

    coeffs = np.zeros((len(dxs), len(dys), len(dthetas), dim, order))
    for k, (dx, dy, dth) in zip(keys, grid):
        ix, iy, it = np.searchsorted(dxs, dx), np.searchsorted(dys, dy), np.searchsorted(dthetas, dth)
        for d, c in enumerate(data[k]["coefficients"].values()):
            c = np.asarray(c, dtype=np.float64)
            coeffs[ix, iy, it, d, : len(c)] = c

    np.savez_compressed(
        dst,
        coeffs=coeffs,
        dxs=dxs,
        dys=dys,
        dthetas=dthetas,
        period=float(first["period"]),
        fps=float(first["fps"]),
        startend_double_support_ratio=float(first["startend_double_support_ratio"]),
    )
    print(f"wrote {dst}: grid {coeffs.shape[:3]}, dim {dim}, order {order}, "
          f"period {first['period']}s @ {first['fps']} fps")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("dst")
    a = p.parse_args()
    convert(a.src, a.dst)
