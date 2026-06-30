from __future__ import annotations

import os
import sys
import numpy as np
import time

BASE = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

# Local imports from same folder
sys.path.insert(0, os.path.dirname(__file__))
from importlib import import_module
plb = import_module("py_lower_bounds")
rec_cov_bound = plb.rec_cov_bound
geometric_bound = plb.geometric_bound
nonneg_nuclear_norm_bound = plb.nonneg_nuclear_norm_bound
self_scaled_bound = plb.self_scaled_bound
hyperplane_separation_bound = plb.hyperplane_separation_bound


def lowerbounds_nnr_py(X: np.ndarray, rnrank: int | None = None):
    rc, _ = rec_cov_bound(X)
    if rnrank is not None:
        geo = geometric_bound(int(np.linalg.matrix_rank(X)), int(rnrank))
    else:
        geo = 0
    nnucnorm = nonneg_nuclear_norm_bound(X)
    tausos = self_scaled_bound(X)
    Z = X - 0.5 + 1e-6
    Z[Z < 0] = -1000
    alphaZ, _, _ = hyperplane_separation_bound(Z)
    hypsep = float(np.sum(Z * X)) / alphaZ / max(1.0, float(np.max(X)))
    return rc, geo, nnucnorm, tausos, hypsep


def main():
    n = 6
    X = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            X[i, j] = float((i + 1 - (j + 1)) ** 2)
    print("X =\n", X)
    start = time.perf_counter()
    rc, geo, nnucnorm, tausos, hypsep = lowerbounds_nnr_py(X, rnrank=n)
    elapsed = time.perf_counter() - start
    print("Lower bounds (rc, geo, nnucnorm, tausos, hypsep):")
    print(rc, geo, nnucnorm, tausos, hypsep)
    print(f"Elapsed: {elapsed:.2f}s")
    print("% The restricted nonnegative rank for linear EDM is n")


if __name__ == "__main__":
    main()

