from __future__ import annotations

import os
import sys
import numpy as np

BASE = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

# Implement local aggregator to mirror MATLAB lowerbounds_nnr.m using our helpers
from importlib import import_module
sys.path.insert(0, os.path.dirname(__file__))
plb = import_module("py_lower_bounds")
rec_cov_bound = plb.rec_cov_bound
geometric_bound = plb.geometric_bound
nonneg_nuclear_norm_bound = plb.nonneg_nuclear_norm_bound
self_scaled_bound = plb.self_scaled_bound
hyperplane_separation_bound = plb.hyperplane_separation_bound

# Exact NMF heuristic (for comparison, like MATLAB note)
from algorithms import exact_nmf_heuristic  # type: ignore
from algorithms.exact_nmf.exact_nmf import ExactNMFOptions  # type: ignore

def lowerbounds_nnr_py(X: np.ndarray, rnrank: int | None = None):
    rc, _ = rec_cov_bound(X)
    if rnrank is not None:
        geo = geometric_bound(int(np.linalg.matrix_rank(X)), int(rnrank))
    else:
        geo = 0
    nnucnorm = nonneg_nuclear_norm_bound(X)
    tausos = self_scaled_bound(X)
    # Hyperplane separation bound as in MATLAB uses Z = X - 0.5 + 1e-6 then alphaZ
    Z = X - 0.5 + 1e-6
    Z[Z < 0] = -1000
    alphaZ, _, _ = hyperplane_separation_bound(Z)
    hypsep = float(np.sum(Z * X)) / alphaZ / max(1.0, float(np.max(X)))
    return rc, geo, nnucnorm, tausos, hypsep


def main():
    n = 3
    # Build A: all binary vectors of length n as rows
    A = ((np.arange(2**n)[:, None] & (1 << np.arange(n))) > 0).astype(int)
    B = A.copy()
    X = np.zeros((2**n, 2**n), dtype=float)
    for j in range(B.shape[0]):
        X[:, j] = (1 - A @ B[j, :].astype(float)) ** 2
    # MATLAB-style display
    np.set_printoptions(edgeitems=100, linewidth=200, suppress=True)
    print("X =")
    print(X)
    r = 2**n - 1
    print(f"r = {r}")
    # Attempt exact NMF heuristic briefly (as in MATLAB script)
    opts = ExactNMFOptions(maxiter=3, num_inits=1, display=0)
    Wtry, Htry, e_hist, _ = exact_nmf_heuristic(X, r, opts)
    rel_err = float(np.linalg.norm(X - Wtry @ Htry, ord="fro") / max(np.linalg.norm(X, ord="fro"), 1e-16))
    print("Exact NMF heuristic (3 iters) relative error =")
    print(rel_err)
    # Then compute lower bounds
    rc, geo, nnucnorm, tausos, hypsep = lowerbounds_nnr_py(X)
    print("[rc, geo, nnucnorm, tausos, hypsep] =")
    print(rc, geo, nnucnorm, tausos, hypsep)


if __name__ == "__main__":
    main()

