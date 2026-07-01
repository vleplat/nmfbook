from __future__ import annotations

import os
import sys
import numpy as np

BASE = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from algorithms.separable_nmf.septrisym_nmf import septrisym_nmf
from utils.silence_warnings import silence_numpy_warnings


def main():
    silence_numpy_warnings()
    # Small numerical test for separable tri-symmetric NMF
    r = 5
    n = 20
    rng = np.random.default_rng(0)
    W = np.vstack([np.eye(r), rng.random((n, r))])
    # Normalize columns to sum to 1
    W = W / (np.sum(W, axis=0, keepdims=True) + 1e-16)
    # Permute rows
    perm = rng.permutation(n + r)
    W = W[perm, :]
    # Symmetric S
    S = rng.random((r, r))
    S = 0.5 * (S + S.T)
    # Build A
    A = W @ S @ W.T
    # Recover (Wt, St)
    Wt, St = septrisym_nmf(A, r)
    err = np.linalg.norm(A - Wt @ St @ Wt.T, "fro")
    print("Recovered Wt (first 5x5 block):")
    print(Wt[:5, :5])
    print("Recovered St:")
    print(St)
    print(f"The recovered (Wt,St) satisfy ||A - Wt*St*Wt^T||_F = {err:2.2e}.")


if __name__ == "__main__":
    main()

