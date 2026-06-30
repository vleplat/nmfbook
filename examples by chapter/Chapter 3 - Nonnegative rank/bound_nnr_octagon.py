from __future__ import annotations

import os
import sys
import numpy as np

BASE = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from .py_lower_bounds import rec_cov_bound, geometric_bound, nonneg_nuclear_norm_bound, self_scaled_bound


def main():
    # MATLAB-style prints and pauses
    a = np.sqrt(2.0) / 2.0
    A = np.array([
        [1, a, 0, -a, -1, -a, 0, a],
        [0, a, 1, a, 0, -a, -1, -a],
    ]).T
    AT = A.T
    print("*****************************************************")
    print("Illustration of the lower bounds for the nonnegative")
    print("  rank on the slack matrix of the regular octagon")
    print("*****************************************************")
    print("The octagon can be represented as {x | Ax <= b} with")
    print("AT =")
    print(AT)
    b = (1 + a) * np.ones((8, 1))
    bT = b.T
    print("bT =")
    print(bT)
    V = np.array([
        [1 + a, a],
        [a, 1 + a],
        [-a, 1 + a],
        [-(1 + a), a],
        [-(1 + a), -a],
        [-a, -(1 + a)],
        [a, -(1 + a)],
        [1 + a, -a],
    ])
    print("Its vertices are the columns of the matrix")
    print("V' =")
    print(V.T)
    S = b[0, 0] - A @ V.T
    print("Its slack matrix is given by")
    print("S =")
    print(S)
    # Provided exact NMF S=WH (just verify error)
    W = np.array([
        [1, 0, 0, 1, 0, 1 + 2 * a],
        [1 + 2 * a, 0, 0, 0, 1, 2 + 2 * a],
        [1, 1, 0, 0, 0, 1 + 2 * a],
        [0, 2 * a, 1, 0, 0, 1],
        [0, 1, 1 + 2 * a, 0, 1, 0],
        [1, 0, 2 + 2 * a, 0, 1 + 2 * a, 0],
        [0, 0, 1 + 2 * a, 1, 1, 0],
        [0, 0, 1, 2 * a, 0, 1],
    ])
    H = np.array([
        [0, 0, 0, 1, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 1, 1 + 2 * a, 1 + 2 * a],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1 + 2 * a, 1 + 2 * a, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 1, 0, 0],
    ])
    err = np.linalg.norm(S - W @ H, ord="fro")
    print(f"We have ||S-WH||_F = {err:.2e}.")
    # Lower bounds
    rc, _ = rec_cov_bound(S)
    print(f"Rectangle covering bound rc = {rc}")
    # Other bounds (geometric bound needs rnrank; here rnrank=8)
    geo = geometric_bound(int(np.linalg.matrix_rank(S)), 8)
    print("Geometric bound =", geo)
    nnuc = nonneg_nuclear_norm_bound(S)
    print("Nonnegative nuclear norm bound =", nnuc)
    tausos = self_scaled_bound(S)
    print("Self-scaled bound =", tausos)


if __name__ == "__main__":
    main()

