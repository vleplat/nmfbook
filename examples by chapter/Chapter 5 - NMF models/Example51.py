from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")

BASE = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from algorithms.beta_nmf import beta_nmf, BetaNMFOptions


def main():
    m = 100
    n = 100
    r = 10
    numtest = 10  # 100 in the book
    print(f"Running {numtest} experiments:")
    res0 = []
    res2 = []
    for i in range(1, numtest + 1):
        print(f"{i}...", end="")
        rng = np.random.default_rng(i)
        mask = (rng.random((m, n)) < 0.5).astype(float)
        X = rng.random((m, n)) * mask + 1e-6
        # Common random init like MATLAB (options.W/H)
        W_init = rng.random((m, r))
        H_init = rng.random((r, n))
        # IS (beta=0) with same init
        opt0 = BetaNMFOptions(beta=0.0, display=0, timemax=10.0, maxiter=500, W=W_init, H=H_init)
        W0, H0, *_ = beta_nmf(X, r, opt0)
        R0 = X - W0 @ H0
        nR0 = np.linalg.norm(R0, ord="fro") + 1e-16
        res0.append([
            np.linalg.norm(np.maximum(R0, 0.0), ord="fro") / nR0,
            np.linalg.norm(np.maximum(-R0, 0.0), ord="fro") / nR0,
        ])
        # Fro (beta=2) with same init style
        opt2 = BetaNMFOptions(beta=2.0, display=0, timemax=10.0, maxiter=500, W=W_init, H=H_init)
        W2, H2, *_ = beta_nmf(X, r, opt2)
        R2 = X - W2 @ H2
        nR2 = np.linalg.norm(R2, ord="fro") + 1e-16
        res2.append([
            np.linalg.norm(np.maximum(R2, 0.0), ord="fro") / nR2,
            np.linalg.norm(np.maximum(-R2, 0.0), ord="fro") / nR2,
        ])
        if i % 10 == 0:
            print()
    res0 = np.array(res0)
    res2 = np.array(res2)
    print("\nSolutions (W,H) for IS-NMF satisfy on average:")
    print(f"||max(0,X-WH)||_F/||X-WH||_F <= {100*np.mean(res0[:,0]):.2f} %")
    print(f"||max(0,WH-X)||_F/||X-WH||_F >= {100*np.mean(res0[:,1]):.2f} %")
    print("Solutions (W,H) for Fro-NMF satisfy on average:")
    print(f"||max(0,X-WH)||_F/||X-WH||_F >= {100*np.mean(res2[:,0]):.2f} %")
    print(f"||max(0,WH-X)||_F/||X-WH||_F <= {100*np.mean(res2[:,1]):.2f} %")


if __name__ == "__main__":
    main()

