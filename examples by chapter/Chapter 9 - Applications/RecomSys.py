from __future__ import annotations

import os
import sys
import numpy as np

BASE = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from algorithms.wlra.wlra import wlra, WLRAOptions  # noqa: E402


def main():
    # Toy recommender matrix (zeros are missing entries)
    X = np.array(
        [
            [2, 3, 2, 0, 0],
            [0, 1, 0, 3, 2],
            [1, 0, 4, 1, 0],
            [5, 4, 0, 3, 2],
            [0, 1, 2, 0, 4],
            [1, 0, 3, 4, 3],
        ],
        dtype=float,
    )
    r = 3
    P = (X > 0).astype(float)

    # Attempt multiple random initializations until WH is within [0.5, 6.5]
    # Nonnegativity constraints in WLRA (Weighted NMF)
    tet = 1
    seed = 2020
    W = None
    H = None
    while 1 <= tet <= 100:
        opts = WLRAOptions(
            maxiter=500,
            timemax=10.0,
            display=0,
            random_state=seed,
            zeros_are_missing=True,
            ridge=1e-8,
            nonneg=True,
        )
        seed += 1
        W, H, *_ = wlra(X, P, r, opts)
        WH = W @ H
        tet += 1
        if np.min(WH) >= 0.5 and np.max(WH) <= 6.5:
            break

    # Normalize columns of W to be within [0,5] by max-scaling, push scale to H
    for k in range(r):
        sck = float(np.max(W[:, k])) / 5.0 if W.shape[0] > 0 else 1.0
        sck = sck if sck != 0.0 else 1.0
        W[:, k] = W[:, k] / sck
        H[k, :] = H[k, :] * sck

    WH = W @ H

    # Weighted RMSE (normalized by sum(P))
    rmse = np.sqrt(np.sum(((X - WH) ** 2) * P)) / (np.sum(P) + 1e-16)
    print("W =\n", np.array_str(W, precision=4, suppress_small=True))
    print("H =\n", np.array_str(H, precision=4, suppress_small=True))
    print(f"RMSE = ||X - W*H||_P/||P||_P = {rmse:.4f}")
    print("Approximation:\n", np.array_str(WH, precision=4, suppress_small=True))


if __name__ == "__main__":
    main()

