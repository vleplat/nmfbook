from __future__ import annotations

import numpy as np

from nchoose2 import nchoose2
from gamma2param import gamma2param


def synthdatasetSepNMF(m: int, n: int, r: int, delta: float, xp: int, condW: int = 3, diri: float = 0.5):
    """
    Generate synthetic datasets for separable NMF experiments.
    Returns (W, H, Noise).
    """
    rng = np.random.default_rng(0)
    if r > m:
        # mirror MATLAB warning; proceed
        pass
    # Generate W with gamma2 >= 10^(-condW)
    target = 10.0 ** (-condW)
    gamma2 = 0.0
    while gamma2 < target:
        W = rng.random((m, r))
        if xp >= 3:
            # ill-conditioned: enforce singular values ~ logspace(-condW, 0)
            U, _, Vt = np.linalg.svd(W, full_matrices=False)
            sv = np.logspace(-condW, 0, r)
            W = (U * sv) @ Vt
        gamma2, _ = gamma2param(W)
    # Generate H and Noise
    if xp in (1, 3):
        # Dirichlet: columns repeat anchors and additional Dirichlet samples
        alpha = np.full(r, diri) if diri > 0 else rng.random(r)
        Hdir = rng.dirichlet(alpha, size=n).T  # r x n
        H = np.concatenate([np.eye(r), Hdir], axis=1)
        Noise = rng.standard_normal((m, n + r))
        Noise = delta * Noise / (np.linalg.norm(Noise, "fro") + 1e-16) * np.linalg.norm(W @ H, "fro")
    else:
        # Middle points: all midpoints of pairs
        nn = r * (r - 1) // 2
        Hmid = nchoose2(r) * 0.5
        H = np.concatenate([np.eye(r), Hmid], axis=1)
        X = W @ H
        Noise = delta * np.concatenate([np.zeros((m, r)), X[:, r:] - np.mean(W, axis=1, keepdims=True).repeat(nn, axis=1)], axis=1)
    return W, H, Noise

