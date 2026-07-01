from __future__ import annotations

import numpy as np

from algorithms.separable_nmf.snpa_matlab import _nnls_fpgm, SNPAOptions


def gamma2param(W: np.ndarray) -> tuple[float, int]:
    """
    Compute:
      gamma = min_i min_{x in Delta^r} ||W[:,i] - W[:,I] x||_2 / ||W[:,i]||_2
    where I = {1,...,r} \\ {i}, Delta^r = {x >= 0, sum(x) <= 1}.
    Returns (gamma, k) where k is the index attaining the minimum.
    """
    m, r = W.shape
    gamma = float("inf")
    kmin = -1
    for i in range(r):
        Wi = W[:, i]
        J = [j for j in range(r) if j != i]
        C = W[:, J]
        x = _nnls_fpgm(Wi.reshape(-1, 1), C, SNPAOptions(maxitn=200, proj=1, display=0)).ravel()
        res = Wi - C @ x
        gi = np.linalg.norm(res) / (np.linalg.norm(Wi) + 1e-16)
        if gi < gamma:
            gamma = gi
            kmin = i
    return float(gamma), int(kmin)

