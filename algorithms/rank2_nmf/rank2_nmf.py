from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from algorithms.nmf.nnls import NNLSOptions, nnls


@dataclass
class Rank2NMFOptions:
    maxiter: int = 300
    timemax: float = 30.0
    epsilon: float = np.finfo(float).eps
    accuracy: float = 1e-6
    display: int = 1
    random_state: int = 0


def _objective_value(X: np.ndarray, W: np.ndarray, H: np.ndarray) -> float:
    R = X - W @ H
    return 0.5 * float(np.linalg.norm(R, ord="fro") ** 2)


def rank2_nmf(X: np.ndarray, options: Optional[Rank2NMFOptions] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Rank-two NMF heuristic:
      - Initialize from rank-2 SVD, threshold to nonnegative
      - Alternate NNLS for H and W with r=2
    If X has true nonnegative rank 2 and is well-scaled, this often recovers an exact factorization.
    """
    if options is None:
        options = Rank2NMFOptions()
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if np.min(X) < 0:
        raise ValueError("X should be nonnegative")

    m, n = X.shape
    r = 2
    tiny = options.epsilon

    # SVD init
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    U2 = U[:, :2]
    S2 = np.diag(s[:2])
    V2 = Vt[:2, :]
    W = np.maximum(tiny, U2 @ np.sqrt(S2))
    H = np.maximum(tiny, np.sqrt(S2) @ V2)

    # Column rescale
    for k in range(r):
        cmax = max(float(np.max(W[:, k])), tiny)
        W[:, k] /= cmax
        H[k, :] *= cmax

    nnls_opts = NNLSOptions(algo="HALS", inneriter=500, delta=1e-6)

    e_vals: list[float] = []
    t_vals: list[float] = []
    start = __import__("time").perf_counter()
    i = 1
    mintime = 0.1
    cntdis = 0

    def not_converged() -> bool:
        if i <= 12 or len(e_vals) <= 2:
            return True
        return abs(e_vals[-1] - e_vals[-11]) > options.accuracy * abs(e_vals[-1])

    while i <= options.maxiter and (__import__("time").perf_counter() - start) <= options.timemax and not_converged():
        # H update
        H, _, _ = nnls(W, X, nnls_opts)
        # W update
        Wt, _, _ = nnls(H.T, X.T, nnls_opts)
        W = Wt.T

        # Normalize columns
        for k in range(r):
            nrm = max(tiny, float(np.linalg.norm(W[:, k])))
            W[:, k] /= nrm
            H[k, :] *= nrm

        e_cur = _objective_value(X, W, H)
        e_vals.append(e_cur)
        t_vals.append(__import__("time").perf_counter() - start)

        if options.display == 1 and t_vals[-1] >= mintime:
            print(f"{i}...", end="")
            mintime *= 2.0
            cntdis += 1
            if cntdis % 10 == 0:
                print()

        i += 1

    if options.display == 1:
        print()

    return W, H, np.array(e_vals), np.array(t_vals)

