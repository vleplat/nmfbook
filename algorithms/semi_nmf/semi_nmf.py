from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from algorithms.nmf.nnls import NNLSOptions, nnls


@dataclass
class SemiNMFOptions:
    maxiter: int = 100
    timemax: float = 60.0
    epsilon: float = np.finfo(float).eps
    display: int = 1
    random_state: int = 0


def _objective_value(X: np.ndarray, W: np.ndarray, H: np.ndarray) -> float:
    R = X - W @ H
    return float(np.linalg.norm(R, ord="fro"))


def _svd_init_seminmf(X: np.ndarray, r: int) -> Tuple[np.ndarray, np.ndarray]:
    # Faithful port of SVDinitSemiNMF:
    m, n = X.shape
    if r == 1:
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        B = Vt[0, :]
        if np.sum(B > 0) < np.sum(B < 0):
            B = -B
        V = np.maximum(0.0, B.reshape(1, -1))
        # U = X / V (right division): X * V^T * (V V^T)^{-1}
        VVt = V @ V.T
        Umat = (X @ V.T) @ np.linalg.pinv(VVt)
        return Umat, V
    else:
        # Rank-(r-1) SVD
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        A = U[:, : (r - 1)] @ np.diag(s[: (r - 1)])
        B = Vt[: (r - 1), :]
        # init=1: flip signs to maximize minimum entry per row of B
        for i in range(r - 1):
            if np.min(B[i, :]) < np.min(-B[i, :]):
                B[i, :] = -B[i, :]
                A[:, i] = -A[:, i]
        if r == 2:
            Uinit = np.concatenate([A, -A], axis=1)
        else:
            Uinit = np.concatenate([A, -np.sum(A, axis=1, keepdims=True)], axis=1)
        Vtop = B
        Vlast = np.zeros((1, n))
        Vinit = np.vstack([Vtop, Vlast])
        if r >= 3:
            Vinit = Vinit - np.ones((r, 1)) * np.min(np.minimum(0.0, B), axis=0, keepdims=True)
        else:
            Vinit = Vinit - np.ones((r, 1)) * np.minimum(B, 0.0)
        return Uinit, np.maximum(0.0, Vinit)


def semi_nmf(X: np.ndarray, r: int, options: Optional[SemiNMFOptions] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Faithful 2-BCD semi-NMF (MATLAB-like):
      - SVD init for (W,H) with H >= 0
      - Iterate:
         W = X H^T (H H^T)^-1          (unconstrained optimal W)
         H = argmin_{H>=0} ||X - W H|| (HALS via NNLS)
    """
    if options is None:
        options = SemiNMFOptions()
    if X.ndim != 2:
        raise ValueError("X must be 2D")

    m, n = X.shape
    tiny = options.epsilon

    # SVD init
    W, H = _svd_init_seminmf(X, r)
    H = np.maximum(H, tiny)

    e_vals: list[float] = []
    t_vals: list[float] = []
    start = __import__("time").perf_counter()
    i = 1
    mintime = 0.1
    cntdis = 0

    while i <= options.maxiter and (__import__("time").perf_counter() - start) <= options.timemax:
        # Optimal W given H: W = X H^T (H H^T)^-1
        HHT = H @ H.T
        try:
            W = X @ H.T @ np.linalg.pinv(HHT)
        except np.linalg.LinAlgError:
            W = X @ H.T @ np.linalg.pinv(HHT + 1e-8 * np.eye(r))

        # Optimal H >= 0 given W via NNLS (HALS)
        H, _, _ = nnls(W, X, NNLSOptions(algo="HALS", init=H, delta=0.1, inneriter=100, alpha=0.5))
        H = np.maximum(H, tiny)

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

