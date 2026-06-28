from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np

from algorithms.nmf.nnls import NNLSOptions, nnls


@dataclass
class MinVolNMFOptions:
    maxiter: int = 300
    timemax: float = 60.0
    epsilon: float = np.finfo(float).eps
    accuracy: float = 1e-6
    W: Optional[np.ndarray] = None
    H: Optional[np.ndarray] = None
    display: int = 1
    random_state: int = 0
    # Regularization parameters for logdet term
    lam: float = 1e-2
    delta: float = 1e-6
    # Constraint variant: normalize columns of H (sum=1), or rows of H, or columns of W
    constraint: Literal["H_cols_sum1", "H_rows_sum1", "W_cols_sum1"] = "H_cols_sum1"
    # Gradient step size for W on logdet term
    stepW: float = 1e-2


def _objective_value(X: np.ndarray, W: np.ndarray, H: np.ndarray, lam: float, delta: float) -> float:
    R = X - W @ H
    fro = 0.5 * float(np.linalg.norm(R, ord="fro") ** 2)
    G = W.T @ W + delta * np.eye(W.shape[1])
    sign, logdetG = np.linalg.slogdet(G)
    if sign <= 0:
        # Safeguard; treat as large penalty
        return fro + lam * 1e6
    return fro + lam * float(logdetG)


def _normalize_constraints(W: np.ndarray, H: np.ndarray, opt: MinVolNMFOptions) -> None:
    tiny = opt.epsilon
    if opt.constraint == "H_cols_sum1":
        s = np.sum(H, axis=0, keepdims=True)
        s = np.maximum(s, tiny)
        H[:] = H / s
    elif opt.constraint == "H_rows_sum1":
        s = np.sum(H, axis=1, keepdims=True)
        s = np.maximum(s, tiny)
        H[:] = H / s
    elif opt.constraint == "W_cols_sum1":
        s = np.sum(W, axis=0, keepdims=True)
        s = np.maximum(s, tiny)
        W[:] = W / s


def minvol_nmf(X: np.ndarray, r: int, options: Optional[MinVolNMFOptions] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Minimum-volume NMF:
        min ||X - W H||_F^2 + lam * logdet(W^T W + delta I)
    with a simple normalization constraint variant (default: columns of H sum to 1).
    """
    if options is None:
        options = MinVolNMFOptions()
    if np.min(X) < 0:
        raise ValueError("X should be nonnegative")

    m, n = X.shape
    rng = np.random.default_rng(options.random_state)
    tiny = options.epsilon

    if options.W is not None:
        W = np.maximum(tiny, options.W.copy())
        if W.shape != (m, r):
            raise ValueError(f"W has shape {W.shape}, expected {(m, r)}")
    else:
        W = np.maximum(tiny, rng.random((m, r)))

    if options.H is not None:
        H = np.maximum(tiny, options.H.copy())
        if H.shape != (r, n):
            raise ValueError(f"H has shape {H.shape}, expected {(r, n)}")
    else:
        H = np.maximum(tiny, rng.random((r, n)))

    # Initial scaling
    for k in range(r):
        s = max(tiny, float(np.max(W[:, k])))
        W[:, k] /= s
        H[k, :] *= s

    _normalize_constraints(W, H, options)

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

    nnls_opts = NNLSOptions(algo="HALS", inneriter=500, delta=1e-6)

    while i <= options.maxiter and (__import__("time").perf_counter() - start) <= options.timemax and not_converged():
        # H update: standard NNLS then normalization as per constraint
        H, _, _ = nnls(W, X, nnls_opts)
        _normalize_constraints(W, H, options)

        # W update: NNLS step then gradient step for min-vol term, then nonnegativity
        Wt, _, _ = nnls(H.T, X.T, nnls_opts)
        W = np.maximum(tiny, Wt.T)

        # Gradient step for the logdet penalty
        G = W.T @ W + options.delta * np.eye(r)
        Ginv = np.linalg.inv(G)
        grad = 2.0 * W @ Ginv  # derivative of logdet(W^T W + delta I)
        W = np.maximum(tiny, W - options.stepW * options.lam * grad)

        e_cur = _objective_value(X, W, H, options.lam, options.delta)
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

