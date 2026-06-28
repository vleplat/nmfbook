from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class WLRAOptions:
    maxiter: int = 200
    timemax: float = 30.0
    epsilon: float = 1e-12
    accuracy: float = 1e-6
    W: Optional[np.ndarray] = None
    H: Optional[np.ndarray] = None
    display: int = 1
    random_state: int = 0
    # If True, treat zeros in P as missing entries; else just small weights
    zeros_are_missing: bool = True
    # Small Tikhonov regularization to stabilize normal equations
    ridge: float = 1e-8
    # Enforce W >= 0 and H >= 0 (Weighted NMF)
    nonneg: bool = False


def _objective_value(X: np.ndarray, P: np.ndarray, W: np.ndarray, H: np.ndarray) -> float:
    R = X - W @ H
    return 0.5 * float(np.linalg.norm(np.sqrt(P) * R, ord="fro") ** 2)


def wlra(X: np.ndarray, P: np.ndarray, r: int, options: Optional[WLRAOptions] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Weighted Low-Rank Approximation:
        min_{W,H} sum_{i,j} P_{i,j} (X_{i,j} - (W H)_{i,j})^2
    Solved by weighted alternating least squares (WALS).
    """
    if options is None:
        options = WLRAOptions()
    if X.shape != P.shape:
        raise ValueError("X and P must have the same shape")
    m, n = X.shape
    rng = np.random.default_rng(options.random_state)

    if options.W is not None and options.H is not None:
        W = np.asarray(options.W, dtype=float).copy()
        H = np.asarray(options.H, dtype=float).copy()
        if W.shape != (m, r) or H.shape != (r, n):
            raise ValueError(f"W,H shapes are {W.shape}, {H.shape}; expected {(m, r)}, {(r, n)}")
    else:
        if options.nonneg:
            # Nonnegative random initialization (faithful to MATLAB WLRA when nonneg=1)
            W = rng.random((m, r))
            H = rng.random((r, n))
        else:
            # SVD init on unweighted data
            U, s, Vt = np.linalg.svd(X, full_matrices=False)
            U_r = U[:, :r]
            S_r = np.diag(s[:r])
            V_r = Vt[:r, :]
            W = U_r @ np.sqrt(S_r)
            H = np.sqrt(S_r) @ V_r

    e_vals: list[float] = []
    t_vals: list[float] = []
    start = __import__("time").perf_counter()
    i = 1
    mintime = 0.1
    cntdis = 0

    sqrtP = np.sqrt(P)
    ridge = options.ridge

    def not_converged() -> bool:
        if i <= 12 or len(e_vals) <= 2:
            return True
        return abs(e_vals[-1] - e_vals[-11]) > options.accuracy * abs(e_vals[-1])

    while i <= options.maxiter and (__import__("time").perf_counter() - start) <= options.timemax and not_converged():
        if options.nonneg:
            # Weighted NNLS updates using per-column/row solves with weights
            try:
                from scipy.optimize import nnls as scipy_nnls  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise ImportError("scipy is required for nonnegative WLRA (options.nonneg=True)") from exc

            # Update H (each column independently)
            for j in range(n):
                pj = sqrtP[:, j]
                if options.zeros_are_missing and np.all(pj == 0.0):
                    continue
                A = W * pj.reshape(-1, 1)           # (m x r)
                b = pj * X[:, j]                    # (m,)
                hj, _ = scipy_nnls(A, b)
                H[:, j] = hj

            # Update W (each row independently)
            HT = H.T  # (n x r)
            for irow in range(m):
                pi = sqrtP[irow, :]
                if options.zeros_are_missing and np.all(pi == 0.0):
                    continue
                A = HT * pi.reshape(-1, 1)          # (n x r)
                b = pi * X[irow, :]                 # (n,)
                wi, _ = scipy_nnls(A, b)
                W[irow, :] = wi
        else:
            # Unconstrained weighted least squares (normal equations)
            # Update H column-wise
            WT = W.T
            for j in range(n):
                pj = sqrtP[:, j]
                if options.zeros_are_missing and np.all(pj == 0.0):
                    continue
                DjW = W * pj.reshape(-1, 1)
                Aj = WT @ DjW + ridge * np.eye(r)
                bj = WT @ (pj * X[:, j])
                try:
                    H[:, j] = np.linalg.solve(Aj, bj)
                except np.linalg.LinAlgError:
                    H[:, j] = np.linalg.lstsq(Aj, bj, rcond=None)[0]

            # Update W row-wise
            HT = H.T
            for irow in range(m):
                pi = sqrtP[irow, :]
                if options.zeros_are_missing and np.all(pi == 0.0):
                    continue
                DiH = H * pi.reshape(1, -1)
                Ai = DiH @ HT + ridge * np.eye(r)
                bi = (pi * X[irow, :]) @ HT
                try:
                    W[irow, :] = np.linalg.solve(Ai, bi)
                except np.linalg.LinAlgError:
                    W[irow, :] = np.linalg.lstsq(Ai, bi, rcond=None)[0]

        e_cur = _objective_value(X, P, W, H)
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

