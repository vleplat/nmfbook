from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class RecursiveNMUOptions:
    Cnorm: int = 2
    maxiter: int = 100
    display: int = 1


def _wmedian(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    mask = y > 1e-16
    if not np.any(mask):
        return np.zeros(A.shape[0])
    Ay = A[:, mask]
    yy = y[mask].astype(float)
    yy = yy / (float(np.sum(yy)) + 1e-16)
    Ay_scaled = Ay / (yy.reshape(1, -1) + 1e-16)
    m = Ay_scaled.shape[0]
    order = np.argsort(Ay_scaled, axis=1)
    row_idx = np.arange(m)[:, None]
    Yord = np.tile(yy.reshape(1, -1), (m, 1))[row_idx, order]
    csum = np.cumsum(Yord, axis=1)
    idx = np.argmax(csum >= 0.5, axis=1)
    Aord = Ay_scaled[row_idx, order]
    x = Aord[np.arange(m), idx] if m > 1 else Aord[0, idx]
    return np.maximum(0.0, x)


def recursive_nmu(M: np.ndarray, r: int, options: Optional[RecursiveNMUOptions] = None) -> Tuple[np.ndarray, np.ndarray]:
    if options is None:
        options = RecursiveNMUOptions()
    m, n = M.shape
    U = np.zeros((m, r))
    V = np.zeros((n, r))
    R = M.copy()  # residual to pass to next rank-1 factor (updated after each k)
    if options.display:
        print("Recursion started...")
    for k in range(r):
        # Initialization of (x,y) with optimal rank-1 NMF of current residual
        u, s, vt = np.linalg.svd(R, full_matrices=False)
        x = np.abs(u[:, 0]) * np.sqrt(s[0])
        y = np.abs(vt[0, :]) * np.sqrt(s[0])
        U[:, k] = x
        V[:, k] = y
        # Current matrix for this component (fixed within inner loop)
        Mk = R.copy()
        lam = np.maximum(0.0, -(Mk - np.outer(x, y)))
        for j in range(1, options.maxiter + 1):
            # A = M - lambda (independent of current (x,y))
            A = Mk - lam
            if options.Cnorm == 1:
                x = _wmedian(A, y)
                y = _wmedian(A.T, x)
            else:
                x = np.maximum(0.0, A @ y)
                mx = float(np.max(x)) + 1e-16
                x = x / mx
                denom = float(x @ x) + 1e-16
                y = np.maximum(0.0, (A.T @ x) / denom)
            if x.sum() != 0.0 and y.sum() != 0.0:
                Rk = Mk - np.outer(x, y)
                U[:, k] = x
                V[:, k] = y
                lam = np.maximum(0.0, lam - Rk / (j + 1.0))
            else:
                lam = lam / 2.0
                x = U[:, k]
                y = V[:, k]
        # Update residual for next component
        R = np.maximum(0.0, R - np.outer(U[:, k], V[:, k]))
        if options.display:
            print(f"{k+1}...", end="" if (k + 1) % 10 else "\n")
    if options.display:
        print("Done.")
    return U, V.T

