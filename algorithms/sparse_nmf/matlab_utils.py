from __future__ import annotations

from typing import Optional, Tuple, List

import numpy as np


def _power_iter_spectral_norm(A: np.ndarray, iters: int = 30) -> float:
    v = np.random.default_rng(0).random(A.shape[1])
    v /= (np.linalg.norm(v) + 1e-16)
    for _ in range(iters):
        v = A @ v
        n = np.linalg.norm(v)
        if n == 0.0:
            return 1.0
        v /= n
    return float(np.linalg.norm(A @ v))


def project_to_hoyer_sparsity(x: np.ndarray, s: float, tol: float = 1e-8, maxit: int = 100) -> np.ndarray:
    n = x.size
    if n == 0:
        return x
    x = np.maximum(0.0, x.astype(float))
    l2 = np.linalg.norm(x)
    if l2 == 0.0:
        return x
    target_l1_over_l2 = np.sqrt(n) - s * (np.sqrt(n) - 1.0)
    target_l1 = target_l1_over_l2 * l2
    l1 = np.sum(x)
    if abs(l1 - target_l1) <= tol * max(1.0, target_l1):
        return x
    lo, hi = 0.0, float(np.max(x))
    for _ in range(maxit):
        theta = 0.5 * (lo + hi)
        y = np.maximum(0.0, x - theta)
        l1y = float(np.sum(y))
        if abs(l1y - target_l1) <= tol * max(1.0, target_l1):
            x_proj = y
            break
        if l1y > target_l1:
            lo = theta
        else:
            hi = theta
    else:
        x_proj = np.maximum(0.0, x - 0.5 * (lo + hi))
    l2_new = np.linalg.norm(x_proj)
    if l2_new > 0.0:
        x_proj = x_proj * (l2 / l2_new)
    return x_proj


def weightedgroupedsparseproj_col(X: np.ndarray, s: float, w: Optional[List[np.ndarray]] = None) -> np.ndarray:
    if s is None:
        return X
    Xp = X.copy()
    m, r = X.shape
    for k in range(r):
        x = Xp[:, k]
        if w is not None and k < len(w) and w[k] is not None:
            wk = np.asarray(w[k]).reshape(-1)
            wk = np.maximum(1e-16, wk)
            y = x / wk
            y = project_to_hoyer_sparsity(y, s)
            x_proj = np.maximum(0.0, y * wk)
        else:
            x_proj = project_to_hoyer_sparsity(x, s)
        Xp[:, k] = np.maximum(0.0, x_proj)
    return Xp


def fastgradsparseNNLS(X: np.ndarray, W: np.ndarray, H_init: np.ndarray, inneriter: int = 50, s: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    m, n = X.shape
    r = W.shape[1]
    H = np.maximum(0.0, H_init.copy())
    Z = H.copy()
    t = 1.0
    WtW = W.T @ W
    WtX = W.T @ X
    L = _power_iter_spectral_norm(WtW)
    step = 1.0 / (L + 1e-16)
    for _ in range(max(1, inneriter)):
        G = (WtW @ Z) - WtX
        H_next = np.maximum(0.0, Z - step * G)
        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        Z = H_next + ((t - 1.0) / t_next) * (H_next - H)
        H = H_next
        t = t_next
    if s is not None:
        for k in range(r):
            H[k, :] = np.maximum(0.0, project_to_hoyer_sparsity(H[k, :], s))
    XHt = X @ H.T
    HHt = H @ H.T
    return H, XHt, HHt

