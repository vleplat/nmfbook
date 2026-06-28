from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class ONMFOptions:
    maxiter: int = 200
    timemax: float = 30.0
    epsilon: float = np.finfo(float).eps
    accuracy: float = 1e-6
    W: Optional[np.ndarray] = None
    H: Optional[np.ndarray] = None
    display: int = 1
    random_state: int = 0


def _objective_value(X: np.ndarray, W: np.ndarray, H: np.ndarray) -> float:
    R = X - W @ H
    return 0.5 * float(np.linalg.norm(R, ord="fro") ** 2)


def _assign_clusters_cosine(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Assign each column of X to the closest column of W in cosine similarity.
    Returns an integer array of length n with values in [0, r-1].
    """
    # Normalize columns of W and X for cosine scores
    Wn = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-16)
    Xn = X / (np.linalg.norm(X, axis=0, keepdims=True) + 1e-16)
    scores = Wn.T @ Xn  # (r, n)
    return np.argmax(scores, axis=0)


def _h_from_clusters(labels: np.ndarray, r: int) -> np.ndarray:
    """
    Build H with one non-zero per column and HH^T = I.
    For cluster k with n_k assignments, set H[k, j] = 1/sqrt(n_k) for assigned j.
    """
    n = labels.size
    H = np.zeros((r, n), dtype=float)
    # counts per cluster
    counts = np.bincount(labels, minlength=r).astype(float)
    counts[counts == 0.0] = 1.0  # avoid div-by-zero; leaves row zeros if empty
    scales = 1.0 / np.sqrt(counts)
    for j, k in enumerate(labels):
        H[k, j] = scales[k]
    return H


def onmf(X: np.ndarray, r: int, options: Optional[ONMFOptions] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Orthogonal NMF (ONMF) via alternating assignments (Pompili et al., 2014).
    Constraints: W >= 0, H >= 0, and H H^T = I_r.
    """
    if options is None:
        options = ONMFOptions()
    if np.min(X) < 0:
        raise ValueError("X should be nonnegative")

    m, n = X.shape
    rng = np.random.default_rng(options.random_state)

    if options.W is not None:
        W = np.maximum(options.epsilon, options.W.copy())
        if W.shape != (m, r):
            raise ValueError(f"W has shape {W.shape}, expected {(m, r)}")
    else:
        W = rng.random((m, r))

    if options.H is not None:
        H = np.maximum(options.epsilon, options.H.copy())
        if H.shape != (r, n):
            raise ValueError(f"H has shape {H.shape}, expected {(r, n)}")
        # Normalize rows of H to ensure HH^T = I
        row_norms = np.linalg.norm(H, axis=1, keepdims=True) + 1e-16
        H = H / row_norms
    else:
        # Initialize H from random assignments
        labels = rng.integers(low=0, high=r, size=n)
        H = _h_from_clusters(labels, r)

    # Column-normalize W to stabilize
    for k in range(r):
        col_max = max(float(np.max(W[:, k])), options.epsilon)
        W[:, k] /= col_max

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
        # W update (closed form with nonnegativity projection)
        # With HH^T = I, least-squares: W = X H^T
        W = np.maximum(options.epsilon, X @ H.T)
        # Normalize columns to avoid degeneracy
        norms = np.linalg.norm(W, axis=0, keepdims=True) + 1e-16
        W = W / norms

        # H update via cosine-based assignments, then rescale rows to keep HH^T = I
        labels = _assign_clusters_cosine(X, W)
        H = _h_from_clusters(labels, r)

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

