from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class ProjectiveNMFOptions:
    maxiter: int = 500
    timemax: float = 60.0
    epsilon: float = np.finfo(float).eps
    accuracy: float = 1e-6
    W: Optional[np.ndarray] = None
    display: int = 1
    # Rescale columns of W every N iterations (0 disables)
    rescale_every: int = 0
    random_state: int = 0


def _objective_value_raw(X: np.ndarray, W: np.ndarray) -> float:
    # MATLAB e computes sqrt( nX2 - 2*sum(sum(XXtW.*W)) + sum(sum(WtW.*XtWtXtW)) )
    XtW = X.T @ W
    XXtW = X @ XtW
    WtW = W.T @ W
    XtWtXtW = XtW.T @ XtW
    nX2 = float(np.sum(X * X))
    val = nX2 - 2.0 * float(np.sum(XXtW * W)) + float(np.sum(WtW * XtWtXtW))
    return float(np.sqrt(max(0.0, val)))


def projective_nmf(X: np.ndarray, r: int, options: Optional[ProjectiveNMFOptions] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Projective NMF (Yang & Oja): min_{W>=0} ||X - W W^T X||_F^2

    Parameters
    ----------
    X : np.ndarray
        Nonnegative data matrix of shape (m, n).
    r : int
        Target rank (number of columns of W).
    options : ProjectiveNMFOptions, optional
        Algorithm options.

    Returns
    -------
    W : np.ndarray
        Basis matrix of shape (m, r), nonnegative.
    H : np.ndarray
        Coefficient matrix H = W^T X of shape (r, n), nonnegative.
    (e_vals, t_vals) : tuple[np.ndarray, np.ndarray]
        Objective values and elapsed times per iteration.
    """
    if options is None:
        options = ProjectiveNMFOptions()
    if np.min(X) < 0:
        raise ValueError("X should be nonnegative")

    m, _ = X.shape
    rng = np.random.default_rng(options.random_state)

    if options.W is not None:
        W = np.maximum(options.epsilon, options.W.copy())
        if W.shape != (m, r):
            raise ValueError(f"W has shape {W.shape}, expected {(m, r)}")
    else:
        W = rng.random((m, r))

    # Scale columns of W to have max 1
    tiny = options.epsilon
    for k in range(r):
        col_max = max(float(np.max(W[:, k])), tiny)
        W[:, k] /= col_max

    nX2 = float(np.sum(X * X))

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
        # MATLAB-style optimal scaling to avoid oscillations
        XtW = X.T @ W          # (n x r)
        XXtW = X @ XtW         # (m x r)
        WtW = W.T @ W          # (r x r)
        XtWtXtW = XtW.T @ XtW  # (r x r)
        num_alpha = float(np.sum(XXtW * W))
        den_alpha = float(np.sum(WtW * XtWtXtW)) + 1e-16
        alpha = num_alpha / den_alpha if den_alpha > 0 else 1.0
        if not np.isfinite(alpha) or alpha <= 0:
            alpha = 1.0
        s = np.sqrt(alpha)
        W *= s
        XtW *= s
        XXtW *= s
        WtW *= alpha
        XtWtXtW *= alpha

        # Error (unnormalized) like MATLAB, then normalize at the end if needed by caller
        e_cur = float(np.sqrt(max(0.0, nX2 - 2.0 * float(np.sum(XXtW * W)) + float(np.sum(WtW * XtWtXtW)))))
        e_vals.append(e_cur)
        t_vals.append(__import__("time").perf_counter() - start)

        # MU update: W = W .* (2*XXtW) ./ ( W*(XtWtXtW) + (XXtW)*(WtW) )
        denom = (W @ XtWtXtW) + (XXtW @ WtW)
        W = np.maximum(tiny, W * ((2.0 * XXtW) / (denom + tiny)))

        # Optional coupled rescaling to stabilize magnitudes
        if options.rescale_every and (i % options.rescale_every == 0):
            for k in range(r):
                norm_w = max(tiny, float(np.linalg.norm(W[:, k])))
                W[:, k] /= norm_w

        if options.display == 1 and t_vals[-1] >= mintime:
            print(f"{i}...", end="")
            mintime *= 2.0
            cntdis += 1
            if cntdis % 10 == 0:
                print()

        i += 1

    if options.display == 1:
        print()

    H = W.T @ X
    return W, H, (np.array(e_vals), np.array(t_vals))

