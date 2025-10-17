from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np


@dataclass
class NNLSOptions:
    algo: Literal["HALS", "ASET", "MUUP", "FPGM", "ADMM", "ALSH"] = "HALS"
    init: np.ndarray | None = None
    delta: float = 1e-6
    inneriter: int = 500
    alpha: float | None = None


def _compute_wtw_wtx(W: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    WTW = W.T @ W
    WTX = W.T @ X
    return WTW, WTX


def _nnls_hals(X: np.ndarray, W: np.ndarray, options: NNLSOptions) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    m, n = X.shape
    r = W.shape[1]
    WTW, WTX = _compute_wtw_wtx(W, X)

    if options.init is None or options.init.size == 0:
        H = np.maximum(0.0, np.random.default_rng(0).random((r, n)))
    else:
        H = np.maximum(0.0, np.asarray(options.init, dtype=np.float64).copy())

    eps = 1e-16
    diag_WTW = np.diag(WTW) + eps

    H_prev = H.copy()
    first_diff_norm = None

    for it in range(max(1, options.inneriter)):
        # Cyclic updates for each row k of H
        for k in range(r):
            # Gradient step with exact minimizer for row k
            # hk := max(0, hk + (WTX[k,:] - (WTW[k,:] @ H)) / WTW[k,k])
            numerator = WTX[k, :] - (WTW[k, :] @ H)
            H[k, :] = np.maximum(0.0, H[k, :] + numerator / diag_WTW[k])

        # Stopping based on delta proportion of first iteration movement
        diff = np.linalg.norm(H - H_prev, ord="fro")
        if it == 0:
            first_diff_norm = max(diff, eps)
        else:
            if diff <= options.delta * first_diff_norm:
                break
        H_prev[:, :] = H

    return H, WTW, WTX


def _nnls_mu(X: np.ndarray, W: np.ndarray, options: NNLSOptions) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    m, n = X.shape
    r = W.shape[1]
    WTW, WTX = _compute_wtw_wtx(W, X)

    if options.init is None or options.init.size == 0:
        H = np.maximum(0.0, np.random.default_rng(0).random((r, n)))
    else:
        H = np.maximum(0.0, np.asarray(options.init, dtype=np.float64).copy())

    eps = 1e-16
    for _ in range(max(1, options.inneriter)):
        denom = WTW @ H + eps
        H *= WTX / denom
    return H, WTW, WTX


def _nnls_alsh(X: np.ndarray, W: np.ndarray, options: NNLSOptions) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    WTW, WTX = _compute_wtw_wtx(W, X)
    r = WWTW_shape_r = WTW.shape[0]
    # Regularize if ill-conditioned
    try:
        cond = np.linalg.cond(WTW)
    except Exception:
        cond = 1e12
    if cond > 1e6:
        delta = np.trace(WTW) / max(1, r)
        H = np.linalg.solve(WTW + 1e-6 * delta * np.eye(r), WTX)
    else:
        H = np.linalg.solve(WTW, WTX)
    H = np.maximum(0.0, H)
    return H, WTW, WTX


def nnls(W: np.ndarray, X: np.ndarray, options: NNLSOptions | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if options is None:
        options = NNLSOptions()

    algo = options.algo
    if algo == "HALS":
        return _nnls_hals(X, W, options)
    if algo == "MUUP":
        return _nnls_mu(X, W, options)
    if algo == "ALSH":
        return _nnls_alsh(X, W, options)

    if algo in ("ASET", "FPGM", "ADMM"):
        raise NotImplementedError(f"NNLS algorithm '{algo}' not yet implemented")

    raise ValueError(f"Unknown NNLS algo: {algo}")


