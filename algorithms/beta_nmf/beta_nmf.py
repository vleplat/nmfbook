from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np


@dataclass
class BetaNMFOptions:
    maxiter: int = 500
    timemax: float = 60.0
    beta: float = 1.0
    extrapol: Literal["nesterov", "ptsengv1", "ptsengv2", "noextrap"] | None = None
    epsilon: float = np.finfo(float).eps
    accuracy: float = 1e-6
    W: Optional[np.ndarray] = None
    H: Optional[np.ndarray] = None
    display: int = 1
    # Rescale columns of W and rows of H every N iterations (0 disables)
    rescale_every: int = 0


def _nd_mubeta(X: np.ndarray, W: np.ndarray, H: np.ndarray, beta: float) -> tuple[np.ndarray, np.ndarray, Optional[float]]:
    eps = np.finfo(float).eps
    e_val = None
    if beta == 1.0:
        # KL divergence
        XdWH = X / (W @ H + eps)
        N = W.T @ XdWH
        D = (np.sum(W, axis=0).reshape(-1, 1)) + eps
        # error
        Xnnz = X[X > 0]
        XdWHnnz = XdWH[X > 0]
        e_val = np.sum(Xnnz * np.log(XdWHnnz + eps)) - np.sum(X) + np.sum(D * H)
        return N, D, e_val
    elif beta == 2.0:
        N = W.T @ X
        WtW = W.T @ W
        D = WtW @ H + eps
        e_val = 0.5 * (np.linalg.norm(X, ord="fro") ** 2 - 2 * np.sum(N * H) + np.sum(WtW * (H @ H.T)))
        return N, D, e_val
    else:
        WH = W @ H + eps
        N = W.T @ ((WH + eps) ** (beta - 2.0) * X)
        D = W.T @ ((WH + eps) ** (beta - 1.0))
        # exact betadiv not required during updates; skip for speed
        return N, D, None


def _mubeta_update(X: np.ndarray, W: np.ndarray, H: np.ndarray, beta: float, epsilon: float) -> tuple[np.ndarray, Optional[float]]:
    N, D, e = _nd_mubeta(X, W, H, beta)
    ratio = N / (D + np.finfo(float).eps)
    if 1.0 <= beta <= 2.0:
        H_new = np.maximum(epsilon, H * ratio)
    else:
        gamma = 1.0 / (2.0 - beta) if beta < 1.0 else 1.0 / (beta - 1.0)
        H_new = np.maximum(epsilon, H * (ratio ** gamma))
    return H_new, e


def beta_nmf(X: np.ndarray, r: int, options: Optional[BetaNMFOptions] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if options is None:
        options = BetaNMFOptions()

    if np.min(X) < 0:
        raise ValueError("X should be nonnegative")

    m, n = X.shape
    W = np.maximum(options.epsilon, options.W) if options.W is not None else np.random.default_rng(0).random((m, r))
    H = np.maximum(options.epsilon, options.H) if options.H is not None else np.random.default_rng(1).random((r, n))

    beta = options.beta
    if options.extrapol is None:
        if 1.0 <= beta <= 2.0:
            options.extrapol = "nesterov"
        else:
            options.extrapol = "noextrap"

    # scale columns of W to have max 1
    for k in range(r):
        mxk = max(np.max(W[:, k]), options.epsilon)
        W[:, k] = W[:, k] / mxk
        H[k, :] = H[k, :] * mxk

    Hp = H.copy()
    Wp = W.copy()
    cparam = 1e30
    nutprev = 1.0

    e_vals: list[float] = []
    t_vals: list[float] = []

    start = __import__("time").perf_counter()
    i = 1
    # display pacing
    mintime = 0.1
    cntdis = 0

    while i <= options.maxiter and (__import__("time").perf_counter() - start) <= options.timemax and (i <= 12 or len(e_vals) <= 2 or abs(e_vals[-1] - e_vals[-11]) > options.accuracy * abs(e_vals[-1])):
        # compute extrapolated points
        stepH = np.maximum(0.0, H - Hp)
        if options.extrapol == "nesterov":
            nut = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * (nutprev ** 2)))
            extrapolparam = (nutprev - 1.0) / nut
            nutprev = nut
        elif options.extrapol == "ptsengv1":
            t_i = max(i, 1)
            extrapolparam = (t_i - 1.0) / t_i
        elif options.extrapol == "ptsengv2":
            t_i = max(i, 1)
            extrapolparam = t_i / (t_i + 1.0)
        else:
            extrapolparam = 0.0
        normstepH = np.linalg.norm(stepH, ord="fro")
        if normstepH == 0.0:
            extrapH = extrapolparam
        else:
            extrapH = min(extrapolparam, cparam / (i ** (1.5 / 2.0)) / normstepH)
        He = H + extrapH * stepH

        stepW = np.maximum(0.0, W - Wp)
        normstepW = np.linalg.norm(stepW, ord="fro")
        if normstepW == 0.0:
            extrapW = extrapolparam
        else:
            extrapW = min(extrapolparam, cparam / (i ** (1.5 / 2.0)) / normstepW)
        We = W + extrapW * stepW

        Hp = H.copy()
        Wp = W.copy()

        # MU updates with extrapolated points
        H, _ = _mubeta_update(X, We, He, beta, options.epsilon)
        Wt, _ = _mubeta_update(X.T, H.T, We.T, beta, options.epsilon)
        W = Wt.T

        # error value (D_beta(X, WH)) at current iterate
        # Use KL (beta=1) or Fro (beta=2) specialized formula; otherwise compute generic beta-div
        if beta == 1.0:
            eps = np.finfo(float).eps
            XdWH = X / (W @ H + eps)
            Xnnz = X[X > 0]
            XdWHnnz = XdWH[X > 0]
            e_cur = float(np.sum(Xnnz * np.log(XdWHnnz + eps)) - np.sum(X) + np.sum((np.sum(W, axis=0).reshape(-1, 1)) * H))
        elif beta == 2.0:
            WtW = W.T @ W
            e_cur = float(0.5 * (np.linalg.norm(X, ord="fro") ** 2 - 2 * np.sum((W.T @ X) * H) + np.sum(WtW * (H @ H.T))))
        else:
            WH = W @ H + np.finfo(float).eps
            e_cur = float(np.sum((WH ** (beta) - beta * X * (WH ** (beta - 1.0))) / (beta * (beta - 1.0))))

        e_vals.append(e_cur)
        t_vals.append(__import__("time").perf_counter() - start)

        # Optional coupled rescaling to stabilize magnitudes while preserving WH
        if options.rescale_every and (i % options.rescale_every == 0):
            tiny = 1e-16
            for k in range(r):
                nw = max(tiny, float(np.linalg.norm(W[:, k])))
                nh = max(tiny, float(np.linalg.norm(H[k, :])))
                s = np.sqrt(nh / nw)
                W[:, k] *= s
                H[k, :] /= s

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


