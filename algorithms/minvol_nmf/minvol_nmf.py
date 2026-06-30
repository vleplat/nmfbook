from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np

from algorithms.nmf.nnls import NNLSOptions, nnls
from algorithms.separable_nmf.snpa_matlab import _nnls_fpgm, SNPAOptions, snpa_matlab  # FPGM NNLS and SNPA init
from utils.simplex import simplex_proj, simplex_col_proj


@dataclass
class MinVolNMFOptions:
    maxiter: int = 100
    timemax: float = 60.0
    epsilon: float = np.finfo(float).eps
    accuracy: float = 1e-6
    W: Optional[np.ndarray] = None
    H: Optional[np.ndarray] = None
    display: int = 1
    random_state: int = 0
    # Regularization parameters for logdet term
    lam: float = 0.1
    delta: float = 0.1
    inneriter: int = 10
    # Model per MATLAB:
    # 1: H^T e <= e; 2: H e = e; 3: W^T e = e; 4: H^T e = e
    model: Literal[1, 2, 3, 4] = 3
    # Optional target relative error for lambda tuning
    target: Optional[float] = None


def _objective_value(X: np.ndarray, W: np.ndarray, H: np.ndarray, lam: float, delta: float) -> float:
    R = X - W @ H
    fro = float(np.linalg.norm(R, ord="fro") ** 2)
    G = W.T @ W + delta * np.eye(W.shape[1])
    sign, logdetG = np.linalg.slogdet(G)
    if sign <= 0:
        # Safeguard; treat as large penalty
        return fro + lam * 1e6
    return fro + lam * float(logdetG)


def _normalize_wh(W: np.ndarray, H: np.ndarray, model: int, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Faithful port of normalizeWH.m
    if model == 1:
        # H^T e <= e (cols of H sum to at most 1) via simplex projection per column
        Hn = np.column_stack([simplex_proj(H[:, j]) for j in range(H.shape[1])])
        if np.linalg.norm(Hn - H) > 1e-3 * max(1e-16, np.linalg.norm(Hn)):
            H = Hn
            # Reoptimize W since not wlog
            opts = SNPAOptions(maxitn=100, proj=0, display=0)
            Wt = _nnls_fpgm(X.T, H.T, opts)  # returns (r x m)
            W = Wt.T
        H = Hn
    elif model == 2:
        # H e = e (rows of H sum to 1)
        scalH = np.sum(H, axis=1, keepdims=True)
        scalH = np.maximum(scalH, 1e-16)
        H = (H.T / scalH.ravel()).T
        W = W @ np.diagflat(scalH.ravel())
    elif model == 3:
        # W^T e = e (cols of W sum to 1)
        scalW = np.sum(W, axis=0, keepdims=True)
        scalW = np.maximum(scalW, 1e-16)
        H = np.diagflat(scalW.ravel()) @ H
        W = W / scalW
    elif model == 4:
        # H^T e = e (cols of H sum to 1)
        Hn = simplex_col_proj(H)
        if np.linalg.norm(Hn - H) > 1e-3 * max(1e-16, np.linalg.norm(Hn)):
            H = Hn
            # Reoptimize W since not wlog
            opts = SNPAOptions(maxitn=100, proj=0, display=0)
            Wt = _nnls_fpgm(X.T, H.T, opts)
            W = Wt.T
        H = Hn
    return W, H


def _fgm_qp_nonneg(A: np.ndarray, C: np.ndarray, W0: np.ndarray, maxiter: int, proj_flag: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast gradient method for QP with nonnegativity/column-simplex projection (FGMqpnonneg.m).
    Solves min sum_i (x_i^T A x_i - 2 c_i^T x_i) row-wise over W.
    proj_flag: 1 -> nonnegative orthant; 2 -> columns sum-to-one (simplex on columns).
    """
    m, r = C.shape
    if W0 is None or W0.size == 0:
        W = np.zeros((m, r))
    else:
        W = W0.copy()
    if maxiter is None or maxiter <= 0:
        maxiter = 500
    # Lipschitz constant and acceleration parameter
    L = float(np.linalg.norm(A, 2))
    # cond inverses; safeguard if ill-conditioned
    try:
        condA = float(np.linalg.cond(A))
        condAm1 = 1.0 / max(condA, 1e-12)
    except Exception:
        condAm1 = 0.0
    beta = (1 - np.sqrt(condAm1)) / (1 + np.sqrt(condAm1)) if condAm1 > 0 else 0.0
    # Project init
    if proj_flag == 1:
        W = np.maximum(W, 0.0)
    elif proj_flag == 2:
        # Project columns onto simplex
        W = simplex_col_proj(W)
    Y = W.copy()
    e_vals: list[float] = []
    delta = 1e-6
    eps0 = 0.0
    eps = 1.0
    i = 1
    while i <= maxiter and eps >= delta * max(eps0, 1e-16):
        Wp = W.copy()
        # Gradient step from Y: grad = Y A - C
        W = Y - (Y @ A - C) / max(L, 1e-16)
        # Projection
        if proj_flag == 1:
            W = np.maximum(W, 0.0)
        elif proj_flag == 2:
            W = simplex_col_proj(W)
        # Extrapolation
        Y = W + beta * (W - Wp)
        # Objective value
        e_vals.append(float(np.sum((W.T @ W) * A) - 2 * np.sum(W * C)))
        # Restart heuristic
        if i >= 2 and e_vals[-1] > e_vals[-2]:
            Y = W.copy()
        if i == 1:
            eps0 = float(np.linalg.norm(W - Wp, ord="fro"))
        eps = float(np.linalg.norm(W - Wp, ord="fro"))
        i += 1
    return W, np.array(e_vals)


def minvol_nmf(X: np.ndarray, r: int, options: Optional[MinVolNMFOptions] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Faithful port of minvolNMF.m:
      min ||X - W H||_F^2 + lambda' * logdet(W^T W + delta I)
      with model-based normalization constraints.
    """
    if options is None:
        options = MinVolNMFOptions()
    if np.min(X) < 0:
        raise ValueError("X should be nonnegative")

    m, n = X.shape
    rng = np.random.default_rng(options.random_state)
    tiny = options.epsilon

    # Initialization faithful to MATLAB: use SNPA when W,H not both provided
    if options.W is not None and options.H is not None:
        W = np.maximum(tiny, options.W.copy())
        H = np.maximum(tiny, options.H.copy())
        if W.shape != (m, r) or H.shape != (r, n):
            raise ValueError(f"W,H shapes are {W.shape},{H.shape}; expected {(m, r)}, {(r, n)}")
    else:
        # For model 1, use sum<=1 projection during SNPA
        snpa_proj = 1 if options.model == 1 else 0
        K, H0 = snpa_matlab(X, r, SNPAOptions(display=options.display, proj=snpa_proj))
        if len(K) < r:
            if options.display:
                print("SNPA recovered less than r basis vectors; reducing r.")
            r = len(K)
        W = np.maximum(tiny, X[:, K])
        # Refine H via NNLS with init = H0
        snpa_opts0 = SNPAOptions(maxitn=options.inneriter, proj=snpa_proj, display=0, init=H0)
        H = _nnls_fpgm(X, W, snpa_opts0)

    # Initial scaling
    for k in range(r):
        s = max(tiny, float(np.max(W[:, k])))
        W[:, k] /= s
        H[k, :] *= s

    # Normalize per model
    W, H = _normalize_wh(W, H, options.model, X)

    e_vals: list[float] = []
    t_vals: list[float] = []

    start = __import__("time").perf_counter()
    i = 1
    mintime = 0.1
    cntdis = 0

    def not_converged() -> bool:
        if i <= 2 or len(e_vals) <= 1:
            return True
        return abs(e_vals[-1] - e_vals[-2]) > options.accuracy * abs(e_vals[-1])

    # Initial errors and lambda' scaling
    normX2 = float(np.sum(X * X))
    WtW = W.T @ W
    WtX = W.T @ X
    err1 = [max(0.0, normX2 - 2 * float(np.sum(WtX * H)) + float(np.sum(WtW * (H @ H.T))))]
    err2 = [float(np.log(np.linalg.det(WtW + options.delta * np.eye(r)))) if np.linalg.det(WtW + options.delta * np.eye(r)) > 0 else 0.0]
    lam_eff = options.lam * max(1e-6, err1[0]) / max(1e-12, abs(err2[0]))
    e_vals.append(err1[0] + lam_eff * err2[0])

    while i <= options.maxiter and (__import__("time").perf_counter() - start) <= options.timemax and not_converged():
        # *** Update W via FGMqpnonneg ***
        XHt = X @ H.T
        HHt = H @ H.T
        Y = np.linalg.inv(W.T @ W + options.delta * np.eye(r))
        A = lam_eff * Y + HHt
        # Projection model for W
        proj_flag = 1 if (options.model <= 2 or options.model == 4) else 2  # 2 enforces column simplex
        W, _ = _fgm_qp_nonneg(A, XHt, W, options.inneriter, proj_flag)

        # *** Update H via FPGM NNLS ***
        # Set projection for H per model: 1->sum<=1, 2->rows sum=1, 4->sum=1
        proj_code = 0
        if options.model == 1:
            proj_code = 1
        elif options.model == 4:
            proj_code = 3
        elif options.model == 2:
            proj_code = 2
        snpa_opts = SNPAOptions(maxitn=options.inneriter, proj=proj_code, display=0, init=H)
        H = _nnls_fpgm(X, W, snpa_opts)

        # Errors
        WtW = W.T @ W
        WtX = W.T @ X
        e1 = max(0.0, normX2 - 2 * float(np.sum(WtX * H)) + float(np.sum(WtW * (H @ H.T))))
        e2 = float(np.log(np.linalg.det(WtW + options.delta * np.eye(r)))) if np.linalg.det(WtW + options.delta * np.eye(r)) > 0 else 0.0
        err1.append(e1)
        err2.append(e2)
        e_vals.append(e1 + lam_eff * e2)
        t_vals.append(__import__("time").perf_counter() - start)

        if options.display == 1 and t_vals[-1] >= mintime:
            print(f"{i}...", end="")
            mintime *= 2.0
            cntdis += 1
            if cntdis % 10 == 0:
                print()

        # Lambda tuning to approach target relative error
        if options.target is not None:
            rel = np.sqrt(e1) / (np.sqrt(normX2) + 1e-16)
            if rel > options.target + 1e-3:
                lam_eff *= 0.95
            elif rel < options.target - 1e-3:
                lam_eff *= 1.05

        i += 1

    if options.display == 1:
        print()

    return W, H, np.array(e_vals), np.array(t_vals)

