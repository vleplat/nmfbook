from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from utils.simplex import simplex_proj


@dataclass
class SNPAOptions:
    maxitn: int = 200
    normalize: int = 0  # 1 to L1-normalize columns of X
    proj: int = 0       # 1 to enforce sum<=1, >=0; 0 only >=0
    relerr: float = 1e-6
    display: int = 1
    init: Optional[np.ndarray] = None  # optional initial H


def _nnls_fpgm(M: np.ndarray, W: np.ndarray, options: SNPAOptions) -> np.ndarray:
    """
    Solve min_Y 0.5||M - W Y||_F^2 with
      - proj=0: Y >= 0
      - proj=1: columns of Y in { y>=0, sum(y) <= 1 }
    via Fast Projected Gradient Method (Nesterov).
    """
    m, n = M.shape
    r = W.shape[1]
    # Lipschitz constant L = ||W^T W||_2
    WT = W.T
    WTW = WT @ W
    # Power iteration for spectral norm
    v = np.random.default_rng(0).random(r)
    for _ in range(20):
        v = WTW @ v
        nv = np.linalg.norm(v)
        if nv == 0:
            break
        v /= nv
    L = float(np.linalg.norm(WTW @ v))
    if not np.isfinite(L) or L <= 0:
        L = 1.0

    if options.init is not None and options.init.shape == (r, n):
        Y = np.maximum(0.0, options.init.copy())
    else:
        Y = np.zeros((r, n), dtype=float)
    Z = Y.copy()
    t = 1.0
    step = 1.0 / L

    for _ in range(max(1, options.maxitn)):
        # Gradient at Z
        R = (W @ Z) - M  # m x n
        G = WT @ R       # r x n
        Y_next = Z - step * G
        # Projection
        if options.proj == 1:
            # project each column onto {y>=0, sum(y)<=1}
            for j in range(n):
                Y_next[:, j] = simplex_proj(Y_next[:, j])
        else:
            Y_next = np.maximum(0.0, Y_next)
        # Nesterov
        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        Z = Y_next + ((t - 1.0) / t_next) * (Y_next - Y)
        Y = Y_next
        t = t_next
    return Y


def snpa_matlab(X: np.ndarray, r: int, options: Optional[SNPAOptions] = None) -> Tuple[list[int], np.ndarray]:
    """
    Faithful port of SNPA.m (variant with f(.) = ||.||^2).
    Returns:
      - K (list of selected column indices, 0-based)
      - H (r x n) coefficients
    """
    if options is None:
        options = SNPAOptions()
    m, n = X.shape
    Xw = X.copy()
    if options.normalize == 1:
        col_sums = Xw.sum(axis=0, keepdims=True) + 1e-16
        Xw = Xw / col_sums

    normX0 = np.sum(Xw * Xw, axis=0)        # (n,)
    nXmax = float(np.max(normX0))
    normR = normX0.copy()
    XtUK = np.empty((0, n), dtype=float)    # will grow by rows (i added each step)
    UKtUK = np.empty((0, 0), dtype=float)   # will grow as (i x i)
    K: list[int] = []
    U_cols: list[np.ndarray] = []
    i = 1
    # Main loop
    if options.display == 1:
        print("Extraction of the indices by SNPA:")
    H = None
    while i <= r and np.sqrt(float(np.max(normR)) / nXmax) > options.relerr:
        a = float(np.max(normR))
        b_all = np.where((a - normR) / (a + 1e-16) <= 1e-6)[0]
        if b_all.size > 1:
            # tie break using normX0
            sub = normX0[b_all]
            ib = int(np.argmax(sub))
            b = int(b_all[ib])
        else:
            b = int(b_all[0])
        K.append(b)
        u_i = Xw[:, b]
        U_cols.append(u_i)
        # Update XtUK and UKtUK
        XtUK = np.vstack([XtUK, Xw.T @ u_i])  # shape (i, n)
        if i == 1:
            UKtUK = np.array([[float(u_i @ u_i)]], dtype=float)
        else:
            UtUi = np.array([uc @ u_i for uc in U_cols[:-1]], dtype=float)
            UKtUK = np.block([
                [UKtUK, UtUi.reshape(-1, 1)],
                [UtUi.reshape(1, -1), np.array([[float(u_i @ u_i)]])]
            ])
        # Update H
        W = np.column_stack(U_cols)  # m x i
        snpa_opts = SNPAOptions(
            maxitn=options.maxitn,
            normalize=0,
            proj=options.proj,
            relerr=options.relerr,
            display=0,
            init=None if H is None else H
        )
        if H is not None:
            # expand H by adding a new row initialized to one-hot on selected b
            h_new = np.zeros((1, n), dtype=float)
            h_new[0, b] = 1.0
            H = np.vstack([H, h_new])
            snpa_opts.init = H
        H = _nnls_fpgm(Xw, W, snpa_opts)
        # Update normR efficiently
        if i == 1:
            normR = normX0 - 2.0 * (XtUK * H).sum(axis=0) + (H * (UKtUK @ H)).sum(axis=0)
        else:
            normR = normX0 - 2.0 * (XtUK * H).sum(axis=0) + (H * (UKtUK @ H)).sum(axis=0)

        if options.display == 1:
            print(f"{i}...", end="")
            if i % 10 == 0:
                print()
        i += 1
    if options.display == 1:
        print()
    return K, H

