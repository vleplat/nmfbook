from __future__ import annotations

import numpy as np
from typing import Tuple

from .spa_matlab import spa_matlab, SPAOptions
from .snpa_matlab import _nnls_fpgm, SNPAOptions


def septrisym_nmf(A: np.ndarray, r: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Heuristic separable tri-symmetric NMF:
      Given symmetric A ≈ W S W^T with separable W and symmetric S,
      recover (W, S) using SPA and NNLS.
    Returns (W_hat, S_hat).
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square (symmetric).")
    m = A.shape[0]
    # 1) Select r anchor columns using SPA on A (L1-normalization helps)
    K = spa_matlab(A, r, SPAOptions(normalize=1, display=0))
    Wcand = np.maximum(0.0, A[:, K])  # m x r
    # 2) Normalize columns (separable W columns sum to 1 in the example)
    # Optional: take W as nonnegative part of Wcand (columns L1-normalized)
    W = np.maximum(0.0, Wcand)
    col_sums = np.sum(W, axis=0, keepdims=True) + 1e-16
    W = W / col_sums
    # 3) Closed-form S minimizing ||A - W S W^T||_F
    G = W.T @ W
    # Regularize in case of ill-conditioning
    reg = 1e-12 * np.eye(r)
    Ginv = np.linalg.pinv(G + reg)
    S = Ginv @ (W.T @ A @ W) @ Ginv
    S = 0.5 * (S + S.T)
    return W, S

