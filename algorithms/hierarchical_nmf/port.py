from __future__ import annotations

from typing import List, Tuple

import numpy as np


def _fastsvds(M: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Simple wrapper using full SVD
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    return U[:, :k], s[:k], Vt[:k, :]


def _spa_two(A: np.ndarray) -> List[int]:
    """
    Minimal SPA for selecting 2 columns from A (d x n), following SPA.m logic.
    """
    X = A.copy()
    normX0 = np.sum(X * X, axis=0)
    normR = normX0.copy()
    K: List[int] = []
    Ucols: List[np.ndarray] = []
    for i in range(2):
        a = float(np.max(normR))
        b_all = np.where((a - normR) / (a + 1e-16) <= 1e-6)[0]
        if b_all.size > 1:
            sub = normX0[b_all]
            ib = int(np.argmax(sub))
            b = int(b_all[ib])
        else:
            b = int(b_all[0])
        K.append(b)
        u = X[:, b].copy()
        for j in range(i):
            uj = Ucols[j]
            u = u - uj * float(uj @ u)
        nrm = float(np.linalg.norm(u)) + 1e-16
        u = u / nrm
        Ucols.append(u)
        # Update residual norms
        normR = normR - (u @ X) ** 2
    return K


def _anls_entry_rank2_precompute_opt(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    # left: 2x2 or 1x1; right: n x 2 or n x 1; return H (2 x n)
    if left.size == 1:
        H = np.maximum(0.0, (right / float(left)).T)
        return H
    # Unconstrained least squares
    H = np.linalg.lstsq(left, right.T, rcond=None)[0].T  # n x 2
    # Enforce nonnegativity per row if needed
    use_either = ~(H >= 0.0).all(axis=1)
    if np.any(use_either):
        he = np.zeros((use_either.sum(), 2), dtype=float)
        he[:, 0] = np.maximum(0.0, right[use_either, 0] / left[0, 0])
        he[:, 1] = np.maximum(0.0, right[use_either, 1] / left[1, 1])
        cosine_either = he * np.array([np.sqrt(left[0, 0]), np.sqrt(left[1, 1])])[None, :]
        choose_first = cosine_either[:, 0] >= cosine_either[:, 1]
        he[choose_first, 1] = 0.0
        he[~choose_first, 0] = 0.0
        H[use_either, :] = he
    return H.T  # 2 x n


def _fquad(x: np.ndarray, s: float = 0.01) -> float:
    # Evaluate over delta in [0,1]
    n = x.size
    delta = np.arange(0.0, 1.0 + 1e-12, s)
    gs = 0.05
    fdel = np.array([(x <= d).sum() / n for d in delta], dtype=float)
    fdelp = np.zeros_like(fdel)
    if fdel.size >= 2:
        fdelp[0] = (fdel[1] - fdel[0]) / s
    for i in range(2, fdel.size):
        fdelp[i - 1] = (fdel[i] - fdel[i - 2]) / (2 * s)
    if fdel.size >= 2:
        fdelp[-1] = (fdel[-1] - fdel[-2]) / s
    finter = np.zeros_like(fdel)
    for i, d in enumerate(delta):
        deltahigh = min(1.0, d + gs)
        deltalow = max(0.0, d - gs)
        finter[i] = ((x <= deltahigh).sum() - (x < deltalow).sum()) / n / max(1e-12, (deltahigh - deltalow))
    fobj = -np.log(np.clip(fdel * (1.0 - fdel), 1e-12, None)) + np.exp(finter)
    thres = float(delta[int(np.argmin(fobj))])
    return thres


def reprvec_port(M: np.ndarray) -> Tuple[np.ndarray, float]:
    # Best rank-1 approx, then choose representative column
    U, s, Vt = _fastsvds(M, 1)
    u = np.abs(U[:, 0])
    m, n = M.shape
    u0 = u - float(np.mean(u))
    Mm = M - np.mean(M, axis=0, keepdims=True)
    # cosine errors
    denom = np.linalg.norm(u0) * (np.sqrt(np.sum(Mm * Mm, axis=0)) + 1e-16)
    cosang = (Mm.T @ u0) / np.maximum(denom, 1e-16)
    cosang = np.clip(cosang, -1.0, 1.0)
    err = np.arccos(cosang)
    b = int(np.argmin(err))
    return M[:, b], float(s[0])


def rank2nmf_port(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    m, n = M.shape
    if min(m, n) == 1:
        U, s, Vt = _fastsvds(M, 1)
        U = np.abs(U[:, :1])
        V = np.abs(Vt[:1, :])
        return np.hstack([U, U]), np.vstack([V, V]), float(s[0])
    U2, s, Vt2 = _fastsvds(M, 2)
    # K via SPA on S*V'
    S2 = np.diag(s[:2])
    SVt = S2 @ Vt2[:2, :]  # 2 x n
    K = _spa_two(SVt)
    U = np.zeros((M.shape[0], 2), dtype=float)
    if len(K) >= 1:
        U[:, 0] = np.maximum(0.0, (U2 @ S2 @ Vt2)[..., K[0]])
    if len(K) >= 2:
        U[:, 1] = np.maximum(0.0, (U2 @ S2 @ Vt2)[..., K[1]])
    # V via ANLS entry-wise closed form
    left = U.T @ U
    right = M.T @ U  # n x 2
    V = _anls_entry_rank2_precompute_opt(left, right)  # 2 x n
    return U, V, float(s[0])


def splitclust_port(M: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray, float]:
    U, V, s = rank2nmf_port(M)
    # Normalize columns of V to sum to one
    col_sums = V.sum(axis=0, keepdims=True) + 1e-16
    Vn = V / col_sums
    x = Vn[0, :]
    th = _fquad(x)
    K1 = np.where(x >= th)[0]
    K2 = np.where(x < th)[0]
    return [K1, K2], U, s

