from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

from algorithms.separable_nmf import spa_matlab, SPAOptions


def _vertices(F: np.ndarray, g: np.ndarray) -> np.ndarray:
    """
    Compute polygon vertices of { x in R^2 | F x + g >= 0 } from halfspaces.
    F: (m x 2), g: (m,)
    Return P as 2 x p matrix with vertices in order (convex hull).
    """
    m = F.shape[0]
    pts = []
    # Intersections of all pairs of lines F_i x + g_i = 0
    for i in range(m):
        for j in range(i + 1, m):
            A = np.stack([F[i, :], F[j, :]], axis=0)  # 2x2
            b = -np.array([g[i], g[j]], dtype=float)
            try:
                x = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                continue
            # Check feasibility
            if np.all(F @ x + g >= -1e-10):
                pts.append(x)
    if not pts:
        return np.zeros((2, 0))
    P = np.stack(pts, axis=1)  # 2 x p'
    # Order via convex hull
    if P.shape[1] >= 3:
        hull = ConvexHull(P.T)
        ordered = P[:, hull.vertices]
    else:
        ordered = P
    return ordered


def npp_rank3_matrix(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Port of NPPrank3matrix.m:
    - Ensure rank(X)=3 and X>=0
    - Remove zero rows/cols, normalize columns to sum 1
    - Find U = X[:, K] via SPA (r=3), V = U\\X (least squares)
    - Build F,g and compute polygon vertices P
    Returns (P, U, V)
    """
    if np.linalg.matrix_rank(X) != 3 or np.min(X) < 0:
        raise ValueError("X must be rank 3 and nonnegative")
    # Remove zero rows/cols
    keep_rows = np.sum(X, axis=1) > 0
    keep_cols = np.sum(X, axis=0) > 0
    Xr = X[np.ix_(keep_rows, keep_cols)]
    # Normalize columns to sum 1
    Xr = Xr / (np.sum(Xr, axis=0, keepdims=True) + 1e-16)
    # SPA to choose 3 independent columns
    K = spa_matlab(Xr, 3, SPAOptions(normalize=0, precision=1e-6, display=0))
    U = Xr[:, K]
    # Solve V from U V = Xr (least squares)
    V = np.linalg.lstsq(U, Xr, rcond=None)[0]
    # Build F,g
    r = 3
    F = U[:, : r - 1] - U[:, [r - 1]]  # m x 2
    g = U[:, r - 1]  # m
    P = _vertices(F, g)
    # Plot
    if P.shape[1] >= 3:
        Kp = ConvexHull(P.T).vertices
        plt.figure()
        plt.plot(P[0, Kp], P[1, Kp], "rx-", markersize=6, label=r"Outer polygon $\mathcal{B}$")
        Kv = ConvexHull(V[:2, :].T).vertices if V.shape[0] >= 2 and V.shape[1] >= 3 else None
        if Kv is not None:
            plt.plot(V[0, Kv], V[1, Kv], "bo-.", markersize=4, label=r"Inner polygon $\mathcal{A}$")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.close()
    return P, U, V


if __name__ == "__main__":
    # Small test with a rank-3 nonnegative matrix can be added if needed
    pass

