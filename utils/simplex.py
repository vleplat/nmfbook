"""
Simplex projection utilities.

This module provides:
- simplex_proj: projection onto the set { x | x >= 0 and sum(x) <= 1 }
- simplex_col_proj: column-wise projection onto the probability simplex

References:
  Wang, W., & Carreira-Perpinan, M. A. (2013). Projection onto the probability simplex:
  An efficient algorithm with a simple proof, and an application. arXiv:1309.1541
"""

from __future__ import annotations

import numpy as np


def simplex_col_proj(Y: np.ndarray) -> np.ndarray:
    """
    Project each column of Y onto the probability simplex
      Delta = { x in R^d | x >= 0, sum(x) = 1 }.

    Parameters
    ----------
    Y : np.ndarray
        Array of shape (d, n). Each column is projected independently.

    Returns
    -------
    np.ndarray
        Array X of shape (d, n) with columns projected onto the simplex.
    """
    if Y.ndim == 1:
        y = Y.astype(float, copy=True)
        u = np.sort(y)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, u.size + 1) > (cssv - 1))[0]
        if rho.size == 0:
            theta = 0.0
        else:
            rho = rho[-1]
            theta = (cssv[rho] - 1.0) / float(rho + 1)
        x = np.maximum(y - theta, 0.0)
        return x

    if Y.ndim != 2:
        raise ValueError("Y must be a 1D or 2D array")

    d, n = Y.shape
    X = np.empty_like(Y, dtype=float)

    # Vectorized implementation across columns
    # Sort each column in descending order
    U = np.sort(Y, axis=0)[::-1]
    cssv = np.cumsum(U, axis=0)
    j = np.arange(1, d + 1).reshape(-1, 1)
    cond = U * j > (cssv - 1.0)
    # For each column, find last index where condition holds
    rho = cond[::-1].argmax(axis=0)
    rho = (d - 1) - rho
    theta = (cssv[rho, np.arange(n)] - 1.0) / (rho + 1.0)
    X = np.maximum(Y - theta.reshape(1, -1), 0.0)
    return X


def simplex_proj(y: np.ndarray) -> np.ndarray:
    """
    Project vector or columns of a matrix onto S = { x | x >= 0, sum(x) <= 1 }.

    If y is a matrix (2D), projects each column independently.

    Parameters
    ----------
    y : np.ndarray
        1D array (vector) or 2D array (project columns) to be projected.

    Returns
    -------
    np.ndarray
        Projection of y onto S.
    """
    x = np.maximum(y, 0.0)
    if x.ndim == 1:
        s = x.sum()
        if s > 1.0:
            return simplex_col_proj(x)
        return np.minimum(x, 1.0)

    if x.ndim != 2:
        raise ValueError("y must be a 1D or 2D array")

    col_sums = x.sum(axis=0)
    over = col_sums > 1.0
    if not np.any(over):
        # Clip each column to at most 1 entry-wise, matching MATLAB behavior
        return np.minimum(x, 1.0)

    X = x.copy()
    if np.any(over):
        X[:, over] = simplex_col_proj(X[:, over])
    # For non-over columns, clip to 1
    if np.any(~over):
        X[:, ~over] = np.minimum(X[:, ~over], 1.0)
    return X


