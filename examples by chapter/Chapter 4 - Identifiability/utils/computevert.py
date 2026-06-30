from __future__ import annotations

import itertools
import numpy as np


def computevert(A: np.ndarray) -> np.ndarray:
    """
    Compute vertices V (columns) of the polytope { x | e^T x = 1, A x >= 0 }.
    Faithful to computevert.m (enumerates combinations).
    """
    m, r = A.shape
    # choose r-1 rows
    combos = list(itertools.combinations(range(m), r - 1))
    lT = len(combos)
    if lT > 1e6:
        # too large
        return np.zeros((r, 0))
    V = []
    e = np.ones((1, r))
    b = np.zeros((r, 1)); b[-1, 0] = 1.0
    for idxs in combos:
        S = np.vstack([A[list(idxs), :], e])
        if np.linalg.matrix_rank(S) == r:
            try:
                x = np.linalg.solve(S, b)
            except np.linalg.LinAlgError:
                continue
            if np.min(A @ x) >= -1e-6:
                x = x.reshape(-1)
                if not V:
                    V.append(x)
                else:
                    M = np.column_stack(V)
                    if np.min(np.sum((M - x.reshape(-1, 1)) ** 2, axis=0)) > 1e-6:
                        V.append(x)
    if not V:
        return np.zeros((r, 0))
    return np.column_stack(V)

