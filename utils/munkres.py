from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def munkres(cost_matrix: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Hungarian algorithm wrapper (a.k.a. 'munkres') using SciPy.

    Parameters
    ----------
    cost_matrix : np.ndarray
        2D array of shape (m, n) with costs.

    Returns
    -------
    assignment : np.ndarray
        Array of length m, assignment[j] = assigned column index for row j, or -1 if unassigned
        (when m != n).
    total_cost : float
        Sum of costs of the assigned pairs.
    """
    if cost_matrix.ndim != 2:
        raise ValueError("cost_matrix must be 2D")
    m, n = cost_matrix.shape
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    assignment = -np.ones(m, dtype=int)
    assignment[row_ind] = col_ind
    total_cost = float(cost_matrix[row_ind, col_ind].sum())
    return assignment, total_cost

