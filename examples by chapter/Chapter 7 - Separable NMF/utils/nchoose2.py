from __future__ import annotations

import numpy as np


def nchoose2(r: int) -> np.ndarray:
    """
    Generate an r-by-N matrix whose columns have exactly two ones in distinct rows.
    N = r*(r-1)/2.
    """
    R = r * (r - 1) // 2
    H = np.zeros((r, R), dtype=float)
    k = 0
    for i in range(r):
        for j in range(i + 1, r):
            H[i, k] = 1.0
            H[j, k] = 1.0
            k += 1
    return H

