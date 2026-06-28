from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class SPAOptions:
    normalize: int = 0  # 1 to L1-normalize columns of X
    precision: float = 1e-6
    display: int = 1


def spa_matlab(X: np.ndarray, r: int, options: Optional[SPAOptions] = None) -> List[int]:
    """
    Faithful port of SPA.m (Successive Projection Algorithm).
    Returns list K of selected column indices (0-based).
    """
    if options is None:
        options = SPAOptions()
    Xw = X.copy()
    m, n = Xw.shape
    if options.normalize == 1:
        D = 1.0 / (np.sum(Xw, axis=0) + 1e-16)
        Xw = Xw * D.reshape(1, -1)

    normX0 = np.sum(Xw * Xw, axis=0)
    nXmax = float(np.max(normX0))
    normR = normX0.copy()

    K: List[int] = []
    Ucols: List[np.ndarray] = []
    i = 1
    if options.display == 1:
        print("Extraction of the indices by SPA:")
    while i <= r and np.sqrt(float(np.max(normR)) / nXmax) > options.precision:
        a = float(np.max(normR))
        b_all = np.where((a - normR) / (a + 1e-16) <= 1e-6)[0]
        if b_all.size > 1:
            sub = normX0[b_all]
            ib = int(np.argmax(sub))
            b = int(b_all[ib])
        else:
            b = int(b_all[0])
        K.append(b)
        u = Xw[:, b].copy()
        # Orthogonal projection against previously selected normalized columns
        for j in range(len(Ucols)):
            uj = Ucols[j]
            u = u - uj * float(uj @ u)
        # Normalize u
        nu = float(np.linalg.norm(u)) + 1e-16
        u = u / nu
        Ucols.append(u)
        # Update residual norms: ||r_k||^2 = ||r_{k-1}||^2 - (u^T x_k)^2 for all k
        normR = normR - (u @ Xw) ** 2
        if options.display == 1:
            print(f"{i}...", end="")
            if i % 10 == 0:
                print()
        i += 1
    if options.display == 1:
        print()
    return K

