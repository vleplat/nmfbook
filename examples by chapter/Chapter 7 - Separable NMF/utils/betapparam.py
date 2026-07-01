from __future__ import annotations

import numpy as np

from algorithms.separable_nmf.snpa_matlab import snpa_matlab, _nnls_fpgm, SNPAOptions


def _minnormdist(W: np.ndarray) -> float:
    """
    min( min_j ||W[:,j]||_2 , min_{i<j} ||W[:,i]-W[:,j]||_2 / sqrt(2) )
    """
    m, r = W.shape
    mins = [np.linalg.norm(W[:, j]) for j in range(r)]
    for i in range(r):
        wi = W[:, i]
        for j in range(i + 1, r):
            wj = W[:, j]
            mins.append(np.linalg.norm(wi - wj) / np.sqrt(2.0))
    return float(np.min(mins))


def betapparam(W: np.ndarray) -> float:
    """
    Python port of betapparam.m.
    Reorders columns of W using SNPA, then computes:
      betap = min over i=1..r-1 of
              min( min_j ||W_i(:,j)||_2 , min_{p<q} ||W_i(:,p)-W_i(:,q)||_2 / sqrt(2) )
    where W_i = [W(:,i+1:r) - W(:,1:i) * Y_i], and
          Y_i solves min_{Y>=0} ||W(:,i+1:r) - W(:,1:i) Y||_F.
    """
    m, r = W.shape
    # Order columns by SNPA extraction
    K, _ = snpa_matlab(W, r, SNPAOptions(display=0))
    Wk = W[:, K]
    betap = _minnormdist(Wk)
    for i in range(r - 1):
        A = Wk[:, i + 1 : r]   # m x (r-i-1)
        B = Wk[:, : i + 1]     # m x (i+1)
        # Solve A ≈ B Y with Y >= 0
        Y = _nnls_fpgm(A, B, SNPAOptions(maxitn=200, proj=0, display=0))
        Wi = A - (B @ Y)
        betap = min(betap, _minnormdist(Wi))
    return float(betap)

