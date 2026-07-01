from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from algorithms.separable_nmf.spa_matlab import spa_matlab, SPAOptions
from algorithms.separable_nmf.snpa_matlab import _nnls_fpgm, SNPAOptions
from utils.affichage import affichage
from utils.silence_warnings import silence_numpy_warnings


def main():
    silence_numpy_warnings()
    # Separable NMF on CBCL (Figure 7.2)
    data_path = os.path.join(BASE, "data sets", "CBCL.mat")
    import scipy.io as sio
    mat = sio.loadmat(data_path)
    X = np.array(mat["X"], dtype=float)  # CBCL stored as 361 x n here
    r = 49

    # SPA on X^T with column L1 normalization (as in MATLAB option normalize=1)
    K = spa_matlab(X.T, r, SPAOptions(normalize=1, display=1))

    # H = X(K,:) in MATLAB (rows of X indexed by K)
    H = X[K, :]  # shape r x n

    # W = NNLS(H', X')'  -> solve min_{W>=0} ||X - W H||_F^2
    # Implement via FPGM NNLS on the transposed system:
    #   Solve Y = argmin ||X^T - (H^T) Y||_F^2, Y >= 0, then W = Y^T
    Y = _nnls_fpgm(M=X.T, W=H.T, options=SNPAOptions(maxitn=300, normalize=0, proj=0, display=0))
    W = Y.T  # m x r

    # Display basis images W (7-by-19-by-19)
    figs = os.path.join(BASE, "figs")
    os.makedirs(figs, exist_ok=True)
    affichage(W, (19, 19), ncols=7, suptitle="Separable NMF (SPA init) on CBCL", save_path=os.path.join(figs, "ch7_cbcl_sepNMF_basis.pdf"))


if __name__ == "__main__":
    main()

