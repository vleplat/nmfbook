from __future__ import annotations

import os
import sys
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from algorithms.separable_nmf import spa_matlab, SPAOptions
from algorithms.sparse_nmf import sparse_nmf, SparseNMFOptions
from utils.affichage import affichage


def nnls_w_given_h(X: np.ndarray, H: np.ndarray) -> np.ndarray:
    # Solve min_W ||X - W H||_F^2 with W>=0 via normal equations + clip
    HHT = H @ H.T
    try:
        inv = np.linalg.inv(HHT + 1e-8 * np.eye(HHT.shape[0]))
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(HHT)
    W = (X @ H.T) @ inv
    return np.maximum(0.0, W)


def main():
    data_path = os.path.join(BASE, "data sets", "CBCL.mat")
    mat = sio.loadmat(data_path)
    X = np.array(mat["X"], dtype=np.float64)
    m, n = X.shape
    r = 49
    # SPA init on X'
    K = spa_matlab(X.T, r, SPAOptions(display=0))
    H0 = X[K, :]  # r x n
    # For X = m x n, want W0 m x r s.t. X ≈ W0 H0
    W0 = nnls_w_given_h(X, H0)
    # Baseline NMF (sW=0)
    figs = os.path.join(BASE, "figs")
    os.makedirs(figs, exist_ok=True)
    opts = SparseNMFOptions(W=W0, H=H0, maxiter=100, timemax=float("inf"), display=1, sW=None)
    W, H, e, t = sparse_nmf(X, r, opts)
    # Sparse NMF starting from NMF solution
    opts2 = SparseNMFOptions(W=W, H=H, maxiter=100, timemax=float("inf"), display=1, sW=0.85, colproj=1)
    Ws, Hs, es, ts = sparse_nmf(X, r, opts2)
    # Display
    affichage(W, (19, 19), ncols=7, suptitle="NMF", save_path=os.path.join(figs, "cbcl_sparse_nmf_basis.pdf"))
    affichage(Ws, (19, 19), ncols=7, suptitle="sparse NMF (0.85)",
              save_path=os.path.join(figs, "cbcl_sparse_nmf_basis_sparse.pdf"))
    print("Saved CBCL sparse NMF figures in", figs)


if __name__ == "__main__":
    main()

