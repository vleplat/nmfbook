from __future__ import annotations

import os
import sys
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")

BASE = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from algorithms.nmf import fro_nmf, FroNMFOptions
from algorithms.sparse_nmf import sparse_nmf, SparseNMFOptions
from utils.affichage import affichage


def main():
    data_path = os.path.join(BASE, "data sets", "CBCL.mat")
    mat = sio.loadmat(data_path)
    X = np.array(mat["X"], dtype=np.float64)
    r = 25
    r1 = 49
    opts = FroNMFOptions(maxiter=500, timemax=float("inf"), display=1)
    # First layer X ≈ W H
    W, H, *_ = fro_nmf(X, r, opts)
    # Scale columns/rows so ||W[:,k]|| = ||H[k,:]||
    normW = np.sqrt(np.sum(W * W, axis=0)) + 1e-16
    normH = np.sqrt(np.sum(H.T * H.T, axis=0)) + 1e-16
    for k in range(r):
        W[:, k] = W[:, k] / np.sqrt(normW[k]) * np.sqrt(normH[k])
        H[k, :] = H[k, :] / np.sqrt(normH[k]) * np.sqrt(normW[k])
    # Second layer: W ≈ W1 H1 with sparse NMF (sW=0.85)
    opts2 = SparseNMFOptions(maxiter=500, timemax=float("inf"), display=1, sW=0.85, colproj=1)
    W1, H1, *_ = sparse_nmf(W, r1, opts2)
    figs = os.path.join(BASE, "figs")
    os.makedirs(figs, exist_ok=True)
    affichage(W1, (19, 19), ncols=7, suptitle=r"$W_1$ in triNMF $W_1 W_2 H \approx X$",
              save_path=os.path.join(figs, "cbcl_trinmf_W1.pdf"))
    affichage(W, (19, 19), ncols=5, suptitle=r"$W_1 W_2$ in triNMF $W_1 W_2 H \approx X$",
              save_path=os.path.join(figs, "cbcl_trinmf_W1W2.pdf"))
    print("Saved CBCL triNMF figures in", figs)


if __name__ == "__main__":
    main()

