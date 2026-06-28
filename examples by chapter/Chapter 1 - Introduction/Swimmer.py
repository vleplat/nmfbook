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

from algorithms.nmf import fro_nmf, FroNMFOptions
from algorithms.separable_nmf import snpa_matlab, SNPAOptions, solve_h_given_indices
from utils.affichage import affichage


def main():
    data_path = os.path.join(BASE, "data sets", "Swimmer.mat")
    mat = sio.loadmat(data_path)
    # Try common variable names
    X = None
    for key in ("X", "swimmer", "data"):
        if key in mat:
            X = np.array(mat[key], dtype=np.float64)
            break
    if X is None:
        raise KeyError("Swimmer.mat must contain variable 'X' (or 'swimmer', 'data')")
    X = np.maximum(X, 0.0)
    X_alg = X.T  # Match MATLAB: X = X'

    figs = os.path.join(BASE, "figs")
    os.makedirs(figs, exist_ok=True)

    # Image shape per MATLAB script (20 x 11)
    p, n = X.shape
    h, w = 20, 11

    # Show sample images
    affichage(X[:, :32], (h, w), ncols=8, suptitle="Swimmer samples", save_path=os.path.join(figs, "swimmer_samples.pdf"))

    # Match MATLAB: r = 17
    r = 17
    # 1) Standard FroNMF
    np.random.seed(0)
    Wnmf, Hnmf, *_ = fro_nmf(X_alg, r, FroNMFOptions())
    affichage(Hnmf.T, (h, w), ncols=17, suptitle="Basis images with standard NMF", save_path=os.path.join(figs, "swimmer_basis_nmf.pdf"))

    # 2) SNPA (separable NMF) as in MATLAB
    K, Hsnpa = snpa_matlab(X_alg, r, SNPAOptions(maxitn=200, normalize=0, proj=0, relerr=1e-6, display=1))
    affichage(Hsnpa.T, (h, w), ncols=17, suptitle="Basis images with separable NMF (SNPA)", save_path=os.path.join(figs, "swimmer_basis_snpa.pdf"))

    print("Saved swimmer figures in", figs)


if __name__ == "__main__":
    main()

