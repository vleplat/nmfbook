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
from algorithms.onmf import onmf, ONMFOptions
from algorithms.minvol_nmf import minvol_nmf, MinVolNMFOptions
from algorithms.nmu.recursive import recursive_nmu, RecursiveNMUOptions
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

    # 3) ONMF (sensitive to init) — default: SNPA-based init
    # Use SNPA coefficients to build a nonnegative W0
    W0 = np.maximum(1e-12, X_alg @ Hsnpa.T)  # m x r
    # Normalize columns
    W0 = W0 / (np.linalg.norm(W0, axis=0, keepdims=True) + 1e-16)
    Wonmf, Honmf, *_ = onmf(X_alg, r, ONMFOptions(W=W0, display=0, timemax=10.0))
    affichage(Honmf.T, (h, w), ncols=17, suptitle="Basis images with ONMF", save_path=os.path.join(figs, "swimmer_basis_onmf.pdf"))

    # 4) Min-volume NMF
    Wminv, Hminv, *_ = minvol_nmf(X_alg, r, MinVolNMFOptions(W=W0.copy(), H=Hsnpa.copy(), display=0, timemax=30.0))
    affichage(Hminv.T, (h, w), ncols=17, suptitle="Basis images with min-vol NMF", save_path=os.path.join(figs, "swimmer_basis_minvol.pdf"))

    # 5) NMU (recursive)
    Wnmu, Hnmu = recursive_nmu(X_alg, r, RecursiveNMUOptions(Cnorm=1, maxiter=200, display=1))
    affichage(Hnmu.T, (h, w), ncols=17, suptitle="Basis images with NMU", save_path=os.path.join(figs, "swimmer_basis_nmu.pdf"))

    print("Saved swimmer figures in", figs)


if __name__ == "__main__":
    main()

