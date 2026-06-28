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
from algorithms.projective_nmf import projective_nmf, ProjectiveNMFOptions
from utils.affichage import affichage


def main():
    data_path = os.path.join(BASE, "data sets", "CBCL.mat")
    mat = sio.loadmat(data_path)
    X = np.array(mat["X"], dtype=np.float64)
    m, n = X.shape
    r = 49

    # SPA init
    K = spa_matlab(X, r, SPAOptions(normalize=0, display=0))
    W0 = X[:, K]
    # Projective NMF with W initialized, many iterations
    opts = ProjectiveNMFOptions(maxiter=2000, W=W0, display=1)
    W, H, (e_vals, t_vals) = projective_nmf(X, r, opts)

    figs = os.path.join(BASE, "figs")
    os.makedirs(figs, exist_ok=True)
    # Show initialization and solution basis
    affichage(W0, (19, 19), ncols=7, suptitle="Initialization with separable NMF",
              save_path=os.path.join(figs, "cbcl_projective_init_spa.pdf"))
    affichage(W, (19, 19), ncols=7, suptitle="Solution of projective NMF",
              save_path=os.path.join(figs, "cbcl_projective_basis.pdf"))
    # Error vs iterations
    plt.figure()
    plt.plot(e_vals)
    plt.xlabel("Iterations")
    plt.ylabel(r"$\|X - WW^\top X\|_F$")
    plt.tight_layout()
    plt.savefig(os.path.join(figs, "cbcl_projective_error.pdf"), bbox_inches="tight")
    plt.close()
    print("Saved CBCL projective NMF figures in", figs)


if __name__ == "__main__":
    main()

