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
    data_path = os.path.join(BASE, "data sets", "Urban.mat")
    mat = sio.loadmat(data_path)
    X = None
    for key in ("X", "M", "data"):
        if key in mat:
            X = np.array(mat[key], dtype=np.float64)
            break
    if X is None:
        raise KeyError("Urban.mat must contain 'X' (or 'M','data')")
    r = 6
    # SPA init on X'
    K = spa_matlab(X.T, r, SPAOptions(display=0))
    H0 = X[K, :].T  # pixels x r, used as W initialization in Python solver
    opts = ProjectiveNMFOptions(maxiter=500, display=1, W=H0)
    # Run on X' so W has shape (pixels, r), matching MATLAB display
    Wimg, Htmp, (e_vals, t_vals) = projective_nmf(X.T, r, opts)
    figs = os.path.join(BASE, "figs")
    os.makedirs(figs, exist_ok=True)
    affichage(Wimg, (307, 307), ncols=3, suptitle="Solution of projective NMF",
              save_path=os.path.join(figs, "projective_urban_maps.pdf"))
    plt.figure()
    plt.plot(e_vals / (np.linalg.norm(X.T, ord="fro") + 1e-16))
    plt.xlabel("Iterations")
    plt.ylabel(r"$\| X - W W^\top X \|_F / \|X\|_F$")
    plt.title("projective NMF")
    plt.tight_layout()
    plt.savefig(os.path.join(figs, "projective_urban_error.pdf"), bbox_inches="tight")
    plt.close()
    print("Saved ProjectiveNMF Urban figures in", figs)


if __name__ == "__main__":
    main()

