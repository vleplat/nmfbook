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

from algorithms.hierarchical_nmf import hierclust2nmf, HierNMFOptions
from utils.affichage import affichage


def main():
    data_path = os.path.join(BASE, "data sets", "Urban.mat")
    mat = sio.loadmat(data_path)
    # Common variable names
    X = None
    for key in ("X", "M", "data"):
        if key in mat:
            X = np.array(mat[key], dtype=np.float64)
            break
    if X is None:
        raise KeyError("Urban.mat must contain 'X' (or 'M', 'data')")
    X = np.maximum(X, 0.0)

    figs = os.path.join(BASE, "figs")
    os.makedirs(figs, exist_ok=True)

    r = 6
    W, H = hierclust2nmf(X, r, HierNMFOptions(display=1))

    # Plot basis spectra (columns of W)
    plt.figure(figsize=(8, 6))
    for k in range(r):
        plt.plot(W[:, k], label=f"comp {k+1}")
    plt.xlabel("wavelength index")
    plt.ylabel("amplitude")
    plt.title("Urban - hierarchical NMF basis spectra")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(figs, "urban_basis_spectra.pdf"), bbox_inches="tight")
    plt.close()

    # Display abundance maps with affichage, matching MATLAB (perrow=3, 307x307)
    perrow = 3
    Li = 307
    Co = 307
    # H has shape (r, n); affichage expects columns as images: use H.T (n, r)
    affichage(
        H.T,
        (Li, Co),
        ncols=perrow,
        suptitle="Urban - abundance maps (rows of H)",
        save_path=os.path.join(figs, "urban_abundances_maps.pdf"),
    )

    print("Saved Urban figures in", figs)


if __name__ == "__main__":
    main()

