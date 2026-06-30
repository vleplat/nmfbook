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

from algorithms.onmf import onmf, ONMFOptions
from algorithms.separable_nmf.spa_matlab import spa_matlab, SPAOptions
from utils.affichage import affichage


def main():
    data_path = os.path.join(BASE, "data sets", "Urban.mat")
    import scipy.io as sio
    mat = sio.loadmat(data_path)
    X = np.array(mat["X"], dtype=float)  # Urban scripts do not transpose
    r = 6
    # SPA init
    K = spa_matlab(X, r, SPAOptions(display=0))
    W0 = X[:, K]
    # Run alternating ONMF
    W, H, e, _ = onmf(X, r, ONMFOptions(W=W0, display=1))
    figs = os.path.join(BASE, "figs")
    os.makedirs(figs, exist_ok=True)
    # Display abundance maps: rows of H reshaped 307x307; MATLAB reorders [2 1 6 3 5 4]
    order = [1, 0, 5, 2, 4, 3]
    affichage(H[order, :].T, (307, 307), ncols=3, suptitle="Orthogonal NMF on Urban", save_path=os.path.join(figs, "ch4_onmf_urban_maps.pdf"))
    # Error vs iterations
    plt.figure()
    plt.plot(e)
    plt.xlabel("Iterations")
    plt.ylabel(r"Relative error: $\|X-WH\|_F / \|X\|_F$")
    plt.title("Orthogonal NMF on Urban")
    plt.tight_layout()
    plt.savefig(os.path.join(figs, "ch4_onmf_urban_error.pdf"), bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()

