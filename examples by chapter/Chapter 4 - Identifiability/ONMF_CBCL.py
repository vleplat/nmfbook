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

from algorithms.onmf import onmf, ONMFOptions  # faithful alternatingONMF port
from algorithms.separable_nmf.spa_matlab import spa_matlab, SPAOptions
from utils.affichage import affichage
from utils.silence_warnings import silence_numpy_warnings


def main():
    silence_numpy_warnings()
    data_path = os.path.join(BASE, "data sets", "CBCL.mat")
    import scipy.io as sio
    mat = sio.loadmat(data_path)
    X = np.array(mat["X"], dtype=float).T  # X = X' in MATLAB
    r = 49
    # SPA init on X
    K = spa_matlab(X, r, SPAOptions(display=0))
    W0 = X[:, K]
    # Run alternating ONMF
    W, H, e, _ = onmf(X, r, ONMFOptions(W=W0, display=1))
    figs = os.path.join(BASE, "figs")
    os.makedirs(figs, exist_ok=True)
    # Display H' basis images (7x19x19)
    affichage(H.T, (19, 19), ncols=7, suptitle="Orthogonal NMF on CBCL", save_path=os.path.join(figs, "ch4_onmf_cbcl_basis.pdf"))
    # Error vs iterations
    plt.figure()
    plt.plot(e)
    plt.xlabel("Iterations")
    plt.ylabel(r"$\|X-WH\|_F / \|X\|_F$")
    plt.title("Orthogonal NMF on CBCL")
    plt.tight_layout()
    plt.savefig(os.path.join(figs, "ch4_onmf_cbcl_error.pdf"), bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()

