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

from algorithms import fro_nmf  # noqa: E402
from algorithms.nmf.fro_nmf import FroNMFOptions  # noqa: E402
from algorithms.separable_nmf.snpa_matlab import snpa_matlab  # noqa: E402


def main():
    data_path = os.path.join(BASE, "data sets", "microarrayIFNbeta.mat")
    mat = sio.loadmat(data_path)
    if "X" not in mat:
        raise KeyError("microarrayIFNbeta.mat must contain X")
    X = np.array(mat["X"], dtype=float)
    r = 3

    # SNPA initialization as in MATLAB
    Ks, Hs = snpa_matlab(X, r)
    W0 = X[:, Ks]
    H0 = Hs

    opts = FroNMFOptions(
        display=1,
        maxiter=100,
        algo="HALS",
        beta0=0.0,
        init_W=W0,
        init_H=H0,
        rescale_every=0,
    )
    W, H, e, t, _ = fro_nmf(X, r, opts)

    # Column-wise normalization: max(W[:,i]) = 1, push scale into H[i,:]
    for i in range(r):
        mwi = float(np.max(W[:, i])) if W.shape[0] > 0 else 1.0
        if mwi > 0:
            W[:, i] = W[:, i] / mwi
            H[i, :] = H[i, :] * mwi

    figs = os.path.join(BASE, "figs")
    os.makedirs(figs, exist_ok=True)
    plt.figure()
    plt.imshow(W, aspect="auto", cmap="gray")
    plt.colorbar()
    plt.title(r"Basis matrix $W$")
    plt.tight_layout()
    plt.savefig(os.path.join(figs, "microarray_W.pdf"), bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()

