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

from algorithms.nmu import recursive_nmu, RecursiveNMUOptions
from utils.affichage import affichage


def main():
    data_path = os.path.join(BASE, "data sets", "CBCL.mat")
    mat = sio.loadmat(data_path)
    X = np.array(mat["X"], dtype=np.float64)
    # MATLAB: X = X'; r = 49; recursiveNMU(X,r,2,200)
    X_alg = X.T
    r = 49
    W, H = recursive_nmu(X_alg, r, RecursiveNMUOptions(Cnorm=2, maxiter=200, display=1))
    # Display like MATLAB: affichage(H,7,19,19) where columns are images
    # Here H has shape (r, pixels). We want columns = 19*19 → use H.T
    figs = os.path.join(BASE, "figs")
    os.makedirs(figs, exist_ok=True)
    affichage(H.T, (19, 19), ncols=7, suptitle="Recursive NMU on CBCL",
              save_path=os.path.join(figs, "cbcl_nmu_basis.pdf"))
    print("Saved CBCL NMU figures in", figs)


if __name__ == "__main__":
    main()

