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

from algorithms.semi_nmf import semi_nmf, SemiNMFOptions
from utils.affichage import affichage


def main():
    data_path = os.path.join(BASE, "data sets", "CBCL.mat")
    mat = sio.loadmat(data_path)
    X = np.array(mat["X"], dtype=np.float64)
    r = 49
    # semi-NMF in MATLAB script uses X' and returns [H,W,...] then W=W'
    # We run on X.T and display nonnegative part of W.T as basis images.
    W, H, *_ = semi_nmf(X.T, r, SemiNMFOptions(display=1))
    # Display basis elements as images from H' (pixels x r)
    Wimg = np.maximum(0.0, H.T)
    figs = os.path.join(BASE, "figs")
    os.makedirs(figs, exist_ok=True)
    affichage(Wimg, (19, 19), ncols=7, suptitle="Basis elements of semi-NMF",
              save_path=os.path.join(figs, "cbcl_seminmf_basis.pdf"))
    print("Saved CBCL semi-NMF figures in", figs)


if __name__ == "__main__":
    main()

