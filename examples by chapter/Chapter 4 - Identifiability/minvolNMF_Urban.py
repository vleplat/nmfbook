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

from algorithms.minvol_nmf import minvol_nmf, MinVolNMFOptions
from utils.affichage import affichage


def main():
    # Example 4.48: Urban hyperspectral image, r=6
    data_path = os.path.join(BASE, "data sets", "Urban.mat")
    import scipy.io as sio
    mat = sio.loadmat(data_path)
    X = np.array(mat["X"], dtype=float)
    r = 6
    figs = os.path.join(BASE, "figs")
    os.makedirs(figs, exist_ok=True)

    # Common options (faithful to MATLAB script)
    base_opts = MinVolNMFOptions(maxiter=1000, lam=1.0, target=0.05)

    # Model 1 (H^T e <= e)
    print("Running min-vol NMF (1)...")
    opts1 = MinVolNMFOptions(**{**base_opts.__dict__, "model": 1})
    W1, H1, e1, _ = minvol_nmf(X, r, opts1)

    # Model 2 (H e = e)
    print("Running min-vol NMF (2)...")
    opts2 = MinVolNMFOptions(**{**base_opts.__dict__, "model": 2})
    W2, H2, e2, _ = minvol_nmf(X, r, opts2)

    # Model 3 (W^T e = e)
    print("Running min-vol NMF (3)...")
    opts3 = MinVolNMFOptions(**{**base_opts.__dict__, "model": 3})
    W3, H3, e3, _ = minvol_nmf(X, r, opts3)

    # Abundance maps: rows of H as 307x307 images
    affichage(H1.T, (307, 307), ncols=3, suptitle="min-vol NMF (1)", save_path=os.path.join(figs, "ch4_minvol_urban_maps_1.pdf"))
    affichage(H2.T, (307, 307), ncols=3, suptitle="min-vol NMF (2)", save_path=os.path.join(figs, "ch4_minvol_urban_maps_2.pdf"))
    affichage(H3.T, (307, 307), ncols=3, suptitle="min-vol NMF (3)", save_path=os.path.join(figs, "ch4_minvol_urban_maps_3.pdf"))

    # Spectral signatures: columns of W
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.plot(W1); plt.title("min-vol NMF (1)")
    plt.subplot(1, 3, 2); plt.plot(W2); plt.title("min-vol NMF (2)")
    plt.subplot(1, 3, 3); plt.plot(W3); plt.title("min-vol NMF (3)")
    plt.tight_layout()
    plt.savefig(os.path.join(figs, "ch4_minvol_urban_spectra.pdf"), bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()

