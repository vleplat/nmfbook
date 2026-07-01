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
from utils.silence_warnings import silence_numpy_warnings


def main():
    silence_numpy_warnings()
    data_path = os.path.join(BASE, "data sets", "Moffet.mat")
    import scipy.io as sio
    mat = sio.loadmat(data_path)
    X = np.array(mat["X"], dtype=float)
    r = 3  # per MATLAB script
    figs = os.path.join(BASE, "figs")
    os.makedirs(figs, exist_ok=True)
    # Common opts per MATLAB: lambda=1, maxiter=300, target=0.05
    base = MinVolNMFOptions(lam=1.0, maxiter=300, target=0.05, inneriter=20)
    # Model 1: H^T e <= e
    print("Running min-vol NMF with H^T e <= e ...")
    W1, H1, *_ = minvol_nmf(X, r, MinVolNMFOptions(**{**base.__dict__, "model": 1}))
    # Model 4: H^T e == e
    print("Running min-vol NMF with H^T e  = e...")
    W4, H4, *_ = minvol_nmf(X, r, MinVolNMFOptions(**{**base.__dict__, "model": 4}))
    # Diagnostics: column-sum statistics
    def colsum_stats(H: np.ndarray) -> tuple[float, float, float]:
        s = H.sum(axis=0)
        return float(s.min()), float(s.mean()), float(s.max())
    m1 = colsum_stats(H1)
    m4 = colsum_stats(H4)
    print("H1 column-sum stats (min, mean, max) [Ht e <= e]:", m1)
    print("H4 column-sum stats (min, mean, max) [Ht e  = e]:", m4)
    # Display abundance maps: rows of H as 50x50 images
    h, w = 50, 50
    affichage(H1.T, (h, w), ncols=3, suptitle=r"min-vol NMF - $H^T e \leq e$", save_path=os.path.join(figs, "ch4_minvol_moffet_maps_HtE_le_e.pdf"))
    affichage(H4.T, (h, w), ncols=3, suptitle=r"min-vol NMF - $H^T e = e$", save_path=os.path.join(figs, "ch4_minvol_moffet_maps_HtE_eq_e.pdf"))
    # Spectra
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1); plt.plot(W1); plt.title(r"$H^T e \leq e$")
    plt.subplot(1, 2, 2); plt.plot(W4); plt.title(r"$H^T e = e$")
    plt.tight_layout(); plt.savefig(os.path.join(figs, "ch4_minvol_moffet_spectra.pdf"), bbox_inches="tight"); plt.close()


if __name__ == "__main__":
    main()

