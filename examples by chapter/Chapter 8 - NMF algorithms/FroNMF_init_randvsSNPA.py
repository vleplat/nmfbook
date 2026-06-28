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

from algorithms.nmf.fro_nmf import fro_nmf, FroNMFOptions  # noqa: E402
from algorithms.separable_nmf.snpa_matlab import snpa_matlab  # noqa: E402


def main():
    # Data set
    import scipy.io as sio
    data_path = os.path.join(BASE, "data sets", "CBCL.mat")
    mat = sio.loadmat(data_path)
    X = np.array(mat["X"], dtype=float)
    seed = 2020
    r = 49
    # Time-limited runs (5 seconds)
    opts = FroNMFOptions(
        timemax=5.0,
        beta0=0.0,
        maxiter=10**9,
        algo="HALS",
        alpha=0.5,
        display=0,
        accuracy=0.0,
    )

    # SNPA initialization on X' (as in MATLAB)
    print("Running A-HALS with SNPA initialization...")
    Ks, Wsnpa = snpa_matlab(X.T, r)
    H0 = X[Ks, :]  # r x n
    W0 = Wsnpa.T   # m x r
    opts_snpa = FroNMFOptions(**{**opts.__dict__, "init_W": W0, "init_H": H0})
    _, _, ehe, the, _ = fro_nmf(X, r, opts_snpa)

    # Random initialization
    print("Running A-HALS with random initialization...")
    rng = np.random.default_rng(seed)
    m, n = X.shape
    Wrand = rng.random((m, r))
    Hrand = rng.random((r, n))
    opts_rand = FroNMFOptions(**{**opts.__dict__, "init_W": Wrand, "init_H": Hrand})
    _, _, eher, ther, _ = fro_nmf(X, r, opts_rand)

    # Display
    figs = os.path.join(BASE, "figs")
    os.makedirs(figs, exist_ok=True)
    plt.figure()
    # Emulate MATLAB markers/linestyle
    plt.semilogy(the, ehe, "-", marker="o", markevery=max(1, len(the)//20), markersize=8, linewidth=2, label="SNPA")
    plt.semilogy(ther, eher, "--", marker="s", markevery=max(1, len(ther)//20), markersize=8, linewidth=2, label="rand")
    plt.grid(True, which="both")
    plt.legend()
    plt.ylabel(r"$\|X-WH\|_F / \|X\|_F$")
    plt.xlabel("Time (s.)")
    plt.xlim(0, 5)
    plt.ylim(7.5e-2, 0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(figs, "ch8_fronmf_init_rand_vs_snpa.pdf"), bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()

