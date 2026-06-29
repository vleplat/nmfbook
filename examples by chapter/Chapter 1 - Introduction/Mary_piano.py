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
try:
    from algorithms.beta_nmf import beta_nmf, BetaNMFOptions
except ModuleNotFoundError:
    if BASE not in sys.path:
        sys.path.insert(0, BASE)
    from algorithms.beta_nmf import beta_nmf, BetaNMFOptions


def main():
    data_path = os.path.join(BASE, "data sets", "piano_Mary.mat")
    mat = sio.loadmat(data_path)
    if "X" in mat:
        X = np.array(mat["X"], dtype=np.float64)
    elif "Xspectr" in mat:
        X = np.array(mat["Xspectr"], dtype=np.float64)
    else:
        raise KeyError("piano_Mary.mat must contain X or Xspectr")

    r = 4
    opts = BetaNMFOptions(beta=1.0, maxiter=500, timemax=30.0, display=1, rescale_every=1)
    W, H, e, t = beta_nmf(X, r, opts)

    # Plot activations (rows of H) and frequency responses (columns of W)
    fig_dir = os.path.join(BASE, "figs")
    os.makedirs(fig_dir, exist_ok=True)

    timex = (np.arange(H.shape[1]) / 600.0) * 5.0
    freqx = (np.arange(1, W.shape[0] + 1) * 50.0) / 1000.0
    notes = ["C_4", "hammer", "D_4", "E_4"]

    plt.figure(figsize=(10, 8))
    p = 1
    for i in range(r):
        ax1 = plt.subplot(r, 2, p)
        ax1.plot(timex, H[i, :])
        ax1.grid(True)
        ax1.set_ylabel(notes[i] if i < len(notes) else f"comp {i+1}")
        if p == 1:
            ax1.set_title("Activations of the sources")
        if i == r - 1:
            ax1.set_xlabel("Time (s.)")

        ax2 = plt.subplot(r, 2, p + 1)
        ax2.plot(freqx, W[:, i])
        ax2.grid(True)
        ax2.set_ylabel(notes[i] if i < len(notes) else f"comp {i+1}")
        if p == 1:
            ax2.set_title("Frequency response of the sources")
        ax2.set_xlim([freqx[0], freqx[min(49, len(freqx) - 1)]])
        ymax = float(np.max(W[:, i]))
        if not np.isfinite(ymax) or ymax <= 0:
            ymax = 1.0
        ax2.set_ylim([1e-3, ymax * 1.05])
        ax2.set_xlabel("Frequency (kHz)") if i == r - 1 else None
        p += 2
    plt.tight_layout()
    out = os.path.join(fig_dir, "mary_piano_sources.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(e)
    plt.xlabel("Iterations")
    plt.ylabel("D_1(X, WH)")
    plt.title("Evolution of the objective function")
    out2 = os.path.join(fig_dir, "mary_piano_objective.pdf")
    plt.savefig(out2, bbox_inches="tight")
    plt.close()
    print("Saved:", out)
    print("Saved:", out2)


if __name__ == "__main__":
    main()


