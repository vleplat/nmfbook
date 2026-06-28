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


def main():
    data_path = os.path.join(BASE, "data sets", "RamanSMCR.mat")
    mat = sio.loadmat(data_path)
    # Expect W (m x r) and H (r x n) as in MATLAB pack
    if "W" not in mat or "H" not in mat:
        raise KeyError("RamanSMCR.mat must contain W (m x r) and H (r x n)")
    W = np.array(mat["W"], dtype=float)
    H = np.array(mat["H"], dtype=float)
    m, r = W.shape
    rH, n = H.shape
    assert rH == r, "H rows must equal number of columns in W"

    figs = os.path.join(BASE, "figs")
    os.makedirs(figs, exist_ok=True)

    # Plot spectral signatures (columns of W) vs wavenumber
    De = 280
    xw = np.arange(De, De + m)
    plt.figure()
    for k in range(r):
        plt.plot(xw, W[:, k], label=chr(ord('A') + k))
    plt.axis([De, De + m - 1, 0, float(np.max(W))])
    plt.xlabel(r"Wavenumber (cm$^{-1}$)")
    plt.ylabel("Intensity")
    plt.title(r"Columns of $W$ - spectral signatures of the compounds")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figs, "raman_W_spectra.pdf"), bbox_inches="tight")
    plt.close()

    # Plot activations over time (rows of H) vs time 0:1/3:50
    # In MATLAB: plot(0:1/3:50, H'), so H is r x n and is transposed.
    # The x-grid should have length n. 0:1/3:50 yields 151 points; we truncate/reshape to n.
    t = np.arange(0, 50 + 1e-9, 1.0 / 3.0)
    if t.size < n:
        # pad if needed (should not happen for typical data, keep exact length n)
        t = np.arange(n) / 3.0
    t = t[:n]
    plt.figure()
    for k in range(r):
        plt.plot(t, H[k, :], label=chr(ord('A') + k))
    plt.xlabel("Time (s.)")
    plt.ylabel("Concentration")
    plt.title(r"Rows of $H$ - activation over time of the compounds")
    plt.axis([0, 50, 0, 1])
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figs, "raman_H_timecourses.pdf"), bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()

