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

from algorithms.beta_nmf.beta_nmf import beta_nmf, BetaNMFOptions, _mubeta_update  # noqa: E402


def beta_div_reference(X: np.ndarray, beta: float) -> float:
    # Reference factor Xee^T/n used in MATLAB script
    n = X.shape[1]
    e = np.ones((n, 1), dtype=float)
    Xe = X @ e
    Xref = (Xe @ e.T) / n
    eps = np.finfo(float).eps
    if beta == 2.0:
        return 0.5 * float(np.linalg.norm(X - Xref, ord="fro") ** 2)
    elif beta == 1.0:
        Z = X / (Xref + eps)
        Xnnz = X[X > 0]
        Znnz = Z[X > 0]
        return float(np.sum(Xnnz * np.log(Znnz + eps)) - np.sum(X) + np.sum(Xref))
    elif beta == 0.0:
        Z = X / (Xref + eps)
        return float(np.sum(Z - np.log(Z + eps) - 1.0))
    else:
        WH = Xref + eps
        return float(
            np.sum(
                (np.power(X + eps, beta) - beta * X * np.power(WH, beta - 1.0) + (beta - 1.0) * np.power(WH, beta))
                / (beta * (beta - 1.0))
            )
        )


def main():
    # Load CBCL
    data_path = os.path.join(BASE, "data sets", "CBCL.mat")
    import scipy.io as sio
    mat = sio.loadmat(data_path)
    X = np.array(mat["X"], dtype=float)
    m, n = X.shape
    r = 49
    beta = 1.5

    # Initialization
    rng = np.random.default_rng(2020)
    W0 = np.maximum(np.finfo(float).eps, rng.random((m, r)))
    H0 = np.maximum(np.finfo(float).eps, rng.random((r, n)))

    # One MU step pre-improvement (as in MATLAB)
    H1, _ = _mubeta_update(X, W0, H0, beta, epsilon=np.finfo(float).eps)
    W1_T, _ = _mubeta_update(X.T, H1.T, W0.T, beta, epsilon=np.finfo(float).eps)
    W1 = W1_T.T

    # MU without extrapolation
    opts_no = BetaNMFOptions(
        W=W1.copy(),
        H=H1.copy(),
        beta=beta,
        maxiter=100,
        timemax=1e9,
        accuracy=0.0,
        extrapol="noextrap",
        display=0,
    )
    print("***Running MU for beta-NMF without extrapolation***")
    _, _, e_no, _ = beta_nmf(X, r, opts_no)

    # MU with Nesterov extrapolation (MUe)
    opts_ex = BetaNMFOptions(
        W=W1.copy(),
        H=H1.copy(),
        beta=beta,
        maxiter=100,
        timemax=1e9,
        accuracy=0.0,
        extrapol="nesterov",
        display=0,
    )
    print("***Running MU for beta-NMF with extrapolation (MUe)***")
    _, _, e_ex, _ = beta_nmf(X, r, opts_ex)

    # Display
    err0 = beta_div_reference(X, beta)
    mine = min(np.min(e_no), np.min(e_ex)) / max(err0, 1e-16)
    figs = os.path.join(BASE, "figs")
    os.makedirs(figs, exist_ok=True)
    plt.figure()
    y1 = e_no / max(err0, 1e-16) - mine
    y2 = e_ex / max(err0, 1e-16) - mine
    # Avoid nonpositive values for semilogy
    tiny = 1e-16
    y1 = np.maximum(y1, tiny)
    y2 = np.maximum(y2, tiny)
    line1, = plt.semilogy(y1, label="MU")
    line2, = plt.semilogy(y2, "-.", label="MUe")
    ax = plt.gca()
    # Use log tick formatter to avoid '0' labels
    ax.set_yscale("log")
    # Set limits and log ticks
    y_min = float(min(y1[y1 > tiny].min(initial=1e-12), y2[y2 > tiny].min(initial=1e-12)))
    y_max = float(max(y1.max(initial=1.0), y2.max(initial=1.0)))
    if not np.isfinite(y_min) or y_min <= 0:
        y_min = 1e-12
    if not np.isfinite(y_max):
        y_max = 1.0
    ax.set_ylim(y_min, y_max)
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10.0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.LogFormatterMathtext(base=10.0))
    plt.grid(True, which="both", axis="both")
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel(r"$D_{3/2}(X,WH)/D_{3/2}(X,Xee^T/n) - e_{min}$")
    # Ensure y-label is fully visible (math text is wide)
    plt.subplots_adjust(left=0.22)
    plt.tight_layout()
    plt.savefig(os.path.join(figs, "ch8_mu_vs_extrapolatedmu.pdf"), bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()

