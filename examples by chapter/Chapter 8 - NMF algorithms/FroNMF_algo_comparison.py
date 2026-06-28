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
# Add utils folder to path and import sumte
_UTILS_DIR = os.path.join(BASE, "examples by chapter", "Chapter 8 - NMF algorithms", "utils")
if _UTILS_DIR not in sys.path:
    sys.path.insert(0, _UTILS_DIR)
from sumte import sumte  # type: ignore  # noqa: E402


def load_dataset(experience: int) -> tuple[np.ndarray, int]:
    import scipy.io as sio
    if experience in (1, 2):
        mat = sio.loadmat(os.path.join(BASE, "data sets", "CBCL.mat"))
        X = np.array(mat["X"], dtype=float)
        r = 49 if experience == 1 else 10
    elif experience == 3:
        mat = sio.loadmat(os.path.join(BASE, "data sets", "classic.mat"))
        X = np.array(mat["X"], dtype=float)
        r = 30
    elif experience == 4:
        mat = sio.loadmat(os.path.join(BASE, "data sets", "tdt2_top30.mat"))
        X = mat["X"]
        try:
            import scipy.sparse as sp
            if sp.issparse(X):
                X = X.toarray()
        except Exception:
            pass
        X = np.array(X, dtype=float)
        r = 30
    else:
        raise ValueError("experience must be 1..4")
    return X, r


def main():
    # Choose dataset
    experience = 1  # 1: CBCL r=49, 2: CBCL r=10, 3: classic r=30, 4: tdt2 r=30
    X, r = load_dataset(experience)
    m, n = X.shape

    # Options
    options = FroNMFOptions(
        timemax=5.0,       # 30 in the book
        beta0=0.0,
        maxiter=10**9,
        display=0,
        accuracy=0.0,
        alpha=0.5,
        delta=0.1,
    )
    nummarker = 40
    tmf = np.linspace(0.0, options.timemax, nummarker + 1)

    ebest = np.inf
    rng = np.random.default_rng(0)

    # One initialization (30 in the book)
    numinit = 1

    # Storage for averaged curves
    tm = em = tma = ema = ta = ea = tls = els = th = eh = the = ehe = tf = ef = tad = ead = None

    for i in range(1, numinit + 1):
        print(f"***** Initialization {i:2d} of {numinit:2d} *****")
        H = rng.random((r, n))
        W = rng.random((m, r))

        # MU (inneriter=1)
        print("Running MU...")
        opts = FroNMFOptions(**{**options.__dict__, "algo": "MUUP", "inneriter": 1, "init_W": W, "init_H": H})
        _, _, emi, tmi, _ = fro_nmf(X, r, opts)
        ebest = min(ebest, float(np.min(emi)))
        if i == 1:
            tm = tmf
            em = sumte(emi, tmi, tmf) / numinit
        else:
            em += sumte(emi, tmi, tmf) / numinit

        # A-MU (inneriter=100)
        print("Running A-MU...")
        opts = FroNMFOptions(**{**options.__dict__, "algo": "MUUP", "inneriter": 100, "init_W": W, "init_H": H})
        _, _, emai, tmai, _ = fro_nmf(X, r, opts)
        ebest = min(ebest, float(np.min(emai)))
        if i == 1:
            tma = tmf
            ema = sumte(emai, tmai, tmf) / numinit
        else:
            ema += sumte(emai, tmai, tmf) / numinit

        # ALS
        print("Running ALS...")
        opts = FroNMFOptions(**{**options.__dict__, "algo": "ALSH", "init_W": W, "init_H": H})
        _, _, elsi, tlsi, _ = fro_nmf(X, r, opts)
        ebest = min(ebest, float(np.min(elsi)))
        if i == 1:
            tls = tmf
            els = sumte(elsi, tlsi, tmf) / numinit
        else:
            els += sumte(elsi, tlsi, tmf) / numinit

        # A-HALS
        print("Running A-HALS...")
        opts = FroNMFOptions(**{**options.__dict__, "algo": "HALS", "init_W": W, "init_H": H})
        _, _, ehi, thi, _ = fro_nmf(X, r, opts)
        ebest = min(ebest, float(np.min(ehi)))
        if i == 1:
            th = tmf
            eh = sumte(ehi, thi, tmf) / numinit
        else:
            eh += sumte(ehi, thi, tmf) / numinit

        # E-A-HALS (beta0=0.5)
        print("Running E-A-HALS...")
        opts = FroNMFOptions(**{**options.__dict__, "algo": "HALS", "beta0": 0.5, "init_W": W, "init_H": H})
        _, _, ehei, thei, _ = fro_nmf(X, r, opts)
        ebest = min(ebest, float(np.min(ehei)))
        if i == 1:
            the = tmf
            ehe = sumte(ehei, thei, tmf) / numinit
        else:
            ehe += sumte(ehei, thei, tmf) / numinit

        # FPGM
        print("Running FPGM...")
        opts = FroNMFOptions(**{**options.__dict__, "algo": "FPGM", "inneriter": 100, "init_W": W, "init_H": H})
        _, _, efi, tfi, _ = fro_nmf(X, r, opts)
        ebest = min(ebest, float(np.min(efi)))
        if i == 1:
            tf = tmf
            ef = sumte(efi, tfi, tmf) / numinit
        else:
            ef += sumte(efi, tfi, tmf) / numinit

        # AO-ADMM
        print("Running AO-ADMM...")
        # Use delta=0.01 for ADMM as in MATLAB snippet, inneriter=100
        opts = FroNMFOptions(**{**options.__dict__, "algo": "ADMM", "delta": 0.01, "inneriter": 100, "init_W": W, "init_H": H, "alpha": 1.0})
        _, _, eadi, tadi, _ = fro_nmf(X, r, opts)
        ebest = min(ebest, float(np.min(eadi)))
        if i == 1:
            tad = tmf
            ead = sumte(eadi, tadi, tmf) / numinit
        else:
            ead += sumte(eadi, tadi, tmf) / numinit

    # Plot (subset of algorithms)
    figs = os.path.join(BASE, "figs")
    os.makedirs(figs, exist_ok=True)
    plt.figure()
    def plot_curve(t, e, style, label):
        if t is not None and e is not None:
            plt.semilogy(t, e - ebest, style, label=label)
    plot_curve(tm, em, "d-", "MU")
    plot_curve(tma, ema, "d--", "A-MU")
    plot_curve(tls, els, "*-", "ALS")
    plot_curve(th, eh, "s:", "A-HALS")
    plot_curve(the, ehe, "s-", "E-A-HALS")
    plot_curve(tf, ef, "*-", "FPGM")
    plot_curve(tad, ead, "o-.", "AO-ADMM")
    plt.grid(True, which="both")
    plt.legend()
    plt.ylabel(r"$\|X-WH\|_F/\|X\|_F - e_{best}$")
    plt.xlabel("Time (s.)")
    plt.xlim(0, options.timemax)
    # y-axis lower bound from available curves
    ymin = np.inf
    for arr in [em, ema, els, eh, ehe]:
        if arr is not None:
            ymin = min(ymin, float(np.min(arr - ebest)))
    if np.isfinite(ymin):
        plt.ylim(ymin, 0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(figs, "ch8_fronmf_algo_comparison_subset.pdf"), bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()

