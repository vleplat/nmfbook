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

from algorithms.beta_nmf.beta_nmf import beta_nmf, BetaNMFOptions  # noqa: E402


def main():
    # Generate sparse dataset
    m, n, r = 500, 1000, 40
    density = 0.01
    rng = np.random.default_rng(0)
    X = (rng.random((m, n)) < density).astype(float) * rng.random((m, n))

    # Random initialization
    W0 = rng.random((m, r))
    H0 = rng.random((r, n))

    # Standard MU (epsilon = 0)
    opts = BetaNMFOptions(
        W=W0.copy(),
        H=H0.copy(),
        maxiter=100,
        beta=2.0,
        accuracy=0.0,
        epsilon=0.0,
        extrapol="noextrap",
        display=0,
        rescale_every=0,
        scale_init=False,
    )
    print("Running MU (epsilon=0)")
    _, _, es, _ = beta_nmf(X, r, opts)

    # Modified MU (epsilon = machine eps)
    opts_eps = BetaNMFOptions(
        W=W0.copy(),
        H=H0.copy(),
        maxiter=100,
        beta=2.0,
        accuracy=0.0,
        epsilon=np.finfo(float).eps,
        extrapol="noextrap",
        display=0,
        rescale_every=0,
        scale_init=False,
    )
    print("Running modified MU (epsilon=eps)")
    _, _, em, _ = beta_nmf(X, r, opts_eps)

    # Plot
    figs = os.path.join(BASE, "figs")
    os.makedirs(figs, exist_ok=True)
    plt.figure()
    plt.plot(es, label="MU")
    plt.plot(em, "--", label="Modified MU")
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel(r"Error: $\|X-WH\|_F$")
    # MATLAB axis: [ 30 100 em(100) es(30) ]
    try:
        y_min = float(em[99])
    except Exception:
        y_min = float(em[-1])
    try:
        y_max = float(es[29])
    except Exception:
        y_max = float(np.max(es))
    plt.axis([30, 100, y_min, y_max])
    plt.tight_layout()
    plt.savefig(os.path.join(figs, "ch8_mu_vs_modifiedmu.pdf"), bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()

