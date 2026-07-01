from __future__ import annotations

import os
import sys
import numpy as np

BASE = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE not in sys.path:
    sys.path.insert(0, BASE)
UTILS = os.path.join(os.path.dirname(__file__), "utils")
if UTILS not in sys.path:
    sys.path.insert(0, UTILS)

from run_exp_greedy_sepNMF import run_exp_greedy_sepNMF
from scriptfigsepNMF import scriptfigsepNMF
from utils.silence_warnings import silence_numpy_warnings


def main():
    silence_numpy_warnings()
    seed = 2020
    m, r, n = 20, 10, 100
    condW = 6
    nummat = 2
    nalgo = [5, 5, 5]  # MVE-SPA with different LDR modes
    diri = 0.5
    numnoiselevels = 10
    np.random.seed(seed)
    figs = os.path.join(BASE, "figs")
    # Use xp=3 (ill-conditioned Dirichlet), delta per script
    xp = 3
    delta = np.logspace(-6, 0, numnoiselevels)
    results, timings = run_exp_greedy_sepNMF(m, n, r, xp, condW, delta, nummat, nalgo, diri)
    # Relabel curves to match legend (SPA baseline is implicitly shown in compare_greedy script)
    scriptfigsepNMF(results, timings, delta, nalgo, xp, figs)


if __name__ == "__main__":
    main()

