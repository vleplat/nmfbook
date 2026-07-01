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
    # Settings adjusted (smaller than book for speed), matching MATLAB script comments
    seed = 2020
    m, r, n = 20, 10, 100
    condW = 6
    nummat = 1
    nalgo = [1, 2, 3, 4, 5, 6]  # SPA, VCA, FAW, SNPA, MVE-SPA, SPA-SPA
    diri = 1.0  # larger than 0.5 due to smaller m
    numnoiselevels = 10
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    figs = os.path.join(BASE, "figs")
    # Experiments
    # 1) well-conditioned Dirichlet
    xp = 1
    delta1 = np.linspace(0.0, 1.3, numnoiselevels)
    results1, timings1 = run_exp_greedy_sepNMF(m, n, r, xp, condW, delta1, nummat, nalgo, diri)
    scriptfigsepNMF(results1, timings1, delta1, nalgo, xp, figs)
    # 2) well-conditioned Middle points
    xp = 2
    delta2 = np.linspace(0.0, 0.7, numnoiselevels)
    results2, timings2 = run_exp_greedy_sepNMF(m, n, r, xp, condW, delta2, nummat, nalgo, diri)
    scriptfigsepNMF(results2, timings2, delta2, nalgo, xp, figs)
    # 3) ill-conditioned Dirichlet
    xp = 3
    delta3 = np.logspace(-7, 1, numnoiselevels)
    results3, timings3 = run_exp_greedy_sepNMF(m, n, r, xp, condW, delta3, nummat, nalgo, diri)
    scriptfigsepNMF(results3, timings3, delta3, nalgo, xp, figs)
    # 4) ill-conditioned Middle points
    xp = 4
    delta4 = np.logspace(-3, np.log10(2), numnoiselevels)
    results4, timings4 = run_exp_greedy_sepNMF(m, n, r, xp, condW, delta4, nummat, nalgo, diri)
    scriptfigsepNMF(results4, timings4, delta4, nalgo, xp, figs)


if __name__ == "__main__":
    main()

