from __future__ import annotations

import time
from typing import List, Tuple

import numpy as np

from algorithms.separable_nmf.spa_matlab import spa_matlab, SPAOptions
from algorithms.separable_nmf.snpa_matlab import snpa_matlab, SNPAOptions
from algorithms.separable_nmf.separable_nmf import vca as vca_light, faw as faw_light
from synthdatasetSepNMF import synthdatasetSepNMF


def _mve_preprocess(X: np.ndarray, r: int, mode: int) -> np.ndarray:
    """
    Approximate MVE-SPA preprocessing via linear dimensionality reduction:
    - mode 0: none
    - mode 1: truncated SVD to r dims
    - mode 3: random projection to r dims
    Returns transformed data Y with same number of columns as X.
    """
    if mode == 1:
        # truncated SVD: keep top-r left singular vectors
        U, _, _ = np.linalg.svd(X, full_matrices=False)
        Ur = U[:, :r]
        return Ur.T @ X
    if mode == 3:
        rng = np.random.default_rng(0)
        P = rng.standard_normal((r, X.shape[0]))
        P /= np.linalg.norm(P, axis=1, keepdims=True) + 1e-16
        return P @ X
    return X


def _mvespa_indices(X: np.ndarray, r: int, mode: int) -> List[int]:
    Y = _mve_preprocess(X, r, mode)
    return spa_matlab(Y, r, SPAOptions(normalize=1, display=0))


def run_exp_greedy_sepNMF(m: int, n: int, r: int, xp: int, condW: int, delta: np.ndarray, nummat: int, nalgo: List[int], diri: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Port of run_exp_greedy_sepNMF.m
    Returns (results, timings).
    """
    print("*************************************************")
    print({1: "    Experiment: well-conditioned Dirichlet",
           2: "    Experiment: well-conditioned Middle points",
           3: "    Experiment: ill-conditioned Dirichlet",
           4: "    Experiment: ill-conditioned Middle points"}.get(xp, "    Experiment"))
    print("***********************************************")
    results = np.zeros((len(nalgo), len(delta)), dtype=float)
    timings = np.zeros((len(nalgo),), dtype=float)
    print(f"Total number of noise levels: {len(delta):2.0f} ")
    for i, d in enumerate(delta):
        for _ in range(nummat):
            W, H, Noise = synthdatasetSepNMF(m, n, r, float(d), xp, condW, diri)
            Xt = W @ H + Noise
            permu = np.random.default_rng(0).permutation(Xt.shape[1])
            Xt = Xt[:, permu]
            Kstar = np.where(permu < r)[0]
            ldr = 1
            for aidx, algo in enumerate(nalgo):
                t0 = time.process_time()
                if algo == 1:
                    K = spa_matlab(Xt, r, SPAOptions(normalize=1, display=0))
                elif algo == 2:
                    K = vca_light(Xt, r, normalize="l2", random_state=0)
                elif algo == 3:
                    K = faw_light(Xt, r, normalize="l1")
                elif algo == 4:
                    K, _ = snpa_matlab(Xt, r, SNPAOptions(maxitn=500, proj=1, display=0))
                elif algo == 5:
                    mode = {1: 0, 2: 1, 3: 3}.get(ldr, 0)
                    K = _mvespa_indices(Xt, r, mode)
                    ldr += 1
                elif algo == 6:
                    Kpre = spa_matlab(Xt, r, SPAOptions(normalize=1, display=0))
                    # SPA on pinv(Xt[:,Kpre]) * Xt
                    Y = np.linalg.pinv(Xt[:, Kpre]) @ Xt
                    K = spa_matlab(Y, r, SPAOptions(normalize=1, display=0))
                else:
                    K = spa_matlab(Xt, r, SPAOptions(normalize=1, display=0))
                timings[aidx] += time.process_time() - t0
                rcur = len(set(K).intersection(set(Kstar)))
                results[aidx, i] += rcur / nummat / r
        print(f"{i+1}...", end="")
        if (i + 1) % 10 == 0:
            print()
    print()
    return results, timings

