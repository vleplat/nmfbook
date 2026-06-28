from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from algorithms.nmf.nnls import NNLSOptions, nnls
from .matlab_utils import weightedgroupedsparseproj_col, fastgradsparseNNLS
from .matlab_utils import project_to_hoyer_sparsity


@dataclass
class SparseNMFOptions:
    sW: Optional[float] = None
    sH: Optional[float] = None
    maxiter: int = 500
    timemax: float = 5.0
    delta: float = 0.1
    inneriter: int = 10
    W: Optional[np.ndarray] = None
    H: Optional[np.ndarray] = None
    FPGM: int = 0
    colproj: int = 0
    display: int = 1


def _equalize_norms(W: np.ndarray, H: np.ndarray) -> None:
    normW = np.sqrt(np.sum(W * W, axis=0)) + 1e-16
    normH = np.sqrt(np.sum(H.T * H.T, axis=0)) + 1e-16
    r = W.shape[1]
    for k in range(r):
        W[:, k] = W[:, k] / np.sqrt(normW[k]) * np.sqrt(normH[k])
        H[k, :] = H[k, :] / np.sqrt(normH[k]) * np.sqrt(normW[k])


def sparse_nmf(X: np.ndarray, r: int, options: Optional[SparseNMFOptions] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if options is None:
        options = SparseNMFOptions()
    m, n = X.shape
    if options.W is not None and options.H is not None:
        W = options.W.copy()
        H = options.H.copy()
    else:
        W = np.random.default_rng(0).random((m, r))
        H = np.random.default_rng(1).random((r, n))
        num = float(np.sum((W.T @ X) * H))
        den = float(np.sum((W.T @ W) * (H @ H.T))) + 1e-16
        alpha = num / den
        W *= alpha
    if options.sW is not None:
        W = weightedgroupedsparseproj_col(W, options.sW)
    if options.sH is not None:
        H = np.vstack([project_to_hoyer_sparsity(H[k, :], options.sH) for k in range(r)])
        H = np.maximum(0.0, H)
    nX2 = float(np.sum(X * X))
    nX = np.sqrt(nX2)
    e_vals: list[float] = []
    t_vals: list[float] = []
    start = __import__("time").perf_counter()
    itercount = 1
    Wbest = W.copy()
    Hbest = H.copy()
    while itercount <= options.maxiter and (__import__("time").perf_counter() - start) <= options.timemax:
        _equalize_norms(W, H)
        if options.FPGM == 0 and options.sH is None:
            H, _, _ = nnls(W, X, NNLSOptions(algo="HALS", init=H, delta=options.delta, inneriter=options.inneriter, alpha=0.5))
        else:
            H, _, _ = fastgradsparseNNLS(X, W, H, inneriter=options.inneriter, s=options.sH)
        if options.sW is None:
            Wt, _, _ = nnls(H.T, X.T, NNLSOptions(algo="HALS", init=W.T, delta=options.delta, inneriter=options.inneriter, alpha=0.5))
            W = Wt.T
        else:
            Wt, _, _ = nnls(H.T, X.T, NNLSOptions(algo="HALS", init=W.T, delta=options.delta, inneriter=options.inneriter, alpha=0.5))
            W = Wt.T
            W = weightedgroupedsparseproj_col(W, options.sW)
        XHt = X @ H.T
        HHt = H @ H.T
        err = float(np.sqrt(max(0.0, nX2 - 2.0 * np.sum(XHt * W) + np.sum(HHt * (W.T @ W)))) / nX)
        e_vals.append(err)
        t_vals.append(__import__("time").perf_counter() - start)
        if itercount >= 2 and e_vals[-1] <= e_vals[-2]:
            Wbest = W.copy()
            Hbest = H.copy()
        if options.display == 1 and itercount % 10 == 0:
            print(f"{itercount}:{100*e_vals[-1]:.3f} - ", end="")
            if itercount % 100 == 0:
                print()
        itercount += 1
    return Wbest, Hbest, np.array(e_vals), np.array(t_vals)

