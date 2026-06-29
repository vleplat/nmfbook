from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
from algorithms.separable_nmf.snpa_matlab import snpa_matlab, SNPAOptions


@dataclass
class ONMFOptions:
    maxiter: int = 200
    timemax: float = 30.0
    epsilon: float = np.finfo(float).eps
    accuracy: float = 1e-6
    delta: float = 1e-6
    W: Optional[np.ndarray] = None
    H: Optional[np.ndarray] = None
    display: int = 1
    random_state: int = 0


def _orth_nnls(X: np.ndarray, W: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Faithful port of orthNNLS.m.
    """
    m, n = X.shape
    r = W.shape[1]
    norm2w = np.sqrt(np.sum(W * W, axis=0)) + 1e-16
    Wn = W / norm2w.reshape(1, -1)
    A = X.T @ Wn  # (n x r)
    b = np.argmax(A, axis=1)  # best centroid index per column
    H = np.zeros((r, n), dtype=float)
    clusters: List[np.ndarray] = []
    for k in range(r):
        Kk = np.where(b == k)[0]
        clusters.append(Kk)
        if Kk.size > 0:
            H[k, Kk] = (Wn[:, k].T @ X[:, Kk]) / norm2w[k]
    return H, clusters


def _normalize_rows_unit(H: np.ndarray) -> np.ndarray:
    norm2h = np.sqrt(np.sum(H.T * H.T, axis=0)) + 1e-16
    return (H.T * (1.0 / norm2h)).T


def onmf(X: np.ndarray, r: int, options: Optional[ONMFOptions] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Orthogonal NMF (ONMF) via alternating closed-form updates (Pompili et al., 2014).
    Faithful to alternatingONMF.m + orthNNLS.m.
    """
    if options is None:
        options = ONMFOptions()
    m, n = X.shape
    rng = np.random.default_rng(options.random_state)

    # Initialization
    if options.W is not None and options.W.shape == (m, r):
        if options.display:
            print("Initialization provided by the user.")
        W = options.W.copy()
    else:
        if options.H is not None and options.H.shape == (r, n):
            H = np.maximum(options.epsilon, options.H.copy())
            H = _normalize_rows_unit(H)
            W = X @ H.T
        else:
            if options.display:
                print("Initialization by SNPA:")
            K, _ = snpa_matlab(X, r, SNPAOptions(display=options.display))
            if len(K) < r:
                raise RuntimeError("SNPA failed to extract r indices.")
            W = X[:, K]
            H = None

    normX2 = float(np.sum(X * X))
    e_vals: list[float] = []
    t_vals: list[float] = []
    start = __import__("time").perf_counter()
    i = 1
    mintime = 0.1
    cntdis = 0

    if options.display:
        print("Iteration number and relative error of ONMF iterates:")

    def not_converged() -> bool:
        if i <= 3 or len(e_vals) <= 2:
            return True
        return abs(e_vals[-1] - e_vals[-2]) > options.delta

    while i <= options.maxiter and (__import__("time").perf_counter() - start) <= options.timemax and not_converged():
        # 1) H update
        H, clusters = _orth_nnls(X, W)
        # Handle empty clusters: split largest cluster using ONMF on submatrix (r=2)
        sizes = [len(idx) for idx in clusters]
        empty = [k for k, sz in enumerate(sizes) if sz == 0]
        if len(empty) > 0:
            indmax = int(np.argmax(sizes))
            Kmax = clusters[indmax]
            if Kmax.size >= 2:
                subW, subH, _, _ = onmf(X[:, Kmax], 2, ONMFOptions(display=0, maxiter=50, delta=1e-6))
                W[:, indmax] = subW[:, 0]
                H[indmax, Kmax] = subH[0, :]
                tgt = empty[0]
                W[:, tgt] = subW[:, 1]
                H[tgt, Kmax] = subH[1, :]
        # Normalize rows of H
        H = _normalize_rows_unit(H)

        # 2) W update
        for ii in range(r):
            Ki = np.where(H[ii, :] > 0.0)[0]
            if Ki.size > 0:
                W[:, ii] = X[:, Ki] @ H[ii, Ki].T

        # Relative error using identity for ONMF when HH^T=I and W=XH^T
        err = float(np.sqrt(max(0.0, normX2 - float(np.sum(W * W)))) / (np.sqrt(normX2) + 1e-16))
        e_vals.append(err)
        t_vals.append(__import__("time").perf_counter() - start)

        if options.display == 1 and t_vals[-1] >= mintime:
            if err < 1e-4:
                print(f"{i}: {100*err:2.1e}...", end="")
            else:
                print(f"{i}: {100*err:2.2f}...", end="")
            mintime *= 2.0
            cntdis += 1
            if cntdis % 10 == 0:
                print()

        i += 1

    if options.display:
        print()

    return W, H, np.array(e_vals), np.array(t_vals)

