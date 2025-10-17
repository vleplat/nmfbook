from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np

from .nnls import nnls, NNLSOptions


@dataclass
class FroNMFOptions:
    display: int = 1
    maxiter: int = 500
    timemax: float = 60.0
    accuracy: float = 1e-4
    algo: Literal["HALS", "ASET", "MUUP", "FPGM", "ADMM", "ALSH"] = "HALS"
    delta: float = 0.1
    alpha: float = 0.5
    inneriter: int = 100
    extrapolprojH: int = 3
    beta0: float = 0.5
    eta: float = 1.5
    gammabeta: float = 1.01
    gammabetabar: float = 1.005
    init_W: Optional[np.ndarray] = None
    init_H: Optional[np.ndarray] = None
    # Rescale columns of W and rows of H every N iterations (0 disables)
    rescale_every: int = 0


def _scale_init(X: np.ndarray, W: np.ndarray, H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    XHt = X @ H.T
    HHt = H @ H.T
    denom = np.sum(HHt * (W.T @ W))
    if denom <= 0:
        return W, H
    scaling = np.sum(XHt * W) / denom
    W = W * scaling

    # balance column/row norms: ||W[:,k]|| = ||H[k,:]||
    normW = np.sqrt(np.sum(W * W, axis=0)) + 1e-16
    normH = np.sqrt(np.sum(H.T * H.T, axis=0)) + 1e-16
    d = np.empty_like(normW)
    for k in range(W.shape[1]):
        W[:, k] = W[:, k] / np.sqrt(normW[k]) * np.sqrt(normH[k])
        d[k] = np.sqrt(normW[k]) / np.sqrt(normH[k])
        H[k, :] = H[k, :] * d[k]
    return W, H


def fro_nmf(
    X: np.ndarray,
    r: int,
    options: Optional[FroNMFOptions] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if options is None:
        options = FroNMFOptions()

    m, n = X.shape
    if options.init_W is None:
        W = np.random.default_rng(0).random((m, r))
    else:
        W = np.maximum(0.0, np.asarray(options.init_W, dtype=np.float64).copy())
    if options.init_H is None:
        H = np.random.default_rng(1).random((r, n))
    else:
        H = np.maximum(0.0, np.asarray(options.init_H, dtype=np.float64).copy())

    if options.algo == "MUUP":
        beta0 = 0.0
    else:
        beta0 = options.beta0

    # ensure beta and gamma relations
    if not (options.eta > options.gammabeta > options.gammabetabar):
        raise ValueError("eta > gammabeta > gammabetabar must hold")
    if not (0.0 <= beta0 <= 1.0):
        raise ValueError("beta0 must be in [0,1]")

    # scale init
    W, H = _scale_init(X, W, H)

    # extrapolation vars
    Wy = W.copy()
    Hy = H.copy()
    beta = [beta0]
    betamax = 1.0

    nX = np.linalg.norm(X, ord="fro")
    XHt = X @ H.T
    HHt = H @ H.T
    e0 = np.sqrt(max(0.0, nX**2 - 2 * np.sum(XHt * W) + np.sum(HHt * (W.T @ W)))) / max(nX, 1e-16)
    e = [e0]
    etrue = [e0]
    t = [0.0]
    emin = e0
    Wbest = W.copy()
    Hbest = H.copy()

    start = __import__("time").perf_counter()
    i = 1

    while i <= options.maxiter and t[-1] < options.timemax and (i <= 12 or abs(e[-1] - e[max(0, len(e)-11-1)]) >= options.accuracy):
        # NNLS for H with Wy
        nnls_opts = NNLSOptions(algo=options.algo, init=Hy, delta=options.delta, inneriter=options.inneriter, alpha=options.alpha)
        Hn, _, _ = nnls(Wy, X, nnls_opts)
        if options.extrapolprojH >= 2:
            Hy = Hn + beta[-1] * (Hn - H)
        else:
            Hy = Hn
        if options.extrapolprojH == 3:
            Hy = np.maximum(0.0, Hy)

        # NNLS for W with Hy
        nnls_opts_w = NNLSOptions(algo=options.algo, init=Wy.T, delta=options.delta, inneriter=options.inneriter, alpha=options.alpha)
        Wn_T, HyHyT_T, XHyT_T = nnls(Hy.T, X.T, nnls_opts_w)
        Wn = Wn_T.T
        XHyT = XHyT_T.T
        HyHyT = HyHyT_T.T

        Wy = Wn + beta[-1] * (Wn - W)
        if options.extrapolprojH == 1:
            Hy = Hn + beta[-1] * (Hn - H)

        # relative error
        ei = max(0.0, nX**2 - 2 * np.sum(XHyT * Wn) + np.sum(HyHyT * (Wn.T @ Wn)))
        ei = np.sqrt(ei) / max(nX, 1e-16)
        e.append(ei)
        t.append(__import__("time").perf_counter() - start)

        # adjust beta
        if e[-1] > e[-2]:
            if options.algo == "ADMM":
                options.delta = options.delta / 10.0
                options.inneriter = int(np.ceil(1.5 * options.inneriter))
            # rescale W, H
            normW = np.sqrt(np.sum(W * W, axis=0)) + 1e-16
            normH = np.sqrt(np.sum(H.T * H.T, axis=0)) + 1e-16
            for k in range(r):
                W[:, k] = W[:, k] / np.sqrt(normW[k]) * np.sqrt(normH[k])
                H[k, :] = H[k, :] / np.sqrt(normH[k]) * np.sqrt(normW[k])
            Wy = W.copy()
            Hy = H.copy()
            betamax = beta[-1] if i == 1 else beta[-2]
            beta.append(beta[-1] / options.eta)
        else:
            W = Wn
            H = Hn
            beta.append(min(betamax, beta[-1] * options.gammabeta))
            betamax = min(1.0, betamax * options.gammabetabar)

        if e[-1] <= emin:
            Wbest = Wn
            Hbest = np.maximum(Hy, 0.0)
            emin = e[-1]

        # Optional coupled rescaling to stabilize magnitudes while preserving WH
        if options.rescale_every and (i % options.rescale_every == 0):
            tiny = 1e-16
            for k in range(r):
                nw = max(tiny, float(np.linalg.norm(W[:, k])))
                nh = max(tiny, float(np.linalg.norm(H[k, :])))
                s = np.sqrt(nh / nw)
                W[:, k] *= s
                H[k, :] /= s

        i += 1

    return Wbest, Hbest, np.array(e), np.array(t), np.array(etrue)


