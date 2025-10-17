"""
Exact coordinate descent for symmetric NMF (dense version):

Given symmetric nonnegative A in R^{n x n} and rank r, solve
    min_{H >= 0} 1/2 * ||A - H H^T||_F^2

This is a pure-Python/Numpy implementation of the dense branch of the
original MATLAB + MEX code by Vandaele et al. (2016). It follows the same
options interface and logging behavior as closely as practical.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np


@dataclass
class SymNMFOptions:
    maxiter: int = 100
    timelimit: float = 5.0
    display: Literal["on", "off"] = "on"
    shuffle_columns: int = 0  # 0 or 1
    initmatrix: Literal["zeros", "dense01"] = "zeros"
    seed: int = -1


def _init_H(n: int, r: int, options: SymNMFOptions) -> np.ndarray:
    if options.initmatrix == "zeros":
        return np.zeros((n, r), dtype=float)
    if options.initmatrix == "dense01":
        rng = np.random.default_rng(None if options.seed == -1 else options.seed)
        return rng.random((n, r))
    raise ValueError("options.initmatrix must be 'zeros' or 'dense01'")


def _scale_initial_H(A: np.ndarray, H: np.ndarray) -> np.ndarray:
    if np.sum(H) <= 0:
        return H
    # Use explicit dot to avoid potential BLAS warnings in some envs
    HtH = H.T.dot(H)
    nHtH = np.linalg.norm(HtH, ord="fro") ** 2
    HtA = H.T.dot(A)
    HtAHt = np.sum(HtA * H.T)
    scaling = HtAHt / nHtH if nHtH > 0 else 1.0
    return np.sqrt(max(scaling, 0.0)) * H


def _objective(A: np.ndarray, H: np.ndarray, nA: Optional[float] = None) -> float:
    if nA is None:
        nA = np.linalg.norm(A, ord="fro") ** 2
    HtH = H.T.dot(H)
    nHtH = np.linalg.norm(HtH, ord="fro") ** 2
    HtA = H.T.dot(A)
    HtAHt = np.sum(HtA * H.T)
    return 0.5 * (nA - 2.0 * HtAHt + nHtH)


def _best_polynomial_root(a: float, b: float) -> float:
    """
    Solve min_{x >= 0} x^3 + a x + b following the piecewise closed form used
    in the reference MEX implementation.
    """
    a3 = 4.0 * (a ** 3)
    b2 = 27.0 * (b ** 2)
    delta = a3 + b2

    if delta <= 0:
        r3 = 2.0 * np.sqrt(max(-a / 3.0, 0.0))
        # th3 = atan2(sqrt(-delta/108), -b/2) / 3
        th3 = np.arctan2(np.sqrt(max(-delta / 108.0, 0.0)), -0.5 * b) / 3.0
        ymax = 0.0
        xopt = 0.0
        for k in (0, 2, 4):
            x = r3 * np.cos(th3 + (k * np.pi) / 3.0)
            if x >= 0.0:
                y = (x ** 4) / 4.0 + a * (x ** 2) / 2.0 + b * x
                if y < ymax:
                    ymax = y
                    xopt = x
        return xopt
    else:
        z = np.sqrt(delta / 27.0)
        x = np.cbrt(0.5 * (-b + z)) + np.cbrt(0.5 * (-b - z))
        y = (x ** 4) / 4.0 + a * (x ** 2) / 2.0 + b * x
        if (y < 0.0) and (x >= 0.0):
            return float(x)
        return 0.0


def symnmf(
    A: np.ndarray,
    Hr: int | np.ndarray,
    options: Optional[SymNMFOptions] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Symmetric NMF via exact coordinate descent (dense A).

    Parameters
    ----------
    A : np.ndarray
        Symmetric nonnegative matrix of shape (n, n).
    Hr : int | np.ndarray
        Either inner rank r, or initial matrix H of shape (n, r).
    options : SymNMFOptions, optional
        Algorithm options.

    Returns
    -------
    H : np.ndarray
        Nonnegative factor of shape (n, r).
    e : np.ndarray
        Sequence of objective values (starting with initial), shape (L,).
    t : np.ndarray
        Sequence of cumulative wall-clock times, aligned with e.
    """
    if options is None:
        options = SymNMFOptions()

    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]
    assert A.shape == (n, n), "A must be square"

    if isinstance(Hr, (int, np.integer)):
        r = int(Hr)
        H = _init_H(n, r, options)
    else:
        H = np.asarray(Hr, dtype=np.float64)
        n2, r = H.shape
        if n2 != n:
            raise ValueError("The size of A and H must match in the first dimension")

    # Validate and clamp options similar to MATLAB code
    maxiter = options.maxiter if (isinstance(options.maxiter, int) and options.maxiter > 0) else 100
    if maxiter > 1_000_000:
        maxiter = 1_000_000
    timelimit = options.timelimit if (isinstance(options.timelimit, (int, float)) and options.timelimit > 0) else 5.0
    shuffle_columns = 1 if options.shuffle_columns == 1 else 0

    # Scale initial H
    H = _scale_initial_H(A, H)

    # Initial objective
    nA = np.linalg.norm(A, ord="fro") ** 2
    e0 = _objective(A, H, nA)

    if options.display == "on":
        msg = f"Factorizing a {n}x{n} matrix using r={r} (maxiter={maxiter}, timelimit={timelimit})"
        if options.initmatrix == "zeros":
            msg += f"\nInitial matrix: zeros({n},{r})"
        elif options.initmatrix == "dense01":
            msg += f"\nInitial matrix: rand({n},{r}) with seed={options.seed}"
        if shuffle_columns:
            msg += "\nThe columns are shuffled"
        msg += f"\nInitial objective function={e0:1.5g}"
        print(msg)

    # Pre-alloc logs (following MATLAB behavior)
    et = np.full(maxiter, -1.0)
    tt = np.full(maxiter, -1.0)

    # Precompute helpers allocated once
    nL = np.empty(n, dtype=np.float64)
    nC = np.empty(r, dtype=np.float64)
    HH = np.empty((r, r), dtype=np.float64)
    AH = np.empty(n)
    HHH = np.empty(n)
    icol = np.arange(r)

    def precomputations():
        # Match MEX precomputations exactly
        # nL[i] = norm(H[i,:])^2
        for i in range(n):
            s = 0.0
            for j in range(r):
                hij = H[i, j]
                s += hij * hij
            nL[i] = s
        # nC[j] = norm(H[:,j])^2
        for j in range(r):
            s = 0.0
            for i in range(n):
                hij = H[i, j]
                s += hij * hij
            nC[j] = s
        # HH = H^T H (fill symmetric entries)
        for j1 in range(r):
            for j2 in range(j1, r):
                s = 0.0
                for i in range(n):
                    s += H[i, j1] * H[i, j2]
                HH[j2, j1] = s
                HH[j1, j2] = s

    precomputations()

    t0 = time.perf_counter()

    for it in range(maxiter):
        fdecrease = 0.0
        icol[:] = np.arange(r)
        if shuffle_columns:
            np.random.shuffle(icol)

        for kk in range(r):
            k = int(icol[kk])
            Hkn = H[:, k]
            HHkr = HH[k, :]

            # Precompute AH[i] values using column access like in MEX
            for i in range(n):
                # AHi = sum_j Hkn[j] * A[j, i]
                s = 0.0
                for j in range(n):
                    s += Hkn[j] * A[j, i]
                AH[i] = s

            for i in range(n):
                # HHH[i] as in MEX: sum_j<=k HHkr[j]*H[i,j] + sum_{j>k} HH[j,k]*H[i,j]
                s_HHH = 0.0
                if k >= 0:
                    for j in range(0, k + 1):
                        s_HHH += HHkr[j] * H[i, j]
                for j in range(k + 1, r):
                    s_HHH += HH[j, k] * H[i, j]
                HHH[i] = s_HHH

                hold = Hkn[i]
                a = nC[k] + nL[i] - A[i, i] - 2.0 * (hold ** 2)
                b = HHH[i] - AH[i] - Hkn[i] * a - (hold ** 3)
                hnew = _best_polynomial_root(a, b)
                s1 = hold - hnew

                if s1 != 0.0:
                    Hkn[i] = hnew
                    s2 = (hnew ** 2) - (hold ** 2)
                    nC[k] += s2
                    nL[i] += s2
                    # Update AH for rows >= i using column i of A (A[j,i])
                    for j2 in range(i, n):
                        AH[j2] -= A[j2, i] * s1
                    # Update HH
                    if k > 0:
                        HHkr[:k] -= s1 * H[i, :k]
                    HH[k, k] += s2
                    if k + 1 < r:
                        HH[k + 1 :, k] -= s1 * H[i, k + 1 :]
                    fdecrease += 4.0 * b * s1 - 2.0 * a * s2 + (hold ** 4) - (hnew ** 4)

        et[it] = fdecrease
        tt[it] = time.perf_counter() - t0

        if tt[it] >= timelimit:
            # Fill remaining with -1 to match MATLAB's semantics
            if it + 1 < maxiter:
                et[it + 1 :] = -1.0
                tt[it + 1 :] = -1.0
            break

    # Reconstruct objective trace e from e0 and et decreases
    e_vals = [e0]
    for decr in et:
        if decr == -1.0:
            break
        e_vals.append(e_vals[-1] - 0.5 * decr)
    e = np.array(e_vals)
    if len(e) == 1:
        t = np.array([0.0])
    else:
        # t[0] = 0, then previous tt values up to len(e)-1
        t = np.concatenate([[0.0], tt[: len(e) - 1]])

    if options.display == "on":
        ef = _objective(A, H, nA)
        print(f"Final objective function={ef:1.5g}")

    return H, e, t


