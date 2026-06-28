from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np

from algorithms.nmf.nnls import NNLSOptions, nnls


@dataclass
class ExactNMFOptions:
    maxiter: int = 200
    timemax: float = 60.0
    num_inits: int = 20
    tolerance: float = 1e-12  # relative Frobenius threshold for exactness
    nnls_algo: Literal["HALS", "MUUP", "ALSH"] = "HALS"
    display: int = 1
    random_state: int = 0
    W: Optional[np.ndarray] = None
    H: Optional[np.ndarray] = None


def _relative_error(X: np.ndarray, W: np.ndarray, H: np.ndarray) -> float:
    num = np.linalg.norm(X - W @ H, ord="fro")
    den = max(np.linalg.norm(X, ord="fro"), 1e-16)
    return float(num / den)


def exact_nmf_heuristic(X: np.ndarray, r: int, options: Optional[ExactNMFOptions] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Heuristic for Exact NMF: attempts to find W, H >= 0 with X = W H exactly (up to tolerance).
    Multiple random initializations with alternating NNLS updates are tried; the best solution
    (lowest residual) is returned, and early-stopped if the relative error <= tolerance.
    """
    if options is None:
        options = ExactNMFOptions()
    if np.min(X) < 0:
        raise ValueError("X should be nonnegative")

    m, n = X.shape
    rng = np.random.default_rng(options.random_state)
    tiny = 1e-16

    best_W: np.ndarray | None = None
    best_H: np.ndarray | None = None
    best_err: float = float("inf")
    best_evals: list[float] = []
    best_tvals: list[float] = []

    # Prepare NNLS options
    nnls_opts = NNLSOptions(algo=options.nnls_algo, inneriter=500, delta=1e-6)

    # Overall stopwatch
    t0 = __import__("time").perf_counter()

    for init_id in range(max(1, options.num_inits)):
        # Initialize W, H
        if options.W is not None and options.H is not None and init_id == 0:
            W = np.maximum(tiny, options.W.copy())
            H = np.maximum(tiny, options.H.copy())
        else:
            W = np.maximum(tiny, rng.random((m, r)))
            H = np.maximum(tiny, rng.random((r, n)))

        # Normalize columns of W to avoid scale degeneracy
        for k in range(r):
            scale = max(float(np.max(W[:, k])), tiny)
            W[:, k] /= scale
            H[k, :] *= scale

        e_vals: list[float] = []
        t_vals: list[float] = []

        start = __import__("time").perf_counter()
        i = 1
        mintime = 0.1
        cntdis = 0

        def not_converged() -> bool:
            if i <= 12 or len(e_vals) <= 2:
                return True
            return abs(e_vals[-1] - e_vals[-11]) > 1e-10 * abs(e_vals[-1])

        while (
            i <= options.maxiter
            and (__import__("time").perf_counter() - start) <= options.timemax
            and (__import__("time").perf_counter() - t0) <= options.timemax
            and not_converged()
        ):
            # H update: solve min_{H>=0} ||X - W H||_F
            H, _, _ = nnls(W, X, nnls_opts)
            # W update: solve min_{W>=0} ||X - W H||_F
            Wt, _, _ = nnls(H.T, X.T, nnls_opts)
            W = Wt.T

            # Column rescale to maintain numerical stability
            for k in range(r):
                norm_w = max(tiny, float(np.linalg.norm(W[:, k])))
                W[:, k] /= norm_w
                H[k, :] *= norm_w

            err = _relative_error(X, W, H)
            e_vals.append(err)
            t_vals.append(__import__("time").perf_counter() - start)

            if options.display == 1 and t_vals[-1] >= mintime:
                print(f"[ExactNMF init {init_id+1}] {i}...", end="")
                mintime *= 2.0
                cntdis += 1
                if cntdis % 10 == 0:
                    print()

            # Early exit if exact enough
            if err <= options.tolerance:
                break

            i += 1

        if options.display == 1:
            print()

        # Keep best solution
        if e_vals and e_vals[-1] < best_err:
            best_err = e_vals[-1]
            best_W = W.copy()
            best_H = H.copy()
            best_evals = e_vals[:]
            best_tvals = t_vals[:]

        # Global time budget
        if (__import__("time").perf_counter() - t0) > options.timemax:
            break

        # If already exact enough, stop trying more inits
        if best_err <= options.tolerance:
            break

    if best_W is None or best_H is None:
        # Fallback to one random attempt if all failed to start (shouldn't happen)
        best_W = np.maximum(tiny, rng.random((m, r)))
        best_H = np.maximum(tiny, rng.random((r, n)))
        best_evals = [ _relative_error(X, best_W, best_H) ]
        best_tvals = [ 0.0 ]

    return best_W, best_H, np.array(best_evals), np.array(best_tvals)

