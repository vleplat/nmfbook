from __future__ import annotations

import os
import sys
import numpy as np

BASE = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "utils"))
from computevert import computevert
from scipy.optimize import nnls
import cvxpy as cp


def ssc1_nec_cond(H: np.ndarray) -> int:
    """
    Faithful port of SSC1_nec_cond.m:
    Checks whether all e - e_k belong to the relative interior of conv(H),
    using nonnegative least squares tests.
    Returns 1 if necessary condition for SSC1 is satisfied, else 0.
    """
    H = np.array(H, dtype=float, copy=True)
    r, n = H.shape
    debug = os.environ.get("SSC_DEBUG", "") == "1"
    if np.min(H) < 0:
        if debug:
            print("Fail: negative entry")
        return 0
    thres = 1e-6
    # MATLAB-like rank with SVD tolerance
    def matlab_rank(A: np.ndarray) -> int:
        s = np.linalg.svd(A, compute_uv=False)
        tol = max(A.shape) * np.finfo(float).eps * (s[0] if s.size > 0 else 0.0)
        return int(np.sum(s > tol))
    if matlab_rank(H) < r:
        if debug:
            print("Fail: rank(H) < r")
        return 0
    # Remove zero columns
    col_sums = H.sum(axis=0)
    H = H[:, col_sums > 0]
    n = H.shape[1]
    # Normalize columns to sum to 1
    col_sums = H.sum(axis=0, keepdims=True)
    col_sums = np.maximum(col_sums, 1e-16)
    H = H / col_sums
    # Remove duplicated/near-duplicated columns (angle threshold)
    col_norms = np.sqrt(np.sum(H * H, axis=0, keepdims=True))
    col_norms = np.maximum(col_norms, 1e-16)
    Hn2 = H / col_norms
    anglesH = np.triu(Hn2.T @ Hn2, 1)
    row, col = np.where(anglesH >= 1 - 100 * thres)
    ac = np.unique(col)
    keep = np.setdiff1d(np.arange(n), ac, assume_unique=True)
    H = H[:, keep]
    # Check number of zeros in each row is at least r-1
    if np.min((H == 0).sum(axis=1)) < (r - 1):
        if debug:
            print("Fail: zeros per row < r-1")
        return 0
    n = H.shape[1]
    # cvxpy-based NNLS to mimic lsqnonneg
    def nnls_cvx(C: np.ndarray, d: np.ndarray) -> float:
        # Prefer SciPy Lawson-Hanson for speed; fallback to cvxpy if it fails
        try:
            xv, _ = nnls(C, d)
            res = float(np.linalg.norm(C @ xv - d))
            return res * res
        except Exception:
            x = cp.Variable(C.shape[1], nonneg=True)
            obj = cp.Minimize(cp.norm2(C @ x - d))
            prob = cp.Problem(obj)
            prob.solve(solver=cp.SCS, verbose=False, eps=1e-9, max_iters=5000)
            if x.value is None:
                return float("inf")
            res = float(np.linalg.norm(C @ x.value - d))
            return res * res
    # Loop to check condition for each k
    for k in range(r):
        Ik = np.where(H[k, :] == 0)[0]
        if Ik.size == 0:
            if debug:
                print(f"Fail: Ik empty for k={k}")
            return 0
        if matlab_rank(H[:, Ik]) < r - 1:
            if debug:
                print(f"Fail: rank(H[:,Ik]) < r-1 for k={k}")
            return 0
        # Select vertices among Ik: for each column, test if it's a vertex
        Ikb = []
        for i, idx in enumerate(Ik):
            C = H[:, np.delete(Ik, i)]
            d = H[:, idx]
            res2 = nnls_cvx(C, d)
            if res2 > thres:
                Ikb.append(idx)
        if len(Ikb) == 0:
            if debug:
                print(f"Fail: Ikb empty for k={k}")
            return 0
        # Check whether e - e_k in relint conv(H[:, Ikb]) via NNLS
        d = np.ones(r); d[k] = 0.0
        C = H[:, Ikb]
        res2 = nnls_cvx(C, d)
        if res2 > thres:
            if debug:
                print(f"Fail: membership residual {res2} > thres for k={k}")
            return 0
    return 1


def isSSC_full(H: np.ndarray):
    """
    Port of isSSC.m:
    Returns ssc1, ssc2, vert (xs), SSCnec.
    """
    SSCnec = ssc1_nec_cond(H)
    if SSCnec == 0:
        return 0, 0, np.array([]), 0
    r, n = H.shape
    V = computevert(H.T)  # r x num_vertices
    if V.size == 0:
        return 0, 0, np.array([]), 0
    normV = np.sum(V * V, axis=0)  # ||x||^2 per vertex
    ind = np.where(normV >= 1 - 1e-12)[0]
    if ind.size == 0:
        # pick maximum norm anyway
        b = int(np.argmax(normV))
        vert = V[:, b]
    else:
        # choose among those with norm>=1 the one with smallest possible entry
        mins = np.min(V[:, ind], axis=0)
        b_local = int(np.argmin(mins))
        vert = V[:, ind[b_local]]
    if np.max(normV) > 1 + 1e-9:
        ssc1 = 0
        ssc2 = 0
    else:
        ssc1 = 1
        # SSC2 requires unit vectors among vertices; approximate by count
        ssc2 = 1
        if ind.size > r:
            ssc2 = 0
    return ssc1, ssc2, vert, 1

