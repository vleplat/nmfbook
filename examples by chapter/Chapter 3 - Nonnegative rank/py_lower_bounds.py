from __future__ import annotations

import math
from typing import List, Tuple, Optional
import numpy as np

try:
    # SciPy MILP for rectangle covering
    from scipy.optimize import milp, LinearConstraint, Bounds
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

try:
    import cvxpy as cp
    CVXPY_OK = True
except Exception:
    CVXPY_OK = False


def vec(M: np.ndarray) -> np.ndarray:
    return M.reshape(-1, order="F")


def rec_cov_bound(X: np.ndarray) -> Tuple[int, List[np.ndarray]]:
    """
    Rectangle covering bound via naive enumeration + MILP:
    - Enumerate rectangles induced by subsets of rows with positive columns
    - Solve min 1^T z s.t. A z >= bin(X), z binary
    Returns (rcX, rec) with rec list of 0/1 rectangles (m x n).
    """
    m, n = X.shape
    if min(m, n) > 10:
        print("Warning: min(m,n) is rather large; this may be slow.")
    # If m > n, transpose to reduce enumeration like MATLAB
    transposed = False
    if m > n:
        X = X.T
        m, n = X.shape
        transposed = True
    # Enumerate rectangles
    A_cols: List[np.ndarray] = []
    for mask in range(1, 2**m):  # skip zero mask
        row_idx = [i for i in range(m) if (mask >> i) & 1]
        if len(row_idx) == 0:
            continue
        if len(row_idx) > 1:
            valcol = np.min(X[row_idx, :], axis=0)
        else:
            valcol = X[row_idx[0], :]
        col_idx = np.where(valcol > 0)[0]
        S = np.zeros((m, n), dtype=int)
        if len(row_idx) > 0 and len(col_idx) > 0:
            S[np.ix_(row_idx, col_idx)] = 1
        A_cols.append(vec(S))
    if not A_cols:
        return 0, []
    A = np.column_stack(A_cols)  # (m*n) x numrec
    b = vec((X > 0).astype(int)).astype(float)
    numrec = A.shape[1]
    # MILP: min 1^T z s.t. A z >= b, 0<=z<=1, z integer
    if SCIPY_OK:
        c = np.ones(numrec, dtype=float)
        integrality = np.ones(numrec, dtype=int)
        # Constraint is A z >= b → multiply by -1 to fit Ax <= ub
        lc = LinearConstraint(-A, lb=-np.inf*np.ones(m*n), ub=-b)
        bounds = Bounds(0, 1)
        res = milp(c=c, constraints=[lc], integrality=integrality, bounds=bounds)
        if res.x is None:
            # Fallback to greedy if MILP did not find a solution (e.g., missing solver)
            z = np.zeros(numrec, dtype=int)
            covered = (A @ z) >= b
            while not np.all(covered):
                gains = (A.T @ (~covered).astype(int))
                idx = int(np.argmax(gains))
                if gains[idx] == 0:
                    break
                z[idx] = 1
                covered = (A @ z) >= b
        else:
            z = res.x.round().astype(int)
    else:
        # Greedy fallback (may be suboptimal)
        z = np.zeros(numrec, dtype=int)
        covered = (A @ z) >= b
        while not np.all(covered):
            gains = (A.T @ (~covered).astype(int))
            idx = int(np.argmax(gains))
            if gains[idx] == 0:
                break
            z[idx] = 1
            covered = (A @ z) >= b
    idxs = np.where(z == 1)[0]
    rec = [A[:, i].reshape((m, n), order="F") for i in idxs]
    rcX = len(rec)
    if transposed:
        rec = [r.T for r in rec]
    return rcX, rec


def geometric_bound(rankX: int, restrnnrankX: int) -> int:
    """
    Faithful port of geometric_bound.m
    """
    def comb(a: int, b: int) -> int:
        if a < 0 or b < 0 or a - b < 0:
            return 0
        b = min(b, a - b)
        c = 1
        t = 1
        for i in range(a - b + 1, a + 1):
            c = c * i // t
            t += 1
        return c

    def faces(n: int, d: int, k: int) -> int:
        if d % 2 == 0:
            s = 0
            l = d // 2
            for i in range(0, l):
                s += (comb(d - i, k + 1 - i) + comb(i, k + 1 - d + i)) * comb(n - d - 1 + i, i)
            s += (comb(d - l, k + 1 - l) + comb(l, k + 1 - d + l)) * comb(n - d - 1 + l, l) // 2
            return s
        else:
            s = 0
            l = d // 2
            for i in range(0, l + 1):
                s += (comb(d - i, k + 1 - i) + comb(i, k + 1 - d + i)) * comb(n - d - 1 + i, i)
            return s

    def phifct(r: int, rp: int) -> int:
        phirrp = rp
        for rw in range(r + 1, rp + 1):
            if r > 3:
                phirrp = max(phirrp, faces(rp, rw - 1, rw - r))
            else:
                phirrp = max(phirrp, min(faces(rp, rw - 1, rw - r), faces(rp, rw - 1, rw - r + 1)))
        return phirrp

    lowerbnd = rankX
    while phifct(rankX, lowerbnd) < restrnnrankX:
        lowerbnd += 1
    return lowerbnd


def hyperplane_separation_bound(Z: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Faithful brute-force for small sizes (min(m,n) <= 14):
    alpha = max_{binary rank-1 R} <Z, R>
    """
    m, n = Z.shape
    if min(m, n) > 14:
        print("Warning: min(m,n) is large; this may be slow.")
    alphaZ = -np.inf
    best_x = None
    best_y = None
    # Enumerate all binary row masks
    Xbin = np.array([[(i >> k) & 1 for k in range(m)] for i in range(2**m)], dtype=int)
    for i in range(2**m):
        xpos = np.where(Xbin[i] > 0)[0]
        if xpos.size == 0:
            continue
        if xpos.size > 1:
            valcol = np.sum(Z[xpos, :], axis=0)
        else:
            valcol = Z[xpos[0], :]
        ypos = np.where(valcol > 0)[0]
        y = np.zeros(n, dtype=int)
        y[ypos] = 1
        vali = float(np.sum(valcol[ypos]))
        if vali > alphaZ:
            alphaZ = vali
            best_x = Xbin[i]
            best_y = y
    return alphaZ, best_x, best_y


def nonneg_nuclear_norm_bound(X: np.ndarray) -> float:
    if not CVXPY_OK:
        print("CVXPY not installed; skipping nonnegative nuclear norm bound.")
        return float("nan")
    m, n = X.shape
    Y = cp.Variable((m, m), PSD=True)
    Z = cp.Variable((n, n), PSD=True)
    B = cp.bmat([[Y, X], [X.T, Z]])
    constraints = [B >> 0, B >= 0]
    obj = cp.Minimize(0.5 * (cp.trace(Y) + cp.trace(Z)))
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.SCS, verbose=False)
    nu = 0.5 * (Y.value.trace() + Z.value.trace())
    lb = (nu / np.linalg.norm(X, "fro")) ** 2
    return float(lb)


def self_scaled_bound(X: np.ndarray) -> float:
    if not CVXPY_OK:
        print("CVXPY not installed; skipping self-scaled bound.")
        return float("nan")
    m, n = X.shape
    Y = cp.Variable((m * n, m * n), PSD=True)
    tausos = cp.Variable(nonneg=True)
    # Build block matrix [[tausos, vec(X)'], [vec(X), Y]] >= 0
    vx = vec(X)[:, None]
    top = cp.hstack([cp.reshape(tausos, (1, 1)), vx.T])
    bot = cp.hstack([vx, Y])
    M = cp.vstack([top, bot])
    constr = [M >> 0]
    # Additional diagonal and symmetry constraints
    for i in range(m):
        for j in range(n):
            idx = i + j * m
            constr.append(Y[idx, idx] <= X[i, j] ** 2)
            for k in range(i + 1, m):
                for l in range(j + 1, n):
                    idx1 = i + (l) * m
                    idx2 = k + (j) * m
                    idx3 = i + (j) * m
                    idx4 = k + (l) * m
                    # Y(i,(j-1)m+k,(l-1)m) == Y(i,(l-1)m,k,(j-1)m) translated
                    constr.append(Y[idx3, idx4] == Y[idx1, idx2])
    prob = cp.Problem(cp.Minimize(tausos), constr)
    prob.solve(solver=cp.SCS, verbose=False)
    return float(tausos.value)

