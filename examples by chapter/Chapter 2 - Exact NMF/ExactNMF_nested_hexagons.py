from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

BASE = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from algorithms.exact_nmf import exact_nmf_heuristic, ExactNMFOptions
# Local import of NPPrank3matrix port
from NPPrank3matrix import npp_rank3_matrix


def build_Xa(a: float) -> tuple[np.ndarray, int]:
    if not np.isfinite(a):
        Xa = np.array(
            [
                [0, 1, 2, 2, 1, 0],
                [0, 0, 1, 2, 2, 1],
                [1, 0, 0, 1, 2, 2],
                [2, 1, 0, 0, 1, 2],
                [2, 2, 1, 0, 0, 1],
                [1, 2, 2, 1, 0, 0],
            ],
            dtype=float,
        )
        r = 5
        return Xa, r
    Xa = np.array(
        [
            [1, a, 2 * a - 1, 2 * a - 1, a, 1],
            [1, 1, a, 2 * a - 1, 2 * a - 1, a],
            [a, 1, 1, a, 2 * a - 1, 2 * a - 1],
            [2 * a - 1, a, 1, 1, a, 2 * a - 1],
            [2 * a - 1, 2 * a - 1, a, 1, 1, a],
            [a, 2 * a - 1, 2 * a - 1, a, 1, 1],
        ],
        dtype=float,
    )
    if a <= 2:
        r = 3
    elif a <= 3:
        r = 4
    else:
        r = 5
    return Xa, r


def main():
    figs = os.path.join(BASE, "figs")
    os.makedirs(figs, exist_ok=True)

    a = 3.0
    Xa, r = build_Xa(a)

    # Exact NMF heuristic
    opts = ExactNMFOptions(tolerance=1e-9, random_state=1, display=1)
    W, H, _, _ = exact_nmf_heuristic(Xa, r, opts)

    # Swap for 2D display if needed (mirror MATLAB logic)
    sw = np.linalg.svd(W, compute_uv=False)
    sh = np.linalg.svd(H, compute_uv=False)
    tol = 1e-6
    if (sw.size > 3 and sw[3] > tol) and (sh.size >= 4 and sh[3] < tol):
        Xa = Xa.T
        Wold = W.copy()
        W = H.T
        H = Wold.T
        sw = sh

    # NPP instance visualization
    P = U = V = None
    try:
        P, U, V = npp_rank3_matrix(Xa)
    except Exception:
        P = U = V = None
    # Plot outer polygon and inner polygon exactly like MATLAB
    if P is not None and V is not None and P.shape[1] >= 3:
        Kp = ConvexHull(P.T).vertices
        plt.figure()
        plt.plot(P[0, Kp], P[1, Kp], "rx-", markersize=15, label=r"Outer polygon $\mathcal{B}$")
        if V.shape[0] >= 2 and V.shape[1] >= 3:
            Kv = ConvexHull(V[:2, :].T).vertices
            plt.plot(V[0, Kv], V[1, Kv], "bo-.", markersize=12, label=r"Inner polygon $\mathcal{A}$")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(figs, "exactnmf_nested_hexagons_polygons.pdf"), bbox_inches="tight")
        plt.close()

    # If rank(W)<=3, show conv(W) in 2D via C = U\W
    if sw.size == 3 or (sw.size > 3 and sw[3] < tol):
        # Normalize columns of W and adjust H exactly like MATLAB
        sumWcol = np.sum(W, axis=0)
        sumWcol[sumWcol <= 1e-16] = 1.0
        W = W * (1.0 / sumWcol)
        H = (sumWcol.reshape(-1, 1)) * H
        try:
            Uuse = U if U is not None else Xa[:, :3]
            # C = U\W (least-squares)
            C = np.linalg.lstsq(Uuse, W, rcond=None)[0]
            if C.shape[0] >= 2 and C.shape[1] >= 3:
                Kc = ConvexHull(C[:2, :].T).vertices
                plt.figure()
                if P is not None and P.shape[1] >= 3:
                    Kp = ConvexHull(P.T).vertices
                    plt.plot(P[0, Kp], P[1, Kp], "rx-", markersize=15, label=r"$\Delta^6 \cap$ col$(X_a)$")
                plt.plot(C[0, Kc], C[1, Kc], "ks--", markersize=15, label=r"conv$(W)$")
                plt.axis("equal")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(figs, "exactnmf_nested_hexagons_convW.pdf"), bbox_inches="tight")
                plt.close()
        except Exception:
            pass

    print("Saved Exact NMF nested hexagons figures in", figs)


if __name__ == "__main__":
    main()

