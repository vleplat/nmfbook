from __future__ import annotations

import os
import sys
import numpy as np

BASE = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

sys.path.insert(0, os.path.dirname(__file__))
from isSSC_full import isSSC_full


def main():
    # Example 4.29 - sufficiently scattered condition (4x6, 2-sparse)
    H = np.array([
        [1, 1, 1, 0, 0, 0],
        [1, 0, 0, 1, 1, 0],
        [0, 1, 0, 1, 0, 1],
        [0, 0, 1, 0, 1, 1],
    ], dtype=float)
    print("H =")
    print(H)
    ssc1, ssc2, xs, ssc1nec = isSSC_full(H)
    if ssc1nec == 0:
        print("H does not satisfy the necessary condition for the SSC.")
        return
    print("An optimal vertex xs of max ||x||^2 s.t. H^T x >= 0, e^T x = 1:")
    print(xs)
    if ssc1 == 1:
        print(f"H satisfies SSC1 because ||xs||^2 = {np.linalg.norm(xs)**2:.2f} <= 1.")
    else:
        print(f"H does not satisfy SSC1 because ||xs||^2 = {np.linalg.norm(xs)**2:.2f} > 1.")
    if ssc2 == 1:
        print("H satisfies SSC2 because xs can only be a unit vector.")
    else:
        print("H does not satisfy SSC2 because xs is not a unit vector.")


if __name__ == "__main__":
    main()

