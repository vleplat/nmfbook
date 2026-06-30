from __future__ import annotations

import os
import sys
import time
import numpy as np

BASE = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

sys.path.insert(0, os.path.dirname(__file__))
from importlib import import_module
plb = import_module("py_lower_bounds")
rec_cov_bound = plb.rec_cov_bound


def main():
    t0 = time.perf_counter()
    n = 6
    X = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            X[i, j] = float((i + 1 - (j + 1)) ** 2)
    print("X =\n", X)
    rcX, rec = rec_cov_bound(X)
    print(f"The rectangle covering bound of X is {rcX:.0f}.")
    print("The rectangles covering X are:")
    for i in range(rcX):
        print(f"rec[{i+1}]:\n{rec[i]}")
    print(f"Elapsed {time.perf_counter() - t0:.2f}s.")


if __name__ == "__main__":
    main()

