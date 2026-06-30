from __future__ import annotations

import os
import sys
import numpy as np

BASE = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE not in sys.path:
    sys.path.insert(0, BASE)
sys.path.insert(0, os.path.dirname(__file__))
from isSSC_full import ssc1_nec_cond


def main():
    # Example matrix H to test SSC1 necessary condition (build to mirror .m)
    # Users can edit H to reproduce MATLAB tests
    np.set_printoptions(linewidth=200, suppress=True)
    H = np.array([
        [1, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
    ], dtype=float)
    print("H =")
    print(H)
    print("ssc1_nec_cond(H) =")
    print(ssc1_nec_cond(H))


if __name__ == "__main__":
    main()

