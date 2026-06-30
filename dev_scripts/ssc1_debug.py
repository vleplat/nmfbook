import numpy as np
import os, sys
BASE = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(BASE, "examples by chapter", "Chapter 4 - Identifiability"))
from isSSC_full import ssc1_nec_cond

def one_trial(r=10, n=100, dens=0.2, seed=0):
    rng = np.random.default_rng(seed)
    mask = (rng.random((r, n)) < dens).astype(float)
    H = mask * rng.random((r, n))
    col_sums = H.sum(axis=0)
    while np.any(col_sums == 0):
        zero_idx = np.where(col_sums == 0)[0]
        mask = (rng.random((r, zero_idx.size)) < dens).astype(float)
        H[:, zero_idx] = mask * rng.random((r, zero_idx.size))
        col_sums = H.sum(axis=0)
    return H

if __name__ == "__main__":
    H = one_trial(10, 100, 0.2, seed=42)
    res = ssc1_nec_cond(H)
    print("ssc1_nec_cond =", res)

