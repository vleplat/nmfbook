from __future__ import annotations

import os
import sys
import numpy as np
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

sys.path.insert(0, os.path.dirname(__file__))
from isSSC_full import ssc1_nec_cond


def main():
    # Silence benign runtime warnings (overflow/invalid during dense ops)
    np.seterr(all="ignore")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    # Figure 4.6 illustration (downsized for speed)
    r_list = list(range(10, 51, 10))      # 10:10:50
    d_list = [round(x, 1) for x in np.arange(0.1, 1.0, 0.1)]  # 0.1:0.1:0.9
    n = 100
    nattemps = int(os.environ.get("NATTEMPTS", "20"))
    print("****************************")
    print(f"     Test for n = {n:2d}   ")
    print("****************************")
    nr = len(r_list)
    nd = len(d_list)
    SSCnec = np.zeros((nr, nd), dtype=float)
    rng = np.random.default_rng(0)
    for ir, r in enumerate(r_list):
        for idd, dens in enumerate(d_list):
            success = 0
            for _ in range(nattemps):
                # Generate H randomly like MATLAB sprand: mask * continuous values
                mask = (rng.random((r, n)) < dens).astype(float)
                H = mask * rng.random((r, n))
                # Resample zero columns
                col_sums = H.sum(axis=0)
                while np.any(col_sums == 0):
                    zero_idx = np.where(col_sums == 0)[0]
                    mask = (rng.random((r, zero_idx.size)) < dens).astype(float)
                    H[:, zero_idx] = mask * rng.random((r, zero_idx.size))
                    col_sums = H.sum(axis=0)
                success += int(ssc1_nec_cond(H) == 1)
            SSCnec[ir, idd] = success / nattemps
            print(f"rank = {r:2d}, density = {dens:1.1f}, number of success = {success}/{nattemps}.")
    print("****************************")
    # Display heatmap
    figs = os.path.join(BASE, "figs")
    os.makedirs(figs, exist_ok=True)
    plt.figure()
    plt.imshow(SSCnec[::-1, :], aspect="auto", cmap="gray")
    plt.colorbar()
    plt.title(f"$n = {n}$")
    plt.yticks(range(nr), r_list[::-1])
    plt.ylabel("rank ($r$)")
    plt.xticks(range(nd), d_list)
    plt.xlabel("density ($d$)")
    plt.tight_layout()
    plt.savefig(os.path.join(figs, "ch4_ssc1_nec_cond_illus.pdf"), bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()

