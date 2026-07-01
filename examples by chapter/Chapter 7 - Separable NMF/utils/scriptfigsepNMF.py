from __future__ import annotations

import os
from typing import List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def scriptfigsepNMF(results: np.ndarray, timings: np.ndarray, delta: np.ndarray, nalgo: List[int], xp: int, figs_dir: str) -> None:
    """
    Plot success rate vs noise level for selected algorithms.
    Saves a PDF per experiment.
    """
    labels = {
        1: "SPA",
        2: "VCA",
        3: "FastAnchorWords",
        4: "SNPA",
        5: "MVE-SPA",
        6: "SPA-SPA",
    }
    markers = {
        1: "o-",
        2: "s-",
        3: "x-",
        4: "d-",
        5: "^-.",
        6: "v-",
    }
    plt.figure()
    for a, lab in labels.items():
        if a in nalgo:
            idx = nalgo.index(a)
            plt.plot(delta, results[idx, :], markers.get(a, "-"), label=lab)
    plt.xlabel("noise level (delta)")
    plt.ylabel("success rate")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    os.makedirs(figs_dir, exist_ok=True)
    out = os.path.join(figs_dir, f"ch7_sepNMF_compare_xp{xp}.pdf")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()

