from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from algorithms.hierarchical_nmf.port import splitclust_port, reprvec_port
from algorithms.nmf.nnls import NNLSOptions, nnls


@dataclass
class HierNMFOptions:
    max_splits: Optional[int] = None  # if None, use r-1
    random_state: int = 0
    display: int = 1


def _cluster_error(X: np.ndarray, idx: np.ndarray, w: np.ndarray) -> float:
    if idx.size == 0:
        return 0.0
    H, _, _ = nnls(w.reshape(-1, 1), X[:, idx], NNLSOptions(algo="HALS", inneriter=500, delta=1e-6))
    R = X[:, idx] - w.reshape(-1, 1) @ H
    return float(np.linalg.norm(R, ord="fro") ** 2)


def hierclust2nmf(X: np.ndarray, r: int, options: Optional[HierNMFOptions] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hierarchical rank-two NMF clustering: recursively splits the set of columns using ONMF (r=2),
    collects r leaf clusters, and builds W from the resulting two-way bases; final H is NNLS with W fixed.
    """
    if options is None:
        options = HierNMFOptions()
    m, n = X.shape
    rng = np.random.default_rng(options.random_state)

    # Start with a single cluster containing all indices; basis via reprvec
    u0, _ = reprvec_port(X)
    clusters: List[dict] = [{"idx": np.arange(n), "w": u0}]

    num_splits = 0
    target_leaves = r
    max_splits = options.max_splits if options.max_splits is not None else (r - 1)

    while len(clusters) < target_leaves and num_splits < max_splits:
        # Choose cluster to split: largest error
        errs = [ _cluster_error(X, c["idx"], c["w"]) for c in clusters ]
        split_i = int(np.argmax(errs))
        c = clusters.pop(split_i)
        idx = c["idx"]
        if idx.size <= 2:
            # Cannot split meaningfully; put back and stop
            clusters.append(c)
            break
        # Rank-two split using MATLAB-like splitclust
        Kc, Uc, _ = splitclust_port(X[:, idx])
        idx0 = idx[Kc[0]]
        idx1 = idx[Kc[1]]
        # Basis via reprvec per child
        w0, _ = reprvec_port(X[:, idx0]) if idx0.size > 0 else (np.zeros(m), 0.0)
        w1, _ = reprvec_port(X[:, idx1]) if idx1.size > 0 else (np.zeros(m), 0.0)
        if idx0.size > 0:
            clusters.append({"idx": idx0, "w": w0})
        if idx1.size > 0:
            clusters.append({"idx": idx1, "w": w1})
        num_splits += 1

    # If fewer than r clusters, pad with residual means
    while len(clusters) < r:
        clusters.append({"idx": np.array([], dtype=int), "w": np.maximum(1e-16, rng.random(m))})

    # Build W from cluster bases (first r)
    W = np.zeros((m, r))
    for k in range(r):
        W[:, k] = clusters[k]["w"]
        # Normalize columns to unit max
        mx = max(1e-16, float(np.max(W[:, k])))
        W[:, k] /= mx

    # Final H via NNLS on full X
    H, _, _ = nnls(W, X, NNLSOptions(algo="HALS", inneriter=1000, delta=1e-6))
    return W, H

