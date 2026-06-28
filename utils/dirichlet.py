from __future__ import annotations

from typing import Optional

import numpy as np


def sample_dirichlet(alpha: np.ndarray, n_samples: int = 1, *, random_state: Optional[int] = None) -> np.ndarray:
    """
    Draw samples from a Dirichlet distribution.

    Parameters
    ----------
    alpha : np.ndarray
        Concentration parameters (shape (k,) or (k, )), all > 0.
    n_samples : int
        Number of samples to draw.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    samples : np.ndarray
        Array of shape (k, n_samples) with columns summing to 1.
    """
    alpha = np.asarray(alpha, dtype=float).reshape(-1)
    if np.any(alpha <= 0):
        raise ValueError("All alpha entries must be > 0")
    rng = np.random.default_rng(random_state)
    samples = rng.dirichlet(alpha, size=n_samples)  # (n_samples, k)
    return samples.T  # (k, n_samples) to match column-wise MATLAB style

