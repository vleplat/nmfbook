from __future__ import annotations

from typing import Iterable, Optional, Tuple

import math
import numpy as np
import matplotlib.pyplot as plt


def affichage(
    X: np.ndarray,
    image_shape: Tuple[int, int],
    *,
    ncols: int = 8,
    cmap: str = "gray",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    titles: Optional[Iterable[str]] = None,
    suptitle: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = False,
    tight: bool = True,
) -> None:
    """
    Display the columns of X as images laid out in a grid, similar to MATLAB affichage.m.

    Parameters
    ----------
    X : np.ndarray
        Array of shape (p, n), each column is an image vectorized in column-major order.
    image_shape : (h, w)
        Height and width of each image.
    ncols : int
        Number of columns in the grid.
    cmap : str
        Matplotlib colormap (default 'gray').
    vmin, vmax : float, optional
        Value range for imshow; computed from data if None.
    titles : Iterable[str], optional
        Optional per-image titles.
    suptitle : str, optional
        Figure-level title.
    save_path : str, optional
        If provided, save the figure to this path.
    show : bool
        If True, call plt.show() at the end.
    tight : bool
        If True, apply tight_layout().
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D with shape (p, n)")
    h, w = image_shape
    p, n = X.shape
    if p != h * w:
        raise ValueError(f"Each column should have length {h*w}, got {p}")

    ncols = max(1, int(ncols))
    nrows = math.ceil(n / ncols)

    if vmin is None:
        vmin = float(np.min(X))
    if vmax is None:
        vmax = float(np.max(X))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(1.6 * ncols, 1.6 * nrows))
    axes = np.atleast_2d(axes)

    for idx in range(nrows * ncols):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r, c] if nrows > 1 or ncols > 1 else axes[0, 0]
        ax.axis("off")
        if idx < n:
            img = np.reshape(X[:, idx], (h, w), order="F")
            ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            if titles is not None:
                try:
                    t = list(titles)[idx]
                    ax.set_title(str(t), fontsize=8)
                except Exception:
                    pass
        else:
            ax.imshow(np.zeros((h, w)), cmap=cmap, vmin=vmin, vmax=vmax)

    if suptitle:
        fig.suptitle(suptitle)
    if tight:
        plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

