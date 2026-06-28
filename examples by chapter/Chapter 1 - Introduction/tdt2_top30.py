from __future__ import annotations

import os
import sys
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from algorithms.nmf import fro_nmf, FroNMFOptions


def main():
    data_path = os.path.join(BASE, "data sets", "tdt2_top30.mat")
    mat = sio.loadmat(data_path)
    # Common variable names for document-term matrix
    X = None
    for key in ("X", "Xtdt2", "dtm"):
        if key in mat:
            raw = mat[key]
            try:
                import scipy.sparse as sp
                if sp.issparse(raw):
                    X = raw.astype(np.float64).toarray()
                else:
                    X = np.array(raw, dtype=np.float64)
            except Exception:
                X = np.array(raw, dtype=np.float64)
            break
    if X is None:
        raise KeyError("tdt2_top30.mat must contain 'X' (or 'Xtdt2', 'dtm')")
    X = np.maximum(X, 0.0)

    figs = os.path.join(BASE, "figs")
    os.makedirs(figs, exist_ok=True)

    # Match MATLAB: transpose to documents-by-words
    X = X.T
    # Seed to match MATLAB rng(2020)
    np.random.seed(2020)

    r = 20
    # Use FroNMF with default options, as in MATLAB
    opts = FroNMFOptions()
    W, H, e, t, _ = fro_nmf(X, r, opts)

    # Improved visualization of topic activations (H):
    # - sort documents by their argmax topic
    # - normalize each topic row to [0,1] for contrast
    doc_groups = np.argmax(H, axis=0)
    order = np.argsort(doc_groups, kind="stable")
    H_sorted = H[:, order]
    H_norm = H_sorted / (np.max(H_sorted, axis=1, keepdims=True) + 1e-16)
    plt.figure(figsize=(12, 5))
    plt.imshow(H_norm, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    # draw separators between topic groups
    last = -1
    for k in range(r):
        right = np.searchsorted(doc_groups[order], k, side="right")
        if right != last:
            plt.axvline(x=right - 0.5, color="white", alpha=0.3, linewidth=0.6)
            last = right
    plt.colorbar(shrink=0.7, label="row-normalized activation")
    plt.xlabel("documents (grouped by dominant topic)")
    plt.ylabel("topics")
    plt.title("TDT2 - topic activations (row-normalized, grouped)")
    plt.tight_layout()
    plt.savefig(os.path.join(figs, "tdt2_topics_heatmap.pdf"), bbox_inches="tight")
    plt.close()

    # If words are available, show top-10 words per topic as bar charts (similar to MATLAB listing)
    words = None
    if "words" in mat:
        try:
            arr = mat["words"]
            # Flatten and decode MATLAB cell array of strings
            arr = np.squeeze(arr)
            words = []
            for w in arr:
                if isinstance(w, np.ndarray):
                    # typical case: array(['word'], dtype='<U...') or object array
                    words.append(str(np.squeeze(w)))
                else:
                    words.append(str(w))
        except Exception:
            words = None
    if words is not None and len(words) == W.shape[0]:
        cols = 5
        rows = int(np.ceil(r / cols))
        plt.figure(figsize=(cols * 3.4, rows * 2.6))
        for k in range(r):
            ax = plt.subplot(rows, cols, k + 1)
            col = W[:, k]
            top_idx = np.argsort(-col)[:10]
            top_vals = col[top_idx]
            # normalize within topic for readability
            top_vals = top_vals / (np.max(top_vals) + 1e-16)
            labels = [words[j] for j in top_idx]
            # horizontal bar chart, most important at top
            order10 = np.argsort(top_vals)
            ax.barh(np.arange(10), top_vals[order10], color="#4e79a7")
            ax.set_yticks(np.arange(10))
            ax.set_yticklabels([labels[i] for i in order10], fontsize=7)
            ax.set_xlim(0, 1.05)
            ax.set_title(f"Topic {k+1}", fontsize=9)
            ax.grid(axis="x", alpha=0.2)
        plt.tight_layout()
        plt.savefig(os.path.join(figs, "tdt2_topics_top10_words.pdf"), bbox_inches="tight")
        plt.close()

    # Objective
    plt.figure()
    plt.plot(t, e)
    plt.xlabel("Time (s.)")
    plt.ylabel("Relative error ||X-WH||_F / ||X||_F")
    plt.title("TDT2 - FroNMF objective")
    plt.tight_layout()
    plt.savefig(os.path.join(figs, "tdt2_error_vs_time.pdf"), bbox_inches="tight")
    plt.close()
    print("Saved TDT2 figures in", figs)


if __name__ == "__main__":
    main()

