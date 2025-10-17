import numpy as np
import time

from . import symnmf, SymNMFOptions


def main():
    n = 500
    A = np.random.rand(n, n)
    A = A + A.T

    r = 30
    options = SymNMFOptions(
        maxiter=100,
        timelimit=5.0,
        initmatrix="dense01",
        seed=0,
        shuffle_columns=0,
        display="on",
    )

    t0 = time.perf_counter()
    H1, e1, t1 = symnmf(A, r, options)
    tcyc = time.perf_counter() - t0

    options.shuffle_columns = 1
    t0 = time.perf_counter()
    H2, e2, t2 = symnmf(A, r, options)
    tshuf = time.perf_counter() - t0

    print(f"Cyclic time: {tcyc:0.3f}s, Shuffle time: {tshuf:0.3f}s")

    try:
        import os
        import matplotlib.pyplot as plt

        os.makedirs("figs", exist_ok=True)
        plt.figure()
        plt.plot(t1, e1, label="CD-Cyclic-Rand")
        plt.plot(t2, e2, label="CD-Shuffle-Rand")
        plt.legend()
        plt.xlabel("Time (s.)")
        plt.ylabel("Error  -  1/2 * ||A - HH^T||_F^2")
        plt.tight_layout()
        out = "figs/symnmf_random_error_vs_time.pdf"
        plt.savefig(out, bbox_inches="tight")
        print(f"Saved figure to {out}")
    except Exception as exc:
        print("Matplotlib plotting skipped:", exc)


if __name__ == "__main__":
    main()


