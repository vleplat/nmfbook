from __future__ import annotations

import numpy as np


def sumte(em: np.ndarray, tm: np.ndarray, tmf: np.ndarray) -> np.ndarray:
    """
    Interpolate error values em observed at times tm onto the target time grid tmf.
    Faithful port of utils/sumte.m used for averaging curves over time budgets.
    Assumes max(tm) >= max(tmf).
    """
    em = np.asarray(em, dtype=float).ravel()
    tm = np.asarray(tm, dtype=float).ravel()
    tmf = np.asarray(tmf, dtype=float).ravel()
    if tm.size == 0 or tmf.size == 0:
        return np.array([], dtype=float)
    timemax = float(np.max(tmf))
    if float(np.max(tm)) < timemax:
        raise ValueError("max(tm) should be larger than max(tmf).")
    # Handle equality as in MATLAB
    if float(np.max(tm)) == timemax:
        tm = tm.copy()
        tm[-1] = timemax + 1e-6
    emf = np.empty_like(tmf)
    emf[0] = em[0]
    cnt = 0
    for i in range(1, tmf.size):
        while cnt < tm.size and tm[cnt] <= tmf[i]:
            cnt += 1
        if cnt == 0:
            # no errors before tmf[1]
            emf[i] = em[0]
        else:
            # linear interpolation between (tm[cnt-1], em[cnt-1]) and (tm[cnt], em[cnt])
            # guard cnt==tm.size in pathological cases
            if cnt >= tm.size:
                cnt = tm.size - 1
            t0 = tm[cnt - 1]
            t1 = tm[cnt]
            lam = (tmf[i] - t0) / max(t1 - t0, 1e-16)
            emf[i] = lam * em[cnt] + (1.0 - lam) * em[cnt - 1]
    return emf

