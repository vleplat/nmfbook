from __future__ import annotations

import warnings
import numpy as np


def silence_numpy_warnings() -> None:
    """
    Silence common benign numerical warnings for demos:
    - numpy floating warnings
    - runtime/user warnings from libs
    """
    np.seterr(all="ignore")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

