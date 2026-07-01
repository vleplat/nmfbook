"""
Global warning silencer for example runs.

Python automatically imports 'sitecustomize' on startup if found on sys.path.
Since examples are run from the repository root, this file applies to all demos,
reducing noisy numerical warnings without changing algorithm outputs.
"""
from __future__ import annotations

import warnings
import numpy as np

# Silence common benign numerical warnings from NumPy/SciPy during demo runs.
np.seterr(all="ignore")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

