## NMF Book (Python) - Algorithms, Datasets, and Examples

### Project Description

This repository is a Python translation of the MATLAB companion code for the NMF book. It follows the original structure and re-implements core algorithms and demos so you can reproduce figures and experiments directly in Python.

The codes of this NMF book are divided in 4 folders:

- `algorithms`: contains all the algorithms described in the book, for various NMF models.
- `data sets`: contains all the data sets used in the book.
- `examples by chapter`: contains the numerical examples from the book, classified chapter by chapter; they can be used to generate many figures presented in the book. It also contains some algorithms that are not NMF algorithms (e.g., lower bounds for the nonnegative rank).
- `utils`: contains some useful functions.

For the original MATLAB package: the book is available for free on the webpage `https://sites.google.com/site/nicolasgillis/book`. Please report bugs to `nicolas.gillis@umons.ac.be`. Python port and contributions by Valentin Leplat (`v.leplat@innopolis.ru`).

### Implemented Algorithms (Python)

- Symmetric NMF via exact coordinate descent (`algorithms/symmetric_nmf/`)
  - Entry points: `symnmf(A, r, options)` and `SymNMFOptions`
- Frobenius NMF (2-BCD with extrapolation) (`algorithms/nmf/`)
  - Entry points: `fro_nmf(X, r, options)` and `FroNMFOptions`
  - Includes HALS/MU-based NNLS updates
- β-NMF with multiplicative updates and extrapolation (`algorithms/beta_nmf/`)
  - Entry points: `beta_nmf(X, r, options)` and `BetaNMFOptions`

Notes:
- Many MATLAB helpers are being ported progressively into Python under `utils/`.
- Some algorithms have optional coupled rescaling to stabilize magnitudes without changing the product `WH` (`rescale_every` option).

### Requirements and Installation

- Python 3.9+
- Recommended: create a local virtual environment at the repo root

Commands:

```bash
cd /path/to/nmfbook
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you run headless (no GUI), you may want to set a non-interactive backend:

```bash
export MPLBACKEND=Agg
```

### How to Use

You can import and use algorithms in your own scripts, or run the provided chapter demos. The examples compute the project base path dynamically, so you can run them from anywhere.

#### Symmetric NMF (random demo)

```bash
source .venv/bin/activate
python -m algorithms.symmetric_nmf.runme_random
```

Figures are saved under `figs/`.

#### Chapter 5 – Karate graph (symmetric NMF demo)

```bash
source .venv/bin/activate
python "examples by chapter/Chapter 5 - NMF models/symNMF_karate.py"
```

This generates:
- Rank-1 term heatmaps, adjacency (raw and community-reordered), `H H^T` heatmap, and a community graph visualization in `figs/`.

#### Chapter 1 – CBCL faces (Frobenius NMF)

```bash
source .venv/bin/activate
python "examples by chapter/Chapter 1 - Introduction/CBCL.py"
```

This generates a grid of sample faces, the average face, basis faces from NMF, and the error-vs-time plot in `figs/`.

#### Chapter 1 – Mary has a little lamb (β-NMF, KL)

```bash
source .venv/bin/activate
python "examples by chapter/Chapter 1 - Introduction/Mary_piano.py"
```

This generates activations (H) and frequency responses (W) plots for r=4 sources, and the objective per iteration in `figs/`.

### Data

MAT files are provided in `data sets/`. The examples load these directly (e.g., `CBCL.mat`, `piano_Mary.mat`, `karate.mat`). If you add new datasets, place them in `data sets/` and adjust the example scripts if variable names differ from expectations.

### Troubleshooting

- Module not found (algorithms): ensure the repo root is on `sys.path`. The examples pre-pend the repo root automatically, but if you run Python from unusual contexts, set `PYTHONPATH` to the repository path or run from the repo root.
- Headless plotting warnings: set `MPLBACKEND=Agg`.
- Large-value runtime warnings in β=1 (KL) updates: occasional `divide/overflow` warnings can happen when `WH` has tiny entries. Increase `epsilon` in `BetaNMFOptions` (e.g., `1e-8`), enable `rescale_every`, or wrap KL computations with `np.errstate` in your custom scripts.

### Contributing

- Python code mirrors the MATLAB structure. When porting functions, keep interfaces and behavior faithful, and add small numerical safeguards where needed.
- Use a local `.venv` and run `pip install -r requirements.txt`.
- Save figures to `figs/` with `bbox_inches="tight"` for consistency.

### Credits

- Original MATLAB code and book: Nicolas Gillis (and collaborators)
- Python translation and development: Valentin Leplat (`valentin.leplat@gmail.com`)
- For questions about the original MATLAB code: `nicolas.gillis@umons.ac.be`
