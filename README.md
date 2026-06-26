# NMF Book (Python) - Algorithms, Datasets, and Examples

### Project Description

This repository is a Python translation of the MATLAB companion code for the NMF book written by Nicolas Gillis.

It follows the original structure and re-implements core algorithms and demos so you can reproduce figures and experiments directly in Python.

### Reference book

<p align="center">
  <img src="assets/nmf_book_frontPage.png"
       alt="Front page of Nonnegative Matrix Factorization by Nicolas Gillis"
       width="280">
</p>

This repository is based on the MATLAB companion code for the book:

> Nicolas Gillis, *Nonnegative Matrix Factorization*, SIAM, Philadelphia, 2020.

```bibtex
@book{gillis2020nonnegative,
  title     = {Nonnegative Matrix Factorization},
  author    = {Gillis, Nicolas},
  year      = {2020},
  publisher = {SIAM, Philadelphia}
}
```

The original MATLAB package and the book webpage are available from the author's page:

<https://sites.google.com/site/nicolasgillis/book>

Please report questions about the original MATLAB code to `nicolas.gillis@umons.ac.be`.

Python translation and development: Valentin Leplat (`valentin.leplat@gmail.com`).

### Repository Structure

The codes of this NMF book are divided into 4 folders:

- `algorithms`: contains all the algorithms described in the book, for various NMF models.
- `data sets`: contains all the data sets used in the book.
- `examples by chapter`: contains the numerical examples from the book, classified chapter by chapter. They can be used to generate many figures presented in the book. This folder also contains some algorithms that are not NMF algorithms, for example lower bounds for the nonnegative rank.
- `utils`: contains useful functions used by several algorithms and examples.

### Implemented Algorithms (Python)

- Symmetric NMF via exact coordinate descent (`algorithms/symmetric_nmf/`)
  - Entry points: `symnmf(A, r, options)` and `SymNMFOptions`
- Frobenius NMF, 2-BCD with extrapolation (`algorithms/nmf/`)
  - Entry points: `fro_nmf(X, r, options)` and `FroNMFOptions`
  - Includes HALS/MU-based NNLS updates
- β-NMF with multiplicative updates and extrapolation (`algorithms/beta_nmf/`)
  - Entry points: `beta_nmf(X, r, options)` and `BetaNMFOptions`

Notes:

- Many MATLAB helpers are being ported progressively into Python under `utils/`.
- Some algorithms have optional coupled rescaling to stabilize magnitudes without changing the product `WH`, through the `rescale_every` option.

---

### Quick Start

This section explains how to install the Python version of the NMF book code from scratch.

#### Step 1: Install Python 3.9+

You need Python 3.9 or newer.

Check whether Python is already installed:

```bash
python3 --version
```

If this does not work, try:

```bash
python --version
```

On Windows, open PowerShell and try:

```powershell
py --version
python --version
```

If Python is not installed, download it from:

<https://www.python.org/downloads/>

During the installation on Windows, it is recommended to select:

```text
Add Python to PATH
```

#### Step 2: Download this repository

There are two simple ways to get the code.

##### Option A: clone with git

This is the recommended option if you already use Git.

```bash
git clone <repository-url>
cd <repository-folder>
```

For example:

```bash
git clone https://github.com/<your-github-username>/<repository-name>.git
cd <repository-name>
```

On Windows PowerShell, the commands are the same:

```powershell
git clone https://github.com/<your-github-username>/<repository-name>.git
cd <repository-name>
```

Replace the repository address above by the actual URL of this GitHub repository.

##### Option B: download ZIP

This option does not require Git.

1. Open the GitHub repository page.
2. Click **Code**.
3. Click **Download ZIP**.
4. Extract the ZIP file.
5. Open a terminal in the extracted folder.

To open a terminal in the extracted folder:

- On macOS or Linux, right-click in the folder and choose **Open in Terminal**, or open a terminal and use `cd`.
- On Windows, open the folder, click the address bar, type `powershell`, and press Enter.

#### Step 3: Create a virtual environment

A virtual environment keeps the Python packages for this project separate from the rest of your system.

From the repository root, run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

On Windows `cmd.exe`:

```bat
py -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
```

If PowerShell refuses to activate the environment, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then try again:

```powershell
.\.venv\Scripts\Activate.ps1
```

#### Step 4: Install the required packages

Once the virtual environment is activated, install the required Python packages:

```bash
python -m pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m pip install -r requirements.txt
```

#### Step 5: Run a first example

You can now run one of the examples.

For example, run the symmetric NMF random demo:

```bash
python -m algorithms.symmetric_nmf.runme_random
```

You can also run one of the chapter examples:

```bash
python "examples by chapter/Chapter 1 - Introduction/CBCL.py"
```

Generated figures are saved under `figs/`.

#### Optional: headless mode

If you run the code on a server, or in an environment without a graphical interface, set a non-interactive plotting backend.

On macOS or Linux:

```bash
export MPLBACKEND=Agg
```

On Windows PowerShell:

```powershell
$env:MPLBACKEND = "Agg"
```

#### Leaving the virtual environment

When you are done, run:

```bash
deactivate
```

---

### How to Use

You can import and use algorithms in your own scripts, or run the provided chapter demos.

The examples compute the project base path dynamically, so you can run them from anywhere.

#### Symmetric NMF, random demo

```bash
source .venv/bin/activate
python -m algorithms.symmetric_nmf.runme_random
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m algorithms.symmetric_nmf.runme_random
```

Figures are saved under `figs/`.

#### Chapter 1 - CBCL faces, Frobenius NMF

```bash
source .venv/bin/activate
python "examples by chapter/Chapter 1 - Introduction/CBCL.py"
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python "examples by chapter/Chapter 1 - Introduction/CBCL.py"
```

This generates a grid of sample faces, the average face, basis faces from NMF, and the error-vs-time plot in `figs/`.

#### Chapter 1 - Mary has a little lamb, β-NMF with KL divergence

```bash
source .venv/bin/activate
python "examples by chapter/Chapter 1 - Introduction/Mary_piano.py"
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python "examples by chapter/Chapter 1 - Introduction/Mary_piano.py"
```

This generates activation plots, frequency-response plots, and the objective value per iteration in `figs/`.

#### Chapter 5 - Karate graph, symmetric NMF demo

```bash
source .venv/bin/activate
python "examples by chapter/Chapter 5 - NMF models/symNMF_karate.py"
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python "examples by chapter/Chapter 5 - NMF models/symNMF_karate.py"
```

This generates:

- rank-1 term heatmaps,
- the adjacency matrix, before and after community reordering,
- the `H H^T` heatmap,
- a community graph visualization.

All figures are saved under `figs/`.

---

### Data

MAT files are provided in `data sets/`.

The examples load these files directly, for example:

- `CBCL.mat`
- `piano_Mary.mat`
- `karate.mat`

If you add new datasets, place them in `data sets/` and adjust the example scripts if variable names differ from expectations.

---

### Troubleshooting

#### `ModuleNotFoundError: No module named 'algorithms'`

Make sure that you run the examples from the repository root, or that the repository root is on your Python path.

The provided examples automatically add the repository root to `sys.path`. If you run Python from another folder or from your own scripts, you may need to run from the repository root or set `PYTHONPATH`.

On macOS or Linux:

```bash
export PYTHONPATH=/path/to/repository
```

On Windows PowerShell:

```powershell
$env:PYTHONPATH = "C:\path\to\repository"
```

#### Plotting does not work on a server

If you run the code in a headless environment, set:

```bash
export MPLBACKEND=Agg
```

On Windows PowerShell:

```powershell
$env:MPLBACKEND = "Agg"
```

#### Large-value runtime warnings in β=1, KL updates

Occasional `divide` or `overflow` warnings can happen when `WH` has very small entries.

Possible fixes:

- increase `epsilon` in `BetaNMFOptions`, for example to `1e-8`,
- enable `rescale_every`,
- wrap KL computations with `np.errstate` in custom scripts.

#### The virtual environment is not activated

After activation, your terminal prompt usually starts with:

```text
(.venv)
```

You can also check which Python is used.

On macOS or Linux:

```bash
which python
```

On Windows PowerShell:

```powershell
where python
```

---

### Contributing

- Python code mirrors the MATLAB structure as much as possible.
- When porting functions, keep interfaces and behavior faithful to the original code.
- Add small numerical safeguards where needed.
- Use a local virtual environment.
- Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

- Save figures to `figs/` with `bbox_inches="tight"` for consistency.

---

### Credits

- Original MATLAB code and book: Nicolas Gillis and collaborators.
- Python translation and development: Valentin Leplat (`valentin.leplat@gmail.com`).
- For questions about the original MATLAB code: `nicolas.gillis@umons.ac.be`.
