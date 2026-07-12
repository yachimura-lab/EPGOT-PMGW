# EPOT-pMGW — Entropic Partial Optimal Transport

This repository publishes research code and notebooks on entropic partial optimal transport (EPOT) between Gaussian mixture models (GMMs). The main logic lives in `src/`, together with Jupyter notebooks that reproduce the figures in the paper.

📖 **Rendered notebooks (with outputs)**: https://epot-pmgw.netlify.app/

## Notebooks

Each notebook corresponds to figures in the paper. You can browse them directly on GitHub, or view the fully rendered versions on the documentation site.

| Notebook | Rendered page | Description |
| --- | --- | --- |
| [figure1_2_3.ipynb](docs/notebook/figure1_2_3.ipynb) | [View online](https://epot-pmgw.netlify.app/notebook/figure1_2_3) | Visualization of transport plans from entropic partial OT between GMMs (Figures 1–3) |
| [figure4.ipynb](docs/notebook/figure4.ipynb) | [View online](https://epot-pmgw.netlify.app/notebook/figure4) | Barycentric interpolation of GMMs based on partial OT (Figure 4, uses `cpp_engine`) |
| [figure5.ipynb](docs/notebook/figure5.ipynb) | [View online](https://epot-pmgw.netlify.app/notebook/figure5) | Transport maps between point clouds via barycentric projection (Figure 5) |

## Repository layout

```
.
├── src/
│   ├── epot/               # Main logic (Python package)
│   │   └── __init__.py     #   GMM classes, Gaussian W2, MGW2 / MEW2,
│   │                       #   entropic partial OT, barycentric projection maps
│   ├── computation/        # C++ computation engine (Eigen + pybind11)
│   │   ├── engine.cpp      #   Fast GaussianW2 / W2 barycenter implementations
│   │   └── CMakeLists.txt  #   Builds the cpp_engine module into src/
│   ├── figure1_2_3.ipynb   # Notebook reproducing Figures 1–3
│   ├── figure4.ipynb       # Notebook reproducing Figure 4 (uses cpp_engine)
│   └── figure5.ipynb       # Notebook reproducing Figure 5
├── docs/                   # Quarto documentation site (published on Netlify)
│   └── notebook/           #   Notebooks published on the site
├── pyproject.toml          # Python dependencies (managed with uv)
└── netlify.toml            # Site build / deploy configuration
```

## Setup

### Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (for dependency management)
- CMake, a C++17 compiler, and Eigen3 (to build the C++ engine)

If uv is not installed yet, you can install it with the bundled script:

```bash
./setup-uv.sh
```

### 1. Install Python dependencies

```bash
uv sync
```

### 2. Build the C++ computation engine (cpp_engine)

This builds the fast computation module used by `figure4.ipynb` and others. First install Eigen3 (example for Fedora / Amazon Linux):

```bash
sudo dnf install -y eigen3-devel
```

Then build with CMake. The resulting `cpp_engine` module is placed directly under `src/`, so the notebooks can import it as-is:

```bash
rm -rf src/computation/build

UV_PYTHON="$(uv run python -c 'import sys; print(sys.executable)')"
PYBIND11_DIR="$(uv run python -m pybind11 --cmakedir)"
cmake -S src/computation \
      -B src/computation/build \
      -Dpybind11_DIR="$PYBIND11_DIR" \
      -DPython_EXECUTABLE="$UV_PYTHON"

cmake --build src/computation/build
```

## Running the notebooks

Open the notebooks under `src/` with Jupyter:

```bash
uv run jupyter lab src/
```

## Building the documentation site (for developers)

The site is built with [Quarto](https://quarto.org/) and deployed on Netlify (see `netlify.toml`). To preview locally:

```bash
export QUARTO_PYTHON=$(pwd)/.venv/bin/python
quarto preview docs
```

Notebooks to be published on the site go under `docs/notebook/`.
