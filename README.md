# EPGOT-pMGW — Entropic Partial Gromov Optimal Transport

This repository publishes research code and canonical notebooks for Figures
1–7 of the EPGOT/pMGW paper. Numerical pipelines live in `src/pgot/`; executed
notebooks and figure provenance live under `docs/notebook/`.

Rendered notebooks: https://epgot-pmgw.netlify.app/

## Paper figures

| Paper figure | Notebook | Canonical content |
|---|---|---|
| 1 | [figure1_2_3.ipynb](docs/notebook/figure1_2_3.ipynb) | EPOT at λ=0.3, ε=0.01; continuous weight colorbar |
| 2 | [figure1_2_3.ipynb](docs/notebook/figure1_2_3.ipynb) | Exact 1×5 λ sweep at ε=0.01 |
| 3 | [figure1_2_3.ipynb](docs/notebook/figure1_2_3.ipynb) | 3×7 ε/λ sweep with stopping residuals |
| 4 | [figure4.ipynb](docs/notebook/figure4.ipynb) | Sub-probability interpolation (4.18), closed form (4.17) |
| 5 | [figure5.ipynb](docs/notebook/figure5.ipynb) | Matched-density map (6.2), two-row paper layout |
| 6 | [PMGW_corrected.ipynb](docs/notebook/PMGW_corrected.ipynb) | Four barycentric-projection-based nearest assignments |
| 7 | [PMGW_corrected.ipynb](docs/notebook/PMGW_corrected.ipynb) | Raw MGW/pMGW maps from the same solves as Figure 6 |

The machine-readable mapping is
[`figure_manifest.json`](docs/notebook/figure_manifest.json). Each figure has a
unique PDF and JSON sidecar; the sidecar stores source and vendored-solver
revisions, versions, seeds, fitted-input and coupling hashes, paper and solver
parameters, matched mass, residuals, runtime, solve ID, and output SHA-256.

## Repository layout

```text
.
├── src/pgot/                 # Algorithms and canonical figure pipelines
├── docs/notebook/            # Executed notebooks, manifest, sidecars
├── tests/                    # Mathematical/reproducibility regressions
├── tools/build_notebooks.py  # Rebuild concise notebook source cells
├── tools/execute_notebooks.py# Execute all notebooks from clean kernels
├── pyproject.toml            # Exact direct dependency versions
└── uv.lock                   # Complete resolved environment
```

## Setup

Requirements are Python 3.13.15, `uv`, and the checked-out PGW_Metric
submodule.

```bash
git submodule update --init --recursive
uv sync --frozen
```

## Mathematical conventions

Figures 1–5 use `pgot.entropic_partial_ot`, the extended dummy-point problem
(3.1). The real-real cost is `M - 2*Lambda`, the extended marginals are
`(a, 1)` and `(b, 1)`, and entropy is applied to the full extended coupling.
Therefore `Lambda` is exactly the paper's `lambda`, without rescaling.

For Figures 6–7, point and component costs have different normalization
scales. The notebook uses

```text
Lambda_solver = lambda_tilde * gw_mean_distortion(C1, C2, balanced_coupling)
```

The paper's point-level `lambda=0.01` fixes `lambda_tilde`. The canonical
inputs are `0.010000` at point level and `0.013249` at component level; PGW
and pMGW both transport `300/310` mass. Reducing the coupling from `300×310`
to `6×7` is a solver-dimensionality result, not an end-to-end speed claim at
`N=300`, because GMM fitting also takes time.

## Reproducibility

`.python-version`, exact direct requirements in `pyproject.toml`, and
`uv.lock` fix the environment. Notebook sampling uses explicit local NumPy
generators; Figures 6–7 use separate generators for source, target, and noise.
New stochastic APIs should accept `rng=`, and new GMM fits should accept
`random_state=`.

Rebuild and execute all notebooks:

```bash
MPLCONFIGDIR=/tmp/matplotlib uv run python tools/build_notebooks.py
MPLCONFIGDIR=/tmp/matplotlib NUMBA_CACHE_DIR=/tmp/numba \
  uv run python tools/execute_notebooks.py
```

Run regressions:

```bash
MPLCONFIGDIR=/tmp/matplotlib NUMBA_CACHE_DIR=/tmp/numba \
  uv run python -m unittest discover -s tests -v
```

Build the Quarto site:

```bash
export QUARTO_PYTHON="$PWD/.venv/bin/python"
quarto render docs
```
