# EPGOT-pMGW — Entropic Partial Optimal Transport and Partial Gromov--Wasserstein Distance between Gaussian Mixtures

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
| 6 | [figure6_7.ipynb](docs/notebook/figure6_7.ipynb) | Four barycentric-projection-based nearest assignments |
| 7 | [figure6_7.ipynb](docs/notebook/figure6_7.ipynb) | Raw MGW/pMGW maps from the same solves as Figure 6 |

The machine-readable mapping is
[`figure_manifest.json`](docs/notebook/figure_manifest.json). Each figure has a
unique PNG and JSON sidecar; the sidecar stores source and vendored-solver
revisions, versions, seeds, fitted-input and coupling hashes, paper and solver
parameters, matched mass, residuals, runtime, solve ID, and output SHA-256.

## Reproducibility

`.python-version`, exact direct requirements in `pyproject.toml`, and
`uv.lock` fix the environment. Notebook sampling uses explicit local NumPy
generators; Figures 6–7 use separate generators for source, target, and 50 noise
points.

Requirements are Python 3.13.15, `uv`, and the checked-out PGW_Metric
submodule. To reproduce the executed notebooks from clean kernels:

```bash
git submodule update --init --recursive
uv sync --frozen
MPLCONFIGDIR=/tmp/matplotlib NUMBA_CACHE_DIR=/tmp/numba \
  uv run python tools/execute_notebooks.py
```

Mathematical conventions, repository structure, notebook regeneration,
testing, and documentation-site maintenance are described in the
[development guide](DEVELOPMENT.md).
