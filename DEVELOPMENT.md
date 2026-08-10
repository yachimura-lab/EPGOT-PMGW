# Development guide

This guide contains the repository maintenance and development workflow. For
the paper overview, figure correspondence, and the shortest path to
reproducing the results, see the [README](README.md).

## Repository layout

```text
.
├── src/pgot/                  # Algorithms and canonical figure pipelines
├── docs/notebook/             # Executed notebooks, manifest, sidecars
├── tests/                     # Mathematical/reproducibility regressions
├── tools/build_notebooks.py   # Rebuild concise notebook source cells
├── tools/execute_notebooks.py # Execute all notebooks from clean kernels
├── pyproject.toml             # Exact direct dependency versions
└── uv.lock                    # Complete resolved environment
```

## Development environment

Requirements are Python 3.13.15, `uv`, and the checked-out PGW_Metric
submodule.

```bash
git submodule update --init --recursive
uv sync --frozen
```

`.python-version`, exact direct requirements in `pyproject.toml`, and
`uv.lock` fix the environment. New stochastic APIs should accept `rng=`, and
new GMM fits should accept `random_state=`. Keep notebook sampling on explicit
local NumPy generators; Figures 6–7 use separate generators for source, target,
and noise.

## Mathematical conventions

Figures 1–5 use `pgot.entropic_partial_ot`, the extended dummy-point problem
(3.1). The real-real cost is `M - 2*Lambda`, the extended marginals are
`(a, 1)` and `(b, 1)`, and entropy is applied to the full extended coupling.
Therefore `Lambda` is exactly the paper's `lambda`, without rescaling.

No figure in this repository uses `pgot.legacy`. That module holds
`partial_wasserstein_lagrange_entropic`, which relaxes the mass with
`M - Lambda` and regularizes only the real block, so it solves a different
problem from (3.1); it is kept solely to reproduce results published with it
and is not exported from the package root.

For Figures 6–7, both partial solvers receive the paper's `lambda=0.01`
unchanged, on cost matrices already divided by their own pair maximum. PGW
and pMGW both transport `300/350` mass, leaving the 50 noise points
unmatched.

Partial GW is non-convex, so the Frank-Wolfe starting point is part of the
specification. Both partial solves start from the balanced coupling of the
same representation. From the solver's own default start — the independent
coupling — the `6×7` component problem stalls at the empty coupling at
`lambda=0.01`, even though the coupling reached from the balanced start
scores a strictly better objective.

`partial_mgw_barycentric_map` carries the source points and components into
the target dimension by `P.T` and then maps between the projected source
Gaussian and the target Gaussian. This implementation coincides with
(6.8)–(6.9) in the square case `d = d'`, which is the setting used in the
paper experiments (`d = d' = 3` in Section 7.2). The general case `d > d'`,
where (6.8) builds the component map in the source dimension and applies
`P.T` last, is not implemented.

## Notebook workflow

Rebuild the notebook source cells and then execute all notebooks from clean
kernels:

```bash
MPLCONFIGDIR=/tmp/matplotlib uv run python tools/build_notebooks.py
MPLCONFIGDIR=/tmp/matplotlib NUMBA_CACHE_DIR=/tmp/numba \
  uv run python tools/execute_notebooks.py
```

## Tests

Run the mathematical and reproducibility regressions with:

```bash
MPLCONFIGDIR=/tmp/matplotlib NUMBA_CACHE_DIR=/tmp/numba \
  uv run python -m unittest discover -s tests -v
```

## Documentation site

Build the Quarto site with:

```bash
export QUARTO_PYTHON="$PWD/.venv/bin/python"
quarto render docs
```
