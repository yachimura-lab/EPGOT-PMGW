# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code for **entropic partial Gromov optimal transport (EPGOT) between Gaussian mixture models**. The deliverable is the set of paper figures: the `pgot` Python package holds the algorithms, and notebooks under `docs/notebook/` reproduce Figures 1–5 and are published as a Quarto site on Netlify (https://epgot-pmgw.netlify.app/).

Note the naming split: the repo/site is `EPGOT-pMGW`, the Python package is `pgot` (renamed from `epot` in b68c774 — old names may still appear in stale text).

## Setup

Two things must be in place before anything imports, in this order:

```bash
# 1. Vendored solver (see below) — sparse checkout, NOT `git submodule update --init`
git clone --filter=blob:none --no-checkout https://github.com/mint-vu/PGW_Metric.git vendor/PGW_Metric
git -C vendor/PGW_Metric sparse-checkout init --no-cone
git -C vendor/PGW_Metric sparse-checkout set /lib/ /README.md /environment.yml
git -C vendor/PGW_Metric checkout --detach 8a9002e14d6b17decf35e603e9a9e42aa8465e04

# 2. Python env (installs the root project editable, which puts `src/` on sys.path)
uv sync
```

`vendor/PGW_Metric` is a submodule of an ~832 MB upstream repo; the sparse checkout above fetches only `lib/`. A plain `git submodule update --init` works but downloads everything.

Smoke test that both landed:

```bash
uv run python -c "import pgot; from lib import gromov"
```

## Common commands

```bash
uv run jupyter lab docs/notebook          # edit/run the figure notebooks

export QUARTO_PYTHON=$(pwd)/.venv/bin/python
quarto preview docs                       # local site preview
quarto render docs                        # what Netlify runs (see netlify.toml)
```

There is no test suite, linter config, or CI in this repo. Verification means re-running the affected notebook and checking that the figures still look right.

## Architecture

### `src/pgot/` — the library

`__init__.py` is a flat re-export façade over four topic modules; notebooks import everything from the package root (`from pgot import pMGW2_coup`). Keep new public functions exported there and listed in `__all__`.

- `gaussian.py` — foundation layer. `Gaussian`/`GaussianMixture` containers, closed-form Gaussian W2 (`gaussian_w2_squared`, aliased `GaussianW2` for compatibility), Stiefel-manifold projection and projected gradient descent, and the transport maps `T_map_Gaussian` / `T_mean` / `T_rand`. Everything else builds on this.
- `mgw.py`, `mew.py` — mixture Gromov-Wasserstein and mixture embedded W2, both full (non-partial) transport, via POT (`ot.gromov.*`).
- `partial_ot.py` — entropic *partial* OT between mixture components plus the Figure 5 barycentric-projection maps (`compute_T_X_to_Z`, `compute_T_X_to_Z_C`).
- `partial_mgw.py` — partial MGW: coupling from the vendored solver, matched-mass centering, Stiefel alignment, then `partial_mgw_barycentric_map`.

The recurring pipeline shape across `MGW2_coup`, `MEW2_coup`, and `pMGW2_coup` is: fit an sklearn `GaussianMixture` to each point cloud → build a Gaussian-W2 cost matrix between components → solve for a coupling → recover an orthogonal alignment `P` by projected gradient descent on the Stiefel manifold → push points through a barycentric/randomized map → optionally snap to nearest target points via `NearestNeighbors`. New variants should follow that sequence rather than reinventing it.

What makes the *partial* case different, and where bugs tend to hide: the coupling `Γ` is sub-marginal (`Γ1 ≤ a`, `Γᵀ1 ≤ b`), so the matched mass `Z_λ = ΣΓ` is `< 1`. Centering, alignment, and the barycentric denominator must all use *matched* quantities (`matched_statistics`, `validate_partial_coupling`) rather than the full mixture weights or full mixture density. `partial_mgw_barycentric_map` raises `FloatingPointError` instead of silently dividing by an underflowed density.

### `vendor/` — the partial-GW solver

`pgot.partial_mgw` does `from lib import gromov` and calls `gromov.partial_gromov_ver1`. That `lib` comes from the vendored PGW_Metric checkout, force-included into a wheel by `vendor/pyproject.toml`. Because `__init__.py` imports `partial_mgw`, **`import pgot` fails outright if the submodule is empty** — that is the usual cause of an import error in a fresh clone.

### Reproducibility contract

Every notebook's first cell is `from pgot import set_reproducible; set_reproducible()`, and it must stay first — figures change if any sampling happens before it. `reproducibility.py` seeds a package-level generator (`DEFAULT_SEED = 42`) that every stochastic function reaches through `get_rng(rng)`, plus the process-global `numpy.random` / `random` / `torch` generators the notebooks use directly, and pins `SOURCE_DATE_EPOCH` so saved EPS/PDF/SVG bytes don't differ only by timestamp. New stochastic code must draw from `get_rng(rng)` and accept an `rng=` override; new mixture fits must thread `random_state=DEFAULT_SEED`.

### `docs/` — the published site

Quarto renders `docs/notebook/*.ipynb` into the sidebar automatically (`_quarto.yml`). Notebooks that already carry outputs are published as-is rather than re-executed, so **committed outputs are what readers see** — after changing library code, re-run the affected notebooks and commit them with their outputs. `docs/index.qmd` is currently empty.

## Note on the README

The README's "Repository layout" section still shows the notebooks as `src/figure*.ipynb`; they actually live in `docs/notebook/` (the links in its Notebooks table are correct). It also omits the vendored-submodule setup step. Worth fixing if you touch the README.
