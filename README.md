# Entropic Partial Optimal Transport and Partial Gromov–Wasserstein Distance between Gaussian Mixtures

This repository accompanies the EPGOT-pMGW paper. It provides implementations
of entropic partial optimal transport between Gaussian mixtures and partial
mixture Gromov–Wasserstein matching, together with executed notebooks that
reproduce Figures 1–7. Numerical pipelines live in `src/pgot/`; executed
notebooks and figure provenance live under `docs/notebook/`.

[Rendered notebooks and publication site](https://epgot-pmgw.netlify.app/)

## Authors

- Toshiaki Yachimura, Mathematical Science Center for Co-creative Society,
  Tohoku University, Sendai 980-0845, Japan.
  Email: [toshiaki.yachimura.a4@tohoku.ac.jp](mailto:toshiaki.yachimura.a4@tohoku.ac.jp)

- Xiaocheng Zou, Mathematical Institute, Tohoku University, Sendai 980-8578,
  Japan. Email:
  [zou.xiaocheng.t3@dc.tohoku.ac.jp](mailto:zou.xiaocheng.t3@dc.tohoku.ac.jp)

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

## Limitations

`partial_mgw_barycentric_map` carries the source points and components into
the target dimension by `P.T` and then maps between the projected source
Gaussian and the target Gaussian. This implementation coincides with
(6.8)–(6.9) in the square case `d = d'`, which is the setting used in the
paper experiments (`d = d' = 3` in Section 7.2). The general case `d > d'`,
where (6.8) builds the component map in the source dimension and applies
`P.T` last, is not implemented.

## License

Except for third-party material in the `vendor/PGW_Metric` submodule, this
repository is licensed under the [Creative Commons Attribution 4.0
International License](LICENSE.txt) (`CC-BY-4.0`). The submodule remains
subject to its upstream terms.

Mathematical conventions, repository structure, notebook regeneration,
testing, and documentation-site maintenance are described in the
[development guide](DEVELOPMENT.md).
