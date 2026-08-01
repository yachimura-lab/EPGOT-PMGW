# Vendored research code

`PGW_Metric` is checked out from
<https://github.com/mint-vu/PGW_Metric.git> at commit
`8a9002e14d6b17decf35e603e9a9e42aa8465e04`.

The upstream repository is about 832 MB because it contains experiment results
and images. This checkout therefore uses Git sparse checkout and contains only:

- `lib/`
- `README.md`
- `environment.yml`

The local `pgw-metric-vendored` packaging wrapper makes `lib/` available in
the uv environment. Its core solver dependencies are managed by the root
`pyproject.toml` and `uv.lock`. Some upstream experiment notebooks may require
additional, experiment-specific packages. Use the core library from the
repository root with:

```bash
uv run python -c "from lib import gromov"
```

To reproduce the sparse checkout without downloading the upstream result
artifacts:

```bash
git clone --filter=blob:none --no-checkout https://github.com/mint-vu/PGW_Metric.git vendor/PGW_Metric
git -C vendor/PGW_Metric sparse-checkout init --no-cone
git -C vendor/PGW_Metric sparse-checkout set /lib/ /README.md /environment.yml
git -C vendor/PGW_Metric checkout --detach 8a9002e14d6b17decf35e603e9a9e42aa8465e04
```
