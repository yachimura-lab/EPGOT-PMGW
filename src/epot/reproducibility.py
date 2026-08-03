"""Deterministic random numbers and figure metadata.

The figures in this repository are produced by pipelines that sample point
clouds and fit Gaussian mixtures, both of which are stochastic. Without a
fixed seed every run of a notebook draws different data and sklearn's
``GaussianMixture`` converges to a different local optimum, so the saved
figures change from one run to the next.

Two things are pinned here:

* **Random draws.** Every stochastic step in :mod:`epot` takes its numbers
  from :func:`get_rng`, which defaults to a package-level generator seeded
  with :data:`DEFAULT_SEED` at import time. Calls therefore stay
  statistically independent within a session while the session as a whole
  replays identically. :func:`set_seed` re-seeds that generator together
  with the process-global generators the notebooks use directly.
* **Figure metadata.** Matplotlib stamps the current time into EPS, PDF,
  and SVG files, so saving the same figure twice yields different bytes.
  :func:`freeze_output_timestamps` replaces that stamp with a fixed date.

Notebooks normally just call :func:`set_reproducible` once, before any
sampling.
"""

import os
import random

import numpy as np

#: Seed used by the package generator and by the Gaussian-mixture fits.
DEFAULT_SEED = 42

#: Fixed timestamp written into saved EPS/PDF/SVG files (2000-01-01 UTC).
DEFAULT_SOURCE_DATE_EPOCH = 946684800

_rng = np.random.default_rng(DEFAULT_SEED)


def get_rng(rng=None):
    """Return a :class:`numpy.random.Generator` for ``rng``.

    ``None`` selects the package-level generator, which is seeded with
    :data:`DEFAULT_SEED` and re-seeded by :func:`set_seed`. Anything else is
    passed to :func:`numpy.random.default_rng`, so a seed, a ``SeedSequence``
    or an existing generator all work.
    """
    if rng is None:
        return _rng
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def set_seed(seed=DEFAULT_SEED):
    """Seed every generator that can influence the figures.

    This covers the package generator behind :func:`get_rng`, the legacy
    ``numpy.random`` global state used by the notebooks, the standard-library
    :mod:`random` module, and PyTorch when it is installed.

    Returns the package generator so callers can keep using it directly.
    """
    global _rng

    seed = int(seed)
    _rng = np.random.default_rng(seed)
    np.random.seed(seed)
    random.seed(seed)

    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)

    return _rng


def freeze_output_timestamps(source_date_epoch=DEFAULT_SOURCE_DATE_EPOCH):
    """Pin the creation date Matplotlib writes into EPS/PDF/SVG output.

    Matplotlib reads ``SOURCE_DATE_EPOCH`` when saving a figure (see
    https://reproducible-builds.org/specs/source-date-epoch/); setting it
    makes repeated saves of the same figure byte-identical instead of
    differing only in their timestamp.
    """
    os.environ["SOURCE_DATE_EPOCH"] = str(int(source_date_epoch))


def set_reproducible(
    seed=DEFAULT_SEED,
    source_date_epoch=DEFAULT_SOURCE_DATE_EPOCH,
):
    """Seed the generators and pin figure timestamps in one call.

    Call this at the top of a notebook, before any sampling, so that
    re-running it reproduces the previous figures exactly.
    """
    rng = set_seed(seed)
    freeze_output_timestamps(source_date_epoch)
    return rng
