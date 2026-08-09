"""Superseded solvers, kept only so that older results stay reproducible.

Nothing here implements the paper. The package root deliberately does not
re-export this module, so ``from pgot import ...`` cannot reach these
functions by accident; reproducing an older result takes the explicit
``from pgot.legacy import ...``.
"""

import warnings

import numpy as np


def partial_wasserstein_lagrange_entropic(
    a,
    b,
    M,
    Lambda=None,
    epsilon=1e-2,
    nb_dummies=1,
    log=False,
    **kwargs,
):
    """Deprecated: solves a *different* problem from the paper's (3.1).

    This routine relaxes the mass with ``M - Lambda`` (a single ``Lambda``,
    not ``2*Lambda``) and applies the entropy only to the real block, using
    capped scaling against the inequality constraints ``gamma 1 <= a`` and
    ``gamma^T 1 <= b``. Its ``Lambda`` therefore corresponds to ``2*lambda``
    in the paper's convention, and even after that rescaling its minimizer
    differs from (3.1) because the dummy row and column carry no entropy.

    ``Lambda=None`` substitutes ``max(M) + 1``. That default is kept because
    removing it would change results published with it; it is not a balanced
    limit, and the paper path rejects the analogous default outright.

    Use :func:`pgot.entropic_partial_ot` instead; it implements (3.1) exactly.
    """
    warnings.warn(
        "partial_wasserstein_lagrange_entropic does not implement the "
        "paper's (3.1): it uses 'M - Lambda' instead of 'M - 2*lambda' and "
        "regularizes only the real block. Use entropic_partial_ot instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    del nb_dummies, kwargs
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    M = np.asarray(M, float)

    if Lambda is None:
        Lambda = float(np.max(M)) + 1.0

    cost = M - Lambda
    kernel = np.exp(-cost / epsilon)
    u = np.ones_like(a)
    v = np.ones_like(b)

    for _ in range(2000):
        u_prev = u.copy()
        Kv = kernel @ v
        u = np.minimum(a / (Kv + 1e-300), 1.0)
        KTu = kernel.T @ u
        v = np.minimum(b / (KTu + 1e-300), 1.0)
        if np.linalg.norm(u - u_prev) < 1e-14:
            break

    gamma = np.diag(u) @ kernel @ np.diag(v)

    if log:
        return gamma, {
            "transported_mass": np.sum(gamma),
            "cost": np.sum(gamma * M),
        }
    return gamma
