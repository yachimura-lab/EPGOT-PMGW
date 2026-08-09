"""Partial optimal transport helpers used by the notebook experiments."""

import warnings
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import ot
import scipy.stats as sps
from matplotlib.colors import LogNorm
from scipy.linalg import det, inv, sqrtm
from scipy.optimize import minimize
from sklearn.mixture import GaussianMixture as skGaussianMixture

from .gaussian import GaussianMixture
from .reproducibility import DEFAULT_SEED


def to_numpy(x):
    """Convert a list, array, or tensor-like value to NumPy."""
    try:
        return np.asarray(x)
    except Exception:
        return x


def entropic_partial_ot(
    a,
    b,
    M,
    Lambda=0.0,
    reg=0.01,
    numItermax=1000,
    stopThr=1e-9,
    remove_dummy=True,
    method="sinkhorn",
    log=False,
):
    """Solve the entropic partial OT problem (3.1) of the paper.

    This is the canonical EPOT primitive for the whole package. It builds the
    extended cost ``c_ij - 2*Lambda`` on the real-real block, appends the
    dummy row and column with extended marginals ``(a, 1)`` and ``(b, 1)``,
    and runs Sinkhorn so that the entropy applies to the *whole* extended
    coupling. ``Lambda`` is therefore the paper's penalty ``lambda``
    directly, on the scale of the normalized cost matrix (7.1).

    Returns the real-real block ``omega^{eps,lambda}`` unless
    ``remove_dummy`` is false, in which case the full extended coupling is
    returned.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    M = np.asarray(M, dtype=float)

    M = M - 2.0 * Lambda
    n, m = M.shape
    a_ext = np.append(a, 1.0)
    b_ext = np.append(b, 1.0)
    M_ext = np.zeros((n + 1, m + 1))
    M_ext[:n, :m] = M

    started = perf_counter()
    gamma_ext, sinkhorn_log = ot.sinkhorn(
        a_ext,
        b_ext,
        M_ext,
        reg=reg,
        method=method,
        numItermax=numItermax,
        stopThr=stopThr,
        verbose=False,
        log=True,
        # POT's warning does not expose the residual or let callers recover.
        # We inspect the marginals below and refine the same convex problem
        # when Sinkhorn stalls, so emitting the generic warning here would be
        # both noisy and less informative than the returned diagnostics.
        warn=False,
    )

    row_residual = float(np.max(np.abs(gamma_ext.sum(axis=1) - a_ext)))
    col_residual = float(np.max(np.abs(gamma_ext.sum(axis=0) - b_ext)))
    residual_tolerance = max(10.0 * stopThr, 1e-9)
    fallback_used = max(row_residual, col_residual) > residual_tolerance
    fallback_iterations = 0

    if fallback_used:
        gamma_ext, fallback_iterations = _refine_entropic_transport(
            a_ext,
            b_ext,
            M_ext,
            reg,
            gamma_ext,
            tol=stopThr,
        )
        row_residual = float(np.max(np.abs(gamma_ext.sum(axis=1) - a_ext)))
        col_residual = float(np.max(np.abs(gamma_ext.sum(axis=0) - b_ext)))

    result = gamma_ext[:-1, :-1] if remove_dummy else gamma_ext
    if not log:
        return result

    iterations = sinkhorn_log.get("niter", sinkhorn_log.get("n_iter"))
    return result, {
        "solver": (
            f"POT {method} + SciPy SLSQP refinement"
            if fallback_used
            else f"POT {method}"
        ),
        "iterations": None if iterations is None else int(iterations),
        "fallback_used": fallback_used,
        "fallback_iterations": int(fallback_iterations),
        "row_residual": row_residual,
        "column_residual": col_residual,
        "residual_tolerance": residual_tolerance,
        "converged": max(row_residual, col_residual) <= residual_tolerance,
        "runtime_seconds": perf_counter() - started,
        "matched_mass": float(gamma_ext[:-1, :-1].sum()),
    }


def _refine_entropic_transport(a, b, cost, reg, initial, tol):
    """Refine a small balanced entropic OT plan in the primal.

    The dummy-point formulation can become nearly degenerate when a real
    cost is close to ``2 * lambda``. Matrix scaling then converges extremely
    slowly even for a 3-by-4 problem. SLSQP solves the *same* strictly convex
    entropic objective under exact marginal constraints and is used only
    after the Sinkhorn residual fails the declared tolerance.
    """
    n, m = cost.shape
    if n * m > 10_000:
        raise RuntimeError(
            "Sinkhorn did not reach the marginal tolerance and the exact "
            "refinement is intentionally limited to 10,000 variables."
        )

    constraints = []
    for i in range(n):
        row = np.zeros((n, m))
        row[i, :] = 1.0
        constraints.append(row.ravel())
    # The last column equality is implied by all row equalities and the
    # preceding columns. Dropping it keeps the constraint matrix full rank.
    for j in range(m - 1):
        column = np.zeros((n, m))
        column[:, j] = 1.0
        constraints.append(column.ravel())
    A_eq = np.asarray(constraints)
    b_eq = np.concatenate([a, b[:-1]])
    lower = 1e-15

    def objective(x):
        return float(x @ cost.ravel() + reg * np.sum(x * np.log(x) - x))

    def gradient(x):
        return cost.ravel() + reg * np.log(x)

    equality = {
        "type": "eq",
        "fun": lambda x: A_eq @ x - b_eq,
        "jac": lambda x: A_eq,
    }
    optimized = minimize(
        objective,
        np.maximum(np.asarray(initial, dtype=float).ravel(), lower),
        jac=gradient,
        method="SLSQP",
        bounds=[(lower, None)] * (n * m),
        constraints=equality,
        options={"ftol": min(max(tol * 0.1, 1e-14), 1e-12), "maxiter": 10_000},
    )
    if not optimized.success:
        raise RuntimeError(
            "Entropic OT refinement failed after Sinkhorn stalled: "
            f"{optimized.message}"
        )
    return optimized.x.reshape(n, m), optimized.nit


def densite_theorique2d(mu, Sigma, alpha, x):
    """Compute a 2D GMM density (adapted from judelo/gmmot)."""
    K = mu.shape[0]
    alpha = alpha.reshape(1, K)
    y = 0
    for j in range(K):
        y += alpha[0, j] * sps.multivariate_normal.pdf(
            x,
            mean=mu[j, :],
            cov=Sigma[j, :, :],
        )
    return y


def display_gmm(gmm, n=200, ax=0, bx=1, ay=0, by=1, cmap="gnuplot", axis=None):
    """Display density contours for a 2D Gaussian mixture."""
    if axis is None:
        axis = plt.gca()

    _, pi, mu, covariance = gmm
    x = np.linspace(ax, bx, n)
    y = np.linspace(ay, by, n)
    X, Y = np.meshgrid(x, y)
    points = np.column_stack([X.ravel(), Y.ravel()])
    Z = densite_theorique2d(mu, covariance, pi, points).reshape(X.shape)
    # LogNorm masks exact underflow zeros and otherwise emits a warning.
    # Clipping at the smallest positive float is visually identical because
    # the contour levels start three decades below the density maximum.
    Z = np.maximum(Z, np.finfo(float).tiny)

    Zmax = Z.max()
    levels = np.logspace(np.log10(Zmax * 1e-3), np.log10(Zmax), 8)
    norm = LogNorm(vmin=Zmax * 1e-3, vmax=Zmax)
    axis.contourf(X, Y, Z, levels=levels, cmap=cmap, norm=norm)
    axis.set_aspect("equal")


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

    Use :func:`entropic_partial_ot` instead; it implements (3.1) exactly.
    Kept only so that previously published results remain reproducible.
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


def normalize_cost(C):
    """Normalize a nonzero cost matrix by its maximum value."""
    maximum = np.max(C)
    if maximum > 0:
        C = C / maximum
    return C


def _gaussian_w2_cost(mean0, cov0, mean1, cov1, clip=False):
    mean0 = np.asarray(mean0)
    mean1 = np.asarray(mean1)
    cov0 = np.asarray(cov0)
    cov1 = np.asarray(cov1)
    sqrt_cov1 = sqrtm(cov1).real
    cross_term = sqrtm(sqrt_cov1 @ cov0 @ sqrt_cov1).real
    value = (
        np.sum((mean0 - mean1) ** 2)
        + np.trace(cov0)
        + np.trace(cov1)
        - 2 * np.trace(cross_term)
    )
    return max(0, value) if clip else value


def _fit_figure5_mixtures(
    X,
    Y,
    n_components_X,
    n_components_Y,
    random_state=DEFAULT_SEED,
):
    mixX = skGaussianMixture(
        n_components=n_components_X,
        covariance_type="full",
        random_state=random_state,
    ).fit(X)
    mixY = skGaussianMixture(
        n_components=n_components_Y,
        covariance_type="full",
        random_state=random_state,
    ).fit(Y)
    return mixX, mixY


def _cross_gaussian_cost(mixX, mixY, clip=False):
    cost = np.zeros((len(mixX.weights_), len(mixY.weights_)))
    for k in range(cost.shape[0]):
        for l in range(cost.shape[1]):
            cost[k, l] = _gaussian_w2_cost(
                mixX.means_[k],
                mixX.covariances_[k],
                mixY.means_[l],
                mixY.covariances_[l],
                clip=clip,
            )
    return cost


def _cross_gaussian_cost_gmm(mu, nu, clip=True):
    """Cross-mixture squared Gaussian W2 cost for :class:`GaussianMixture`."""
    cost = np.zeros((mu.K, nu.K))
    for k in range(mu.K):
        for l in range(nu.K):
            cost[k, l] = _gaussian_w2_cost(
                mu.comp_mean[k],
                mu.comp_cov[k],
                nu.comp_mean[l],
                nu.comp_cov[l],
                clip=clip,
            )
    return cost


def gaussian_displacement_interpolation(mean0, cov0, mean1, cov1, t):
    """Return the McCann displacement interpolation (4.17) in closed form.

    ``((1 - t) Id + t T)_# N(mean0, cov0)`` where ``T`` is the Gaussian
    optimal map (4.5). Writing ``B = (1 - t) Id + t A`` with ``A`` the map
    matrix, the result is the Gaussian with mean ``(1-t) mean0 + t mean1``
    and covariance ``B cov0 B^T``. No iteration is needed.
    """
    mean0 = np.asarray(mean0, dtype=float)
    mean1 = np.asarray(mean1, dtype=float)
    cov0 = np.asarray(cov0, dtype=float)
    A = compute_monge_map_matrix(cov0, np.asarray(cov1, dtype=float))
    B = (1.0 - t) * np.eye(cov0.shape[0]) + t * A
    return (1.0 - t) * mean0 + t * mean1, B @ cov0 @ B.T


def entropic_partial_displacement_interpolation(
    mu,
    nu,
    t,
    epsilon=1e-2,
    Lambda=None,
    coupling=None,
    log=False,
):
    """Return the entropic partial displacement interpolation (4.18).

    The result is the **sub-probability** measure
    ``mu^t_{eps,lambda} = sum_kl omega_kl mu^t_kl`` whose total mass is
    ``Z_lambda <= 1``. Weights are the coupling entries themselves: they are
    neither thresholded nor renormalized, so the mass carries through to the
    density. Each ``mu^t_kl`` comes from the closed form (4.17).

    Returns ``(weights, means, covs)``, with ``weights.sum() == Z_lambda``.

    ``Lambda`` is required unless ``coupling`` is supplied, in which case the
    coupling already fixes the penalty and ``Lambda`` is not read.
    """
    if coupling is None:
        gamma, solve_info = entropic_partial_coupling(
            mu, nu, epsilon=epsilon, Lambda=Lambda, log=True
        )
    else:
        gamma = np.asarray(coupling, dtype=float)
        solve_info = {
            "solver": "reused coupling",
            "matched_mass": float(gamma.sum()),
        }

    weights, means, covs = [], [], []
    for k in range(mu.K):
        for l in range(nu.K):
            mean, cov = gaussian_displacement_interpolation(
                mu.comp_mean[k], mu.comp_cov[k], nu.comp_mean[l], nu.comp_cov[l], t
            )
            weights.append(gamma[k, l])
            means.append(mean)
            covs.append(cov)

    weights = np.asarray(weights, dtype=float)
    result = (weights, np.asarray(means), np.asarray(covs))
    if log:
        return result, {**solve_info, "coupling": gamma}
    return result


def entropic_partial_coupling(mu, nu, *, Lambda, epsilon=1e-2, log=False):
    """Solve the paper's component-level EPOT problem once.

    This separates cost construction and the solve from interpolation or
    barycentric plotting. Callers that render several values of ``t`` can
    pass the returned coupling to
    :func:`entropic_partial_displacement_interpolation` instead of silently
    solving the identical problem once per panel.

    ``Lambda`` is the paper's ``lambda`` on the normalized cost scale, and is
    keyword-only and required: this function always solves, and there is no
    penalty that stands in for the balanced problem.
    """
    raw_cost = _cross_gaussian_cost_gmm(mu, nu)
    cost_scale = float(np.max(raw_cost))
    if cost_scale <= 0.0:
        raise ValueError("The cross-mixture Gaussian cost has zero scale.")
    normalized_cost = raw_cost / cost_scale
    gamma, info = _epot_component_coupling(
        mu.weights,
        nu.weights,
        normalized_cost,
        Lambda,
        epsilon,
        log=True,
    )
    if not log:
        return gamma
    return gamma, {
        **info,
        "coupling": gamma,
        "cost_matrix": normalized_cost,
        "cost_normalization_scale": cost_scale,
        "paper_lambda": float(Lambda),
        "solver_lambda": float(Lambda),
        "epsilon": float(epsilon),
    }


def submixture_pdf(X, weights, means, covs):
    """Evaluate a Gaussian mixture density without renormalizing the weights.

    Unlike a probability mixture, ``weights`` may sum to less than one; the
    returned density then integrates to that same mass. This is what (4.18)
    requires.
    """
    X = np.asarray(X, dtype=float)
    total = np.zeros(X.shape[0], dtype=float)
    for w, mean, cov in zip(weights, means, covs):
        if w == 0.0:
            continue
        total += w * sps.multivariate_normal.pdf(X, mean=mean, cov=cov)
    return total


def entropic_partial_barycentric_map(
    X,
    mu,
    nu,
    epsilon=1e-2,
    Lambda=None,
    coupling=None,
    log=False,
):
    """Evaluate the entropic partial barycentric projection map (6.2).

    ``mu`` and ``nu`` are :class:`~pgot.gaussian.GaussianMixture` objects
    holding *known* component parameters, so nothing is fitted here. The
    component coupling is the real-real block of (3.1) applied to the
    weights with the normalized Gaussian cost (7.1), and the map is
    normalized by the matched density ``sum_kl omega_kl p_k(x)``.

    ``Lambda`` is the paper's ``lambda`` on the normalized cost scale. It is
    required unless ``coupling`` is supplied, in which case the coupling
    already fixes the penalty and ``Lambda`` is not read.
    """
    X = np.asarray(X, dtype=float)
    d = X.shape[1]

    if coupling is None:
        gamma, solve_info = entropic_partial_coupling(
            mu, nu, epsilon=epsilon, Lambda=Lambda, log=True
        )
    else:
        gamma = np.asarray(coupling, dtype=float)
        solve_info = {
            "solver": "reused coupling",
            "matched_mass": float(gamma.sum()),
        }

    A = {
        (k, l): compute_monge_map_matrix(mu.comp_cov[k], nu.comp_cov[l])
        for k in range(mu.K)
        for l in range(nu.K)
    }
    component_pdf = np.vstack([mu.comp[k].pdf(X) for k in range(mu.K)])

    row_mass = gamma.sum(axis=1)
    denominator = (row_mass[:, None] * component_pdf).sum(axis=0)
    numerator = np.zeros((X.shape[0], d), dtype=float)
    for k in range(mu.K):
        for l in range(nu.K):
            Tkl = nu.comp_mean[l] + np.einsum(
                "ij,bj->bi", A[(k, l)], X - mu.comp_mean[k]
            )
            numerator += gamma[k, l] * component_pdf[k][:, None] * Tkl

    Z = numerator / (denominator[:, None] + 1e-300)
    if log:
        return Z, {**solve_info, "coupling": gamma}
    return Z


def _epot_component_coupling(
    a,
    b,
    M,
    Lambda,
    epsilon,
    nb_dummies=1,
    log=False,
):
    """Solve (3.1) between mixture components on a normalized cost matrix.

    ``Lambda`` is the paper's ``lambda`` and must be given. No finite value
    reproduces balanced entropic OT: for ``eps > 0`` every entry of the
    extended kernel is strictly positive, so the dummy node keeps some mass
    at any penalty, and the balanced problem is only the limit
    ``Lambda -> infinity``.
    """
    if nb_dummies != 1:
        raise ValueError(
            "The paper's (3.1) uses exactly one dummy point; "
            f"nb_dummies={nb_dummies} is not supported."
        )
    if Lambda is None:
        raise ValueError(
            "Lambda must be specified. Balanced entropic OT is obtained only "
            "in the limit Lambda -> infinity, not at any finite penalty; for "
            "a balanced entropic coupling use ot.sinkhorn(a, b, M, reg=eps)."
        )
    return entropic_partial_ot(
        a,
        b,
        M,
        Lambda=Lambda,
        reg=epsilon,
        log=log,
    )


def compute_monge_map_matrix(CovK, CovL):
    """Return the optimal Gaussian transport-map matrix from K to L."""
    sqrtK = sqrtm(CovK).real
    inv_sqrtK = inv(sqrtK)
    core = sqrtm(sqrtK @ CovL @ sqrtK).real
    return inv_sqrtK @ core @ inv_sqrtK


def compute_T_X_to_Z(
    X,
    Y,
    n_components_X,
    n_components_Y,
    *,
    Lambda,
    epsilon=1e-2,
    nb_dummies=1,
    random_state=DEFAULT_SEED,
    log=False,
):
    """Compute the partial-OT barycentric projection map (6.2).

    ``Lambda`` is keyword-only and required; this entry point always solves.
    It is the paper's ``lambda`` in (3.1)/(4.4), on the scale of the
    normalized cost matrix (7.1).

    This compatibility entry point now uses the matched density
    ``sum_kl omega_kl p_k(x)`` in the denominator, exactly as (6.2)
    requires. It is equivalent to :func:`compute_T_X_to_Z_C`.
    """
    if nb_dummies != 1:
        raise ValueError(
            "The paper's (3.1) uses exactly one dummy point; "
            f"nb_dummies={nb_dummies} is not supported."
        )
    mixX, mixY = _fit_figure5_mixtures(
        X,
        Y,
        n_components_X,
        n_components_Y,
        random_state=random_state,
    )
    mu = GaussianMixture(mixX.weights_, mixX.means_, mixX.covariances_)
    nu = GaussianMixture(mixY.weights_, mixY.means_, mixY.covariances_)
    return entropic_partial_barycentric_map(
        X,
        mu,
        nu,
        epsilon=epsilon,
        Lambda=Lambda,
        log=log,
    )


def compute_T_X_to_Z_C(
    X,
    Y,
    n_components_X,
    n_components_Y,
    *,
    Lambda,
    epsilon=1e-2,
    nb_dummies=1,
    random_state=DEFAULT_SEED,
    log=False,
):
    """Compute Figure 5's entropic partial barycentric projection map (6.2).

    The component coupling is the real-real block of the paper's EPOT
    problem (3.1)/(4.4), obtained from :func:`entropic_partial_ot`. The map
    is normalized by the *matched* density ``sum_kl omega_kl p_k(x)``, as
    (6.2) prescribes, not by the full mixture density.

    ``Lambda`` is keyword-only and required, as in :func:`compute_T_X_to_Z`,
    which this function is equivalent to; it is the paper's ``lambda`` on the
    scale of the normalized cost matrix (7.1), and is passed to the solver
    unchanged.

    This fits a mixture to each point cloud. Section 7.1.5 of the paper
    instead evaluates (6.2) for the *known* mixtures GA and GB, which is
    what :func:`entropic_partial_barycentric_map` does; prefer that when the
    component parameters are available.
    """
    if nb_dummies != 1:
        raise ValueError(
            "The paper's (3.1) uses exactly one dummy point; "
            f"nb_dummies={nb_dummies} is not supported."
        )
    mixX, mixY = _fit_figure5_mixtures(
        X,
        Y,
        n_components_X,
        n_components_Y,
        random_state=random_state,
    )
    mu = GaussianMixture(mixX.weights_, mixX.means_, mixX.covariances_)
    nu = GaussianMixture(mixY.weights_, mixY.means_, mixY.covariances_)
    return entropic_partial_barycentric_map(
        X, mu, nu, epsilon=epsilon, Lambda=Lambda, log=log
    )
