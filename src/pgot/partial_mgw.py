"""Partial mixture Gromov-Wasserstein alignment and barycentric maps."""

import time

import numpy as np
import sklearn.mixture as sklmi
from lib import gromov
from sklearn.neighbors import NearestNeighbors

from .gaussian import (
    GaussianMixture,
    T_map_Gaussian,
    gaussian_cost_matrix,
    gaussian_w2_squared,
    gmm_transform,
    grad,
    proj_stiefel,
)
from .reproducibility import DEFAULT_SEED


def _fit_gmm(points, n_components, random_state=DEFAULT_SEED):
    mixture = sklmi.GaussianMixture(
        n_components=n_components,
        random_state=random_state,
    )
    mixture.fit(points)
    return GaussianMixture(
        mixture.weights_,
        mixture.means_,
        mixture.covariances_,
    )


def gw_mean_distortion(C1, C2, coupling):
    """Mean square-loss distortion of a (possibly partial) GW coupling.

    ``sum_ijkl (C1_ik - C2_jl)^2 G_ij G_kl / (sum G)^2``, evaluated in
    ``O(n^2)`` with the standard square-loss identity. This is the natural
    scale of the distortion term in (2.6) and (5.4): a coupling is worth
    keeping only when ``2 lambda`` exceeds it, so it is what makes the
    penalty comparable between the point-level and component-level
    representations.
    """
    C1 = np.asarray(C1, dtype=float)
    C2 = np.asarray(C2, dtype=float)
    G = np.asarray(coupling, dtype=float)
    mass = G.sum()
    if mass <= 0.0:
        raise ValueError("Cannot take the mean distortion of an empty coupling.")
    row, col = G.sum(axis=1), G.sum(axis=0)
    total = (
        row @ (C1**2) @ row
        - 2.0 * np.sum((C1 @ G) * (G @ C2))
        + col @ (C2**2) @ col
    )
    return float(total / mass**2)


def barycentric_projection(coupling, Y, tol=0.0):
    """Row-barycentric projection of a coupling onto the target points.

    Returns ``(mapped, mask)`` where ``mask`` marks the source rows that
    carry mass above ``tol``. Rows below the tolerance are unmatched: the
    barycentre is undefined there, so they must be dropped rather than
    assigned. For a balanced coupling every row is kept.
    """
    G = np.asarray(coupling, dtype=float)
    Y = np.asarray(Y, dtype=float)
    row_mass = G.sum(axis=1)
    mask = row_mass > tol
    mapped = (G[mask] @ Y) / row_mass[mask, None]
    return mapped, mask


def nearest_target_indices(mapped, Y):
    """Index of the nearest target point for each mapped point."""
    nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(Y)
    return nbrs.kneighbors(np.asarray(mapped, dtype=float),
                           return_distance=False).ravel()


def validate_partial_coupling(gamma, a, b, tol=1e-8):
    """Validate a non-negative, sub-marginal partial coupling."""
    gamma = np.asarray(gamma, dtype=float)
    if np.any(gamma < -tol):
        raise ValueError("Partial coupling Gamma has negative entries.")
    if np.any(gamma.sum(axis=1) > np.asarray(a) + tol):
        raise ValueError(
            "Partial coupling violates the source marginal a (Gamma 1 <= a)."
        )
    if np.any(gamma.sum(axis=0) > np.asarray(b) + tol):
        raise ValueError(
            "Partial coupling violates the target marginal b (Gamma^T 1 <= b)."
        )
    matched_mass = float(gamma.sum())
    if not (0.0 < matched_mass <= 1.0 + tol):
        raise ValueError(
            f"Z_lambda = {matched_mass} is out of the expected (0, 1] range."
        )
    return matched_mass


def matched_statistics(gamma, mu, nu, mass_tol=1e-14):
    """Return matched mass, marginals, and matched source/target means."""
    gamma = np.asarray(gamma, dtype=float)
    matched_mass = float(gamma.sum())
    if not np.isfinite(matched_mass) or matched_mass <= mass_tol:
        raise ValueError(
            "The partial coupling has zero matched mass (Z_lambda <= 0); "
            "barycentric map is undefined."
        )
    row_mass = gamma.sum(axis=1)
    col_mass = gamma.sum(axis=0)
    source_mean = (row_mass[:, None] * mu.comp_mean).sum(axis=0) / matched_mass
    target_mean = (col_mass[:, None] * nu.comp_mean).sum(axis=0) / matched_mass
    return matched_mass, row_mass, col_mass, source_mean, target_mean


def alignment_loss(P, gamma, mu_c, nu_c):
    """Evaluate the matched, centered Gaussian alignment objective."""
    value = 0.0
    for k in range(mu_c.K):
        for l in range(nu_c.K):
            if gamma[k, l] == 0.0:
                continue
            target_mean = P @ nu_c.comp_mean[l]
            target_cov = P @ nu_c.comp_cov[l] @ P.T
            value += gamma[k, l] * gaussian_w2_squared(
                mu_c.comp_mean[k],
                target_mean,
                mu_c.comp_cov[k],
                target_cov,
            )
    return float(value)


def alignment_grad(P, gamma, mu_c, nu_c):
    """Return the gradient of :func:`alignment_loss`."""
    return grad(P, gamma, mu_c, nu_c)


def partial_projected_gradient_descent(
    P0,
    gamma,
    mu_c,
    nu_c,
    step_size=1.0,
    max_iter=150,
    tol=1e-8,
    max_backtrack=20,
):
    """Optimize partial-MGW alignment on the Stiefel manifold.

    Each step is backtracked until :func:`alignment_loss` decreases, so the
    returned matrix is never worse than ``P0``. Without that guard a fixed
    ``step_size=1.0`` can drive the loss up by orders of magnitude and still
    stop on the ``||P_next - P||`` criterion.
    """
    P = P0.copy()
    current = alignment_loss(P, gamma, mu_c, nu_c)
    losses = [current]
    for _ in range(max_iter):
        gradient = alignment_grad(P, gamma, mu_c, nu_c)
        step = step_size
        candidate = None
        for _ in range(max_backtrack):
            trial = proj_stiefel(P - step * gradient)
            value = alignment_loss(trial, gamma, mu_c, nu_c)
            if value <= current:
                candidate = (trial, value)
                break
            step *= 0.5
        if candidate is None:
            break
        P_next, current = candidate
        losses.append(current)
        converged = np.linalg.norm(P_next - P) <= tol * (1.0 + np.linalg.norm(P))
        P = P_next
        if converged:
            break
    return P, losses


def partial_mgw_barycentric_map(
    X,
    mu,
    nu,
    P,
    gamma,
    m0_match,
    m1_match,
    density_tol=None,
):
    """Evaluate the matched-density partial-MGW barycentric map.

    The map is normalized by the matched source density
    ``sum[k,l] gamma[k,l] p_mu_k(x)`` rather than by the full mixture
    density. A zero matched density makes the map undefined and raises
    ``FloatingPointError``.
    """
    X = np.asarray(X, dtype=float)
    single = X.ndim == 1
    X2 = X[None, :] if single else X

    mu_P_centered = gmm_transform(mu, P=P.T, b=-P.T @ m0_match)
    nu_centered = gmm_transform(nu, b=-m1_match)

    component_pdf = np.vstack([mu.comp[k].pdf(X2) for k in range(mu.K)])
    row_mass = gamma.sum(axis=1)
    denominator = (row_mass[:, None] * component_pdf).sum(axis=0)

    if density_tol is None:
        density_tol = np.finfo(float).tiny
    if np.any(denominator <= density_tol):
        raise FloatingPointError(
            "Matched source density underflowed to zero at one or more query "
            "points; the barycentric map is not defined there."
        )

    X_P = np.einsum("ij,bj->bi", P.T, X2 - m0_match)
    numerator = np.zeros((X2.shape[0], nu.dim), dtype=float)
    for k in range(mu.K):
        for l in range(nu.K):
            if gamma[k, l] == 0.0:
                continue
            local_map = T_map_Gaussian(
                X_P,
                mu_P_centered.comp[k],
                nu_centered.comp[l],
            )
            numerator += gamma[k, l] * component_pdf[k][:, None] * local_map

    mapped = numerator / denominator[:, None] + m1_match
    return mapped[0] if single else mapped


def pMGW2_coup(
    X,
    Y,
    Lambda,
    n_components_X,
    n_components_Y,
    solver_tol=1e-12,
    step_size=1.0,
    max_alignment_iter=150,
    normalize_costs=True,
    points=True,
    return_both=False,
    verbose=False,
    random_state=DEFAULT_SEED,
    mu=None,
    nu=None,
    C1=None,
    C2=None,
    coupling=None,
    init_coupling=None,
    log=False,
):
    """Compute a partial-MGW coupling and barycentric map.

    ``Lambda`` is interpreted on the normalized cost scale when
    ``normalize_costs`` is true. The solver coupling is used unchanged for
    matched centering, alignment, and the barycentric map.

    Pass ``mu``/``nu`` to reuse mixtures fitted elsewhere, and ``C1``/``C2``
    to reuse their pairwise Gaussian W2 matrices, so that a balanced and a
    partial run can be compared on exactly the same fit and cost.

    ``init_coupling`` is the Frank-Wolfe starting point handed to the solver.
    The problem is non-convex, and the solver's default start is the
    independent coupling ``outer(a, b)``, which spreads every source component
    over every target component. When a component of ``nu`` has no counterpart
    in ``mu``, that start can leave the solver at the trivial empty coupling
    even where a matching with strictly better objective exists, so callers
    that already hold the balanced coupling should pass it here.
    """
    if mu is None:
        mu = _fit_gmm(X, n_components_X, random_state)
    if nu is None:
        nu = _fit_gmm(Y, n_components_Y, random_state)

    if C1 is None:
        C1 = gaussian_cost_matrix(mu)
    if C2 is None:
        C2 = gaussian_cost_matrix(nu)
    maximum_cost = max(float(C1.max()), float(C2.max()))
    if normalize_costs and maximum_cost > 0.0:
        C1_solver, C2_solver = C1 / maximum_cost, C2 / maximum_cost
    else:
        C1_solver, C2_solver = np.asarray(C1, float), np.asarray(C2, float)

    a = mu.weights
    b = nu.weights
    if verbose:
        print(f"Deriving partial MGW coupling with Lambda={Lambda}")

    start_time = time.time()
    if coupling is None:
        gamma = gromov.partial_gromov_ver1(
            C1_solver,
            C2_solver,
            a,
            b,
            Lambda=Lambda,
            nb_dummies=1,
            G0=None if init_coupling is None else np.asarray(init_coupling, float),
            thres=1,
            numItermax=None,
            numItermax_gw=1000,
            tol=solver_tol,
            log=False,
            verbose=verbose,
            line_search=True,
        )
        solver = "PGW_Metric partial_gromov_ver1"
    else:
        gamma = np.asarray(coupling, dtype=float)
        solver = "reused coupling"
    if np.isnan(gamma).any():
        raise FloatingPointError("Partial GW solver returned NaN entries in Gamma.")

    matched_mass = validate_partial_coupling(gamma, a, b)
    if verbose:
        print(f"Calculation time: {time.time() - start_time:.6f} sec")
        print(f"Total matched mass Z_lambda = {matched_mass:.6f}")

    _, _, _, m0, m1 = matched_statistics(gamma, mu, nu)
    mu_c = gmm_transform(mu, b=-m0)
    nu_c = gmm_transform(nu, b=-m1)
    cross = sum(
        gamma[k, l] * np.outer(mu_c.comp_mean[k], nu_c.comp_mean[l])
        for k in range(mu.K)
        for l in range(nu.K)
    )
    P0 = proj_stiefel(cross)
    P, losses = partial_projected_gradient_descent(
        P0,
        gamma,
        mu_c,
        nu_c,
        step_size=step_size,
        max_iter=max_alignment_iter,
    )

    mapped = partial_mgw_barycentric_map(X, mu, nu, P, gamma, m0, m1)
    if return_both or points:
        nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(Y)
        idx = nbrs.kneighbors(mapped, return_distance=False).ravel()
        if return_both:
            result = (idx, mapped)
        else:
            result = idx
    else:
        result = mapped
    if log:
        return result, {
            "solver": solver,
            "coupling": gamma,
            "matched_mass": matched_mass,
            "alignment": P,
            "alignment_losses": losses,
            "matched_source_mean": m0,
            "matched_target_mean": m1,
            "cost_normalization_scale": maximum_cost if normalize_costs else 1.0,
            "solver_lambda": float(Lambda),
            "runtime_seconds": time.time() - start_time,
        }
    return result
