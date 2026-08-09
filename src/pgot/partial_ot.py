"""Partial optimal transport helpers used by the notebook experiments."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import ot
import scipy.stats as sps
from matplotlib.colors import LogNorm
from scipy.linalg import det, inv, sqrtm
from sklearn.mixture import GaussianMixture as skGaussianMixture

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

    gamma_ext = ot.sinkhorn(
        a_ext,
        b_ext,
        M_ext,
        reg=reg,
        method="sinkhorn",
        numItermax=numItermax,
        stopThr=stopThr,
        verbose=False,
        log=False,
        warn=True,
    )
    if remove_dummy:
        return gamma_ext[:-1, :-1]
    return gamma_ext


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


def _figure5_component_coupling(a, b, M, Lambda, epsilon, nb_dummies):
    """Solve (3.1) between mixture components for the Figure 5 helpers.

    ``Lambda`` is the paper's ``lambda``. ``None`` selects the balanced
    limit: with ``M`` normalized to a maximum of one, ``Lambda = max(M)``
    makes every real-real entry of the extended cost negative, so no mass is
    left on the dummy node.
    """
    if nb_dummies != 1:
        raise ValueError(
            "The paper's (3.1) uses exactly one dummy point; "
            f"nb_dummies={nb_dummies} is not supported."
        )
    if Lambda is None:
        Lambda = float(np.max(M))
    return entropic_partial_ot(a, b, M, Lambda=Lambda, reg=epsilon)


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
    epsilon=1e-2,
    Lambda=None,
    nb_dummies=1,
    random_state=DEFAULT_SEED,
    log=False,
):
    """Compute a partial-OT barycentric projection with the full-mixture denominator.

    ``Lambda`` is the paper's ``lambda`` in (3.1)/(4.4), on the scale of the
    normalized cost matrix (7.1).

    Note that the denominator here is the *full* mixture density
    ``sum_k a_k p_k(x)``, which is the balanced form (6.1) rather than the
    partial form (6.2). Prefer :func:`compute_T_X_to_Z_C` for Figure 5.
    """
    mixX, mixY = _fit_figure5_mixtures(
        X,
        Y,
        n_components_X,
        n_components_Y,
        random_state=random_state,
    )
    a = mixX.weights_
    b = mixY.weights_
    Kx = len(a)
    Ky = len(b)

    M = _cross_gaussian_cost(mixX, mixY)
    M = M / np.max(M)
    gamma = _figure5_component_coupling(a, b, M, Lambda, epsilon, nb_dummies)

    Z = np.zeros_like(X)
    d = X.shape[1]
    for i, x in enumerate(X):
        p = np.array(
            [
                np.exp(
                    -0.5
                    * (
                        (x - mixX.means_[k])
                        @ np.linalg.inv(mixX.covariances_[k])
                        @ (x - mixX.means_[k])
                    )
                )
                / np.sqrt((2 * np.pi) ** d * np.linalg.det(mixX.covariances_[k]))
                for k in range(Kx)
            ]
        )
        denom = np.sum(a * p) + 1e-300
        pi_x = (a * p) / denom
        Tb = np.zeros_like(x)
        for k in range(Kx):
            if a[k] < 1e-15:
                continue
            for l in range(Ky):
                map_matrix = compute_monge_map_matrix(
                    mixX.covariances_[k],
                    mixY.covariances_[l],
                )
                Tkl = mixY.means_[l] + map_matrix @ (x - mixX.means_[k])
                Tb += (gamma[k, l] / a[k]) * pi_x[k] * Tkl
        Z[i] = Tb
    if log:
        return Z, {"matched_mass": float(gamma.sum()), "coupling": gamma}
    return Z


def compute_T_X_to_Z_C(
    X,
    Y,
    n_components_X,
    n_components_Y,
    epsilon=1e-2,
    Lambda=1e-1,
    nb_dummies=1,
    random_state=DEFAULT_SEED,
    log=False,
):
    """Compute Figure 5's entropic partial barycentric projection map (6.2).

    The component coupling is the real-real block of the paper's EPOT
    problem (3.1)/(4.4), obtained from :func:`entropic_partial_ot`. The map
    is normalized by the *matched* density ``sum_kl omega_kl p_k(x)``, as
    (6.2) prescribes, not by the full mixture density.

    ``Lambda`` is the paper's ``lambda``, on the scale of the normalized
    cost matrix (7.1); it is passed to the solver unchanged.
    """
    mixX, mixY = _fit_figure5_mixtures(
        X,
        Y,
        n_components_X,
        n_components_Y,
        random_state=random_state,
    )
    a = mixX.weights_
    b = mixY.weights_
    Kx = len(a)
    Ky = len(b)
    d = X.shape[1]

    M = _cross_gaussian_cost(mixX, mixY, clip=True)
    M = M / np.max(M)
    gamma = _figure5_component_coupling(a, b, M, Lambda, epsilon, nb_dummies)

    # Every component pair contributes to (6.2). Dropping small-weight pairs
    # from the numerator while keeping them in the denominator would bias the
    # map towards zero, so no threshold is applied here.
    A_matrices = {
        (k, l): compute_monge_map_matrix(
            mixX.covariances_[k],
            mixY.covariances_[l],
        )
        for k in range(Kx)
        for l in range(Ky)
    }

    Z = np.zeros_like(X)
    for i, x in enumerate(X):
        p_vals = np.zeros(Kx)
        for k in range(Kx):
            diff = x - mixX.means_[k]
            inv_cov = inv(mixX.covariances_[k])
            exponent = -0.5 * diff @ inv_cov @ diff
            norm_const = np.sqrt((2 * np.pi) ** d * det(mixX.covariances_[k]))
            p_vals[k] = np.exp(exponent) / norm_const

        numerator = np.zeros(d)
        denominator = 0.0
        for k in range(Kx):
            for l in range(Ky):
                denominator += gamma[k, l] * p_vals[k]
                Tkl = mixY.means_[l] + A_matrices[(k, l)] @ (x - mixX.means_[k])
                numerator += gamma[k, l] * p_vals[k] * Tkl
        Z[i] = numerator / (denominator + 1e-300)
    if log:
        return Z, {"matched_mass": float(gamma.sum()), "coupling": gamma}
    return Z
