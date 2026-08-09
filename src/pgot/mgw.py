"""Mixture Gromov-Wasserstein distances and couplings."""

from time import perf_counter

import numpy as np
import ot
import sklearn.mixture as sklmi
from sklearn.neighbors import NearestNeighbors

from .gaussian import (
    GaussianMixture,
    GaussianW2,
    T_mean,
    T_rand,
    gmm_transform,
    proj_gradient_descent,
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


def MGW2_GM(mu, nu):
    """Compute MGW2 between two Gaussian mixtures."""
    return ot.gromov.gromov_wasserstein2(
        mu.pdist(GaussianW2),
        nu.pdist(GaussianW2),
        mu.weights,
        nu.weights,
        loss_fun="square_loss",
    )


def aMGW2_GM(mu, nu, n_step=10, reg_init=1, beta=0.95, verbose=False):
    """Compute MGW2 with an annealed entropic initialization."""
    del verbose
    reg = reg_init
    weights = np.outer(mu.weights, nu.weights)
    for _ in range(n_step):
        weights = ot.gromov.entropic_gromov_wasserstein(
            mu.pdist(GaussianW2),
            nu.pdist(GaussianW2),
            mu.weights,
            nu.weights,
            loss_fun="square_loss",
            epsilon=reg,
            G0=weights,
        )
        reg *= beta
    return ot.gromov.gromov_wasserstein2(
        mu.pdist(GaussianW2),
        nu.pdist(GaussianW2),
        mu.weights,
        nu.weights,
        loss_fun="square_loss",
        G0=weights,
    )


def MGW2(X, Y, n_components=20, annealing=False, random_state=DEFAULT_SEED):
    """Compute MGW2 between two point clouds."""
    mu = _fit_gmm(X, n_components, random_state)
    nu = _fit_gmm(Y, n_components, random_state)
    if annealing:
        return aMGW2_GM(mu, nu)
    return MGW2_GM(mu, nu)


def MGW2_GM_coup(mu, nu, C1=None, C2=None):
    """Return an MGW coupling between Gaussian-mixture components.

    ``C1``/``C2`` override the pairwise Gaussian W2 matrices, so a balanced
    and a partial run can share exactly the same cost.
    """
    return ot.gromov.gromov_wasserstein(
        mu.pdist(GaussianW2) if C1 is None else C1,
        nu.pdist(GaussianW2) if C2 is None else C2,
        mu.weights,
        nu.weights,
        loss_fun="square_loss",
    )


def aMGW2_GM_coup(mu, nu, n_step=10, reg_init=1, beta=0.95, verbose=False):
    """Return an MGW coupling with an annealed entropic initialization."""
    del verbose
    reg = reg_init
    weights = np.outer(mu.weights, nu.weights)
    for _ in range(n_step):
        weights = ot.gromov.entropic_gromov_wasserstein(
            mu.pdist(GaussianW2),
            nu.pdist(GaussianW2),
            mu.weights,
            nu.weights,
            loss_fun="square_loss",
            epsilon=reg,
            G0=weights,
        )
        reg *= beta
    return ot.gromov.gromov_wasserstein(
        mu.pdist(GaussianW2),
        nu.pdist(GaussianW2),
        mu.weights,
        nu.weights,
        loss_fun="square_loss",
        G0=weights,
    )


def MGW2_coup(
    X,
    Y,
    n_components=20,
    annealing=False,
    method="T_rand",
    points=True,
    return_both=False,
    verbose=False,
    random_state=DEFAULT_SEED,
    rng=None,
    mu=None,
    nu=None,
    C1=None,
    C2=None,
    step_size=1.0,
    coupling=None,
    log=False,
):
    """Return an MGW map, or nearest target-point indices, for two point clouds.

    Pass ``mu``/``nu`` and ``C1``/``C2`` to reuse a fit and cost computed
    elsewhere, so that a balanced and a partial run can be compared on
    identical inputs.
    """
    if mu is None:
        if verbose:
            print("fitting mixture 1")
        mu = _fit_gmm(X, n_components, random_state)
    if nu is None:
        if verbose:
            print("fitting mixture 2")
        nu = _fit_gmm(Y, n_components, random_state)

    started = perf_counter()
    if coupling is None:
        if verbose:
            print("deriving coupling between GMMs")
        if annealing:
            weights = aMGW2_GM_coup(mu, nu)
            solver = "POT annealed entropic GW initialization + GW"
        else:
            weights = MGW2_GM_coup(mu, nu, C1=C1, C2=C2)
            solver = "POT gromov_wasserstein"
    else:
        weights = np.asarray(coupling, dtype=float)
        solver = "reused coupling"

    if verbose:
        print("deriving map from coupling")
    # (6.7) is stated for components centered at the matched means (6.6),
    # which for a balanced coupling are the full mixture means. Optimizing on
    # the uncentered mixtures solves a different problem.
    mu_c = gmm_transform(mu, b=-mu.mean())
    nu_c = gmm_transform(nu, b=-nu.mean())
    P = proj_stiefel(
        sum(
            weights[k, l] * np.outer(mu_c.comp_mean[k], nu_c.comp_mean[l])
            for k in range(mu.K)
            for l in range(nu.K)
        )
    )
    P, losses = proj_gradient_descent(P, weights, mu_c, nu_c, step_size)

    if method == "T_mean":
        Z = T_mean(X, mu, nu, P, weights)
    elif method == "T_rand":
        Z = T_rand(X, mu, nu, P, weights, rng=rng)
    else:
        raise ValueError("method must be 'T_mean' or 'T_rand'")

    if return_both or points:
        nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(Y)
        idx = nbrs.kneighbors(Z, return_distance=False).ravel()
        if return_both:
            result = (idx, Z)
        else:
            result = idx
    else:
        result = Z
    if log:
        return result, {
            "solver": solver,
            "coupling": weights,
            "matched_mass": float(weights.sum()),
            "alignment": P,
            "alignment_losses": losses,
            "runtime_seconds": perf_counter() - started,
        }
    return result
