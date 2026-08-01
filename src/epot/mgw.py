"""Mixture Gromov-Wasserstein distances and couplings."""

import numpy as np
import ot
import sklearn.mixture as sklmi
from sklearn.neighbors import NearestNeighbors

from .gaussian import (
    GaussianMixture,
    GaussianW2,
    T_mean,
    T_rand,
    proj_gradient_descent,
    proj_stiefel,
)


def _fit_gmm(points, n_components):
    mixture = sklmi.GaussianMixture(n_components=n_components)
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


def MGW2(X, Y, n_components=20, annealing=False):
    """Compute MGW2 between two point clouds."""
    mu = _fit_gmm(X, n_components)
    nu = _fit_gmm(Y, n_components)
    if annealing:
        return aMGW2_GM(mu, nu)
    return MGW2_GM(mu, nu)


def MGW2_GM_coup(mu, nu):
    """Return an MGW coupling between Gaussian-mixture components."""
    return ot.gromov.gromov_wasserstein(
        mu.pdist(GaussianW2),
        nu.pdist(GaussianW2),
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
):
    """Return an MGW map, or nearest target-point indices, for two point clouds."""
    if verbose:
        print("fitting mixture 1")
    mu = _fit_gmm(X, n_components)
    if verbose:
        print("fitting mixture 2")
    nu = _fit_gmm(Y, n_components)

    if verbose:
        print("deriving coupling between GMMs")
    if annealing:
        weights = aMGW2_GM_coup(mu, nu)
    else:
        weights = MGW2_GM_coup(mu, nu)

    if verbose:
        print("deriving map from coupling")
    P = proj_stiefel(
        sum(
            weights[k, l] * np.outer(mu.comp_mean[k], nu.comp_mean[l])
            for k in range(mu.K)
            for l in range(nu.K)
        )
    )
    P, _ = proj_gradient_descent(P, weights, mu, nu, 1)

    if method == "T_mean":
        Z = T_mean(X, mu, nu, P, weights)
    elif method == "T_rand":
        Z = T_rand(X, mu, nu, P, weights)
    else:
        raise ValueError("method must be 'T_mean' or 'T_rand'")

    if return_both or points:
        nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(Y)
        idx = nbrs.kneighbors(Z, return_distance=False).ravel()
        if return_both:
            return idx, Z
        return idx
    return Z
