"""Embedded Wasserstein and mixture embedded Wasserstein algorithms."""

import itertools

import numpy as np
import ot
import scipy.linalg as spl
import sklearn.mixture as sklmi
from sklearn.neighbors import NearestNeighbors

from .gaussian import (
    GaussianMixture,
    I,
    T_mean,
    T_rand,
    gmm_transform,
    proj_gradient_descent,
    proj_stiefel,
)
from .reproducibility import DEFAULT_SEED, get_rng


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


def MEW2_GM(
    mu,
    nu,
    alpha,
    n_iter_P=150,
    eps=1e-3,
    n_iter_max=1000,
    init="Gaussian",
    initw=None,
    init_phase=True,
    symmetry=True,
    verbose=False,
    rng=None,
):
    """Compute mixture embedded W2 between two Gaussian mixtures."""

    def twistedGaussianW2(mean_0, mean_1, Cov_0, Cov_0_full, Cov_1):
        sqCov_1 = spl.sqrtm(Cov_1).real
        return (
            spl.norm(mean_0 - mean_1) ** 2
            + np.trace(Cov_0_full)
            + np.trace(Cov_1)
            - 2 * np.trace(spl.sqrtm(sqCov_1 @ Cov_0 @ sqCov_1).real)
        )

    def compute_weights_loss(P, mu, nu, loss_only=False, w=None):
        mu_P = gmm_transform(mu, P=P.T)
        M = np.array(
            [
                [
                    twistedGaussianW2(
                        mu_P.comp_mean[k],
                        nu.comp_mean[l],
                        mu_P.comp_cov[k],
                        mu.comp_cov[k],
                        nu.comp_cov[l],
                    )
                    for l in range(nu.K)
                ]
                for k in range(mu.K)
            ]
        )
        if not loss_only:
            weights = ot.emd(mu.weights, nu.weights, M)
        else:
            weights = w
        loss = np.sum(weights * M)
        if loss_only:
            return loss
        return weights, loss

    def initialize(P, mu, nu, symmetry=False):
        P_list = P @ [
            np.diag(a) for a in itertools.product([1, -1], repeat=P.shape[1])
        ]
        weights_list, loss_list = [], []
        for candidate in P_list:
            weights, loss = compute_weights_loss(candidate, mu, nu)
            weights_list.append(weights)
            loss_list.append(loss)
        idx = np.argmin(loss_list)
        P = P_list[idx]
        weights = weights_list[idx]
        loss = loss_list[idx]
        if symmetry:
            P_list = [P[:, a] for a in itertools.permutations(range(P.shape[1]))]
            weights_list, loss_list = [], []
            for candidate in P_list:
                candidate_weights, candidate_loss = compute_weights_loss(
                    candidate,
                    mu,
                    nu,
                )
                weights_list.append(candidate_weights)
                loss_list.append(candidate_loss)
            idx = np.argmin(loss_list)
            P = P_list[idx]
            weights = weights_list[idx]
            loss = loss_list[idx]
        return P, weights, loss

    if isinstance(init, str):
        _, P0 = spl.eigh(mu.cov())
        _, P1 = spl.eigh(nu.cov())
        P0 = P0[:, ::-1]
        P1 = P1[:, ::-1]

    mu = gmm_transform(mu, b=-mu.mean())
    nu = gmm_transform(nu, b=-nu.mean())

    if isinstance(init, str):
        if init == "Gaussian":
            P = P0 @ I(mu.dim, nu.dim) @ P1.T
        elif init == "random":
            P = proj_stiefel(get_rng(rng).random((mu.dim, nu.dim)))
        else:
            raise ValueError("init must be 'Gaussian', 'random', or a matrix")
    else:
        P = init

    if initw is None:
        if init_phase:
            P, weights, loss = initialize(P, mu, nu, symmetry)
        else:
            weights, loss = compute_weights_loss(P, mu, nu)
    else:
        weights = initw
        loss = compute_weights_loss(P, mu, nu, True, weights)

    loss_old = 0
    n_iter = 0
    while n_iter < n_iter_max and np.abs(loss - loss_old) > eps:
        if verbose:
            print("iteration " + str(n_iter) + ": loss = " + str(loss))
        loss_old = loss
        P, _ = proj_gradient_descent(P, weights, mu, nu, alpha, n_iter=n_iter_P)
        weights, loss = compute_weights_loss(P, mu, nu)
        n_iter += 1

    return P, weights, loss


def EW2(X, Y, a=None, b=None, eps=1e-3, n_iter_max=10000, verbose=False, rng=None):
    """Compute embedded W2 between two point clouds."""
    m = X.shape[1]
    n = Y.shape[1]
    if a is None:
        a = ot.unif(X.shape[0])
    if b is None:
        b = ot.unif(Y.shape[0])
    P = proj_stiefel(get_rng(rng).random((m, n)))
    M = ot.dist(X, np.einsum("mn,bn->bm", P, Y))
    weights = ot.emd(a, b, M, numItermax=1000000)
    loss = np.trace(weights.T @ M)
    loss_old = 0
    n_iter = 0
    while n_iter < n_iter_max and np.abs(loss - loss_old) > eps:
        if verbose:
            print("iteration " + str(n_iter) + ": loss = " + str(loss))
        loss_old = loss
        P = proj_stiefel(X.T @ weights @ Y)
        M = ot.dist(X, np.einsum("mn,bn->bm", P, Y))
        weights = ot.emd(a, b, M, numItermax=1000000)
        loss = np.trace(weights.T @ M)
        n_iter += 1
    return P, weights, loss


def aEW2(
    X,
    Y,
    a=None,
    b=None,
    eps=1e-3,
    n_iter_max=10000,
    reg_init=1,
    beta=0.95,
    verbose=False,
    rng=None,
):
    """Compute embedded W2 with an annealed entropic scheme."""
    m = X.shape[1]
    n = Y.shape[1]
    P = proj_stiefel(get_rng(rng).random((m, n)))
    M = ot.dist(X, np.einsum("mn,bn->bm", P, Y))
    if a is None:
        a = ot.unif(X.shape[0])
    if b is None:
        b = ot.unif(Y.shape[0])
    reg = reg_init
    weights = ot.bregman.sinkhorn_stabilized(a, b, M, reg)
    loss = np.trace(weights.T @ M)
    loss_old = 0
    n_iter = 0
    while n_iter < n_iter_max and np.abs(loss - loss_old) > eps:
        if verbose:
            print("iteration " + str(n_iter) + ": loss = " + str(loss))
        loss_old = loss
        reg *= beta
        P = proj_stiefel(X.T @ weights @ Y)
        M = ot.dist(X, np.einsum("mn,bn->bm", P, Y))
        a = ot.unif(X.shape[0])
        b = ot.unif(Y.shape[0])
        weights = ot.bregman.sinkhorn_stabilized(a, b, M, reg)
        loss = np.trace(weights.T @ M)
        n_iter += 1
    return P, weights, loss


def MEW2(
    X,
    Y,
    n_components=20,
    annealing=True,
    n_iter_annealing=10,
    beta=0.99,
    random_state=DEFAULT_SEED,
    rng=None,
):
    """Compute MEW2 between two point clouds."""
    mu = _fit_gmm(X, n_components, random_state)
    nu = _fit_gmm(Y, n_components, random_state)
    if annealing:
        P, _, _ = aEW2(
            mu.comp_mean,
            nu.comp_mean,
            n_iter_max=n_iter_annealing,
            beta=0.99,
            verbose=False,
            rng=rng,
        )
        _, _, loss = MEW2_GM(
            mu,
            nu,
            alpha=0.01,
            eps=1e-3,
            n_iter_P=150,
            init=P,
            init_phase=False,
            symmetry=False,
            verbose=False,
            rng=rng,
        )
    else:
        _, _, loss = MEW2_GM(
            mu,
            nu,
            alpha=0.01,
            eps=1e-3,
            n_iter_P=150,
            init_phase=True,
            symmetry=True,
            verbose=False,
            rng=rng,
        )
    return loss


def MEW2_coup(
    X,
    Y,
    n_components=20,
    annealing=True,
    n_iter_annealing=10,
    beta=0.99,
    method="T_rand",
    points=True,
    return_both=False,
    random_state=DEFAULT_SEED,
    rng=None,
):
    """Return an MEW map, or nearest target-point indices, for point clouds."""
    mu = _fit_gmm(X, n_components, random_state)
    nu = _fit_gmm(Y, n_components, random_state)
    if annealing:
        P, _, _ = aEW2(
            mu.comp_mean,
            nu.comp_mean,
            n_iter_max=n_iter_annealing,
            beta=0.99,
            verbose=False,
            rng=rng,
        )
        P, weights, _ = MEW2_GM(
            mu,
            nu,
            alpha=0.01,
            eps=1e-3,
            n_iter_P=150,
            init=P,
            init_phase=False,
            symmetry=False,
            verbose=False,
            rng=rng,
        )
    else:
        P, weights, _ = MEW2_GM(
            mu,
            nu,
            alpha=0.01,
            eps=1e-3,
            n_iter_P=150,
            init_phase=True,
            symmetry=True,
            verbose=False,
            rng=rng,
        )

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
            return idx, Z
        return idx
    return Z
