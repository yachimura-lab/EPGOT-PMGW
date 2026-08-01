import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as spl
import ot
from scipy.stats import multivariate_normal
import sklearn.mixture as sklmi
import itertools
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import inv, sqrtm
from scipy.linalg import det
import scipy.stats as sps
from matplotlib.colors import LogNorm
from sklearn.mixture import GaussianMixture as skGaussianMixture
import time
from lib import gromov


class Gaussian:
    ### class encoding a Gaussian distribution ###

    def __init__(self, mean, cov):
        self.mean = np.array(mean)
        self.cov = np.array(cov)
        self.dim = len(mean)

    def sample(self, n_samples):
        return np.random.multivariate_normal(self.mean, self.cov, n_samples)

    def pdf(self, x):
        return multivariate_normal.pdf(x, mean=self.mean, cov=self.cov)


class GaussianMixture:
    ### Encode a Gaussian distribution ###

    def __init__(self, weights_list, mean_list, cov_list):
        self.comp = [Gaussian(mean_list[k], cov_list[k]) for k in range(len(mean_list))]
        self.comp_mean = np.array(mean_list)
        self.comp_cov = np.array(cov_list)
        self.weights = np.array(weights_list)
        self.dim = len(mean_list[0])
        self.K = len(mean_list)

    def sample(self, n_samples, return_K=False):
        """
        if return_K, return the list of component indice for each sample
        """
        idx_list = np.random.choice(np.arange(self.K), n_samples, p=self.weights)
        samples_list = [
            np.random.multivariate_normal(self.comp_mean[i], self.comp_cov[i])
            for i in idx_list
        ]
        if return_K:
            return np.array(samples_list), idx_list
        else:
            return np.array(samples_list)

    def mean(self):
        ### return the mean of the Gaussian mixture
        return np.average(self.comp_mean, axis=0, weights=self.weights)

    def cov(self):
        ### return the covariance matrix of the Gaussian mixture
        A = sum(
            [
                self.weights[k]
                * (self.comp_cov[k] + np.outer(self.comp_mean[k], self.comp_mean[k]))
                for k in range(self.K)
            ]
        )
        mean = self.mean()
        return A - np.outer(mean, mean)

    def pdf(self, x):
        return sum(
            [
                self.weights[k]
                * multivariate_normal.pdf(
                    x, mean=self.comp_mean[k], cov=self.comp_cov[k]
                )
                for k in range(self.K)
            ]
        )

    def plot_scatter_2d(self, n_samples, T=None, *args):
        ### utils for Figures 3 and 4
        X, idx_list = self.sample(n_samples, True)
        grad_color = np.array(
            [
                (x / np.max(X[:, 0]), y / np.max(X[:, 1]), x / np.max(X[:, 0]), 1)
                for x, y in X
            ]
        )
        if T is not None:
            Y = np.array([T(X[k], idx_list[k], *args) for k in range(len(X))])
        else:
            Y = X
        cmap_list = [
            "spring",
            "summer",
            "autumn",
            "winter",
            "cool",
            "Wistia",
            "hot",
            "afmhot",
            "gist_heat",
            "copper",
            "viridis",
            "plasma",
            "inferno",
            "magma",
            "cividis",
        ]
        for k in range(self.K):
            Z = Y[idx_list == k]
            plt.scatter(Z[:, 0], Z[:, 1], color=X[idx_list == k][:, 0], s=10)
        plt.axis("equal")

    def pdist(self, f):
        ### compute pairwise distances matrix
        dist_list = [
            [
                f(
                    self.comp_mean[k],
                    self.comp_mean[l],
                    self.comp_cov[k],
                    self.comp_cov[l],
                )
                for l in range(k + 1, self.K)
            ]
            for k in range(self.K)
        ][:-1]
        dist_matrix = np.zeros((self.K, self.K))
        for k in range(self.K):
            for l in range(k + 1, self.K):
                dist_matrix[k, l] = dist_list[k][l - k - 1]
                dist_matrix[l, k] = dist_matrix[k, l]
        return dist_matrix


def gmm_transform(mu, P=None, b=None):
    ### Apply a linear transformation x -> Px + b to the Gaussian mixture
    if P is None:
        P = np.eye(mu.dim)
    if b is None:
        b = np.zeros(P.shape[0])
    mean_list = [P.dot(m) + b for m in mu.comp_mean]
    cov_list = [P @ A @ P.T for A in mu.comp_cov]
    return GaussianMixture(mu.weights, mean_list, cov_list)


def I(m, n):
    ### rectangular matrix (I_n  0)^T
    I = np.zeros((m, n))
    I[:n, :n] = np.eye(n)
    return I


def proj_stiefel(A):
    ### projection on Stiefel Manifold
    m, n = A.shape
    U, s, VT = spl.svd(A)
    return U @ I(m, n) @ VT


def GaussianW2(mean_0, mean_1, Cov_0, Cov_1):
    ### W_2 between Gaussian distributions
    sqCov_1 = spl.sqrtm(Cov_1).real
    return (
        spl.norm(mean_0 - mean_1) ** 2
        + np.trace(Cov_0)
        + np.trace(Cov_1)
        - 2 * np.trace(spl.sqrtm(sqCov_1 @ Cov_0 @ sqCov_1).real)
    )


def gaussian_cost_matrix(gmm):
    K = gmm.K
    C = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            C[i, j] = GaussianW2(
                gmm.comp_mean[i], gmm.comp_mean[j], gmm.comp_cov[i], gmm.comp_cov[j]
            )
    return C


def gaussian_W(gmm1, gmm2):
    K = gmm1.K
    L = gmm2.K
    C = np.zeros((K, L))
    for i in range(K):
        for j in range(L):
            C[i, j] = GaussianW2(
                gmm1.comp_mean[i], gmm2.comp_mean[j], gmm1.comp_cov[i], gmm2.comp_cov[j]
            )
    return C


def grad_gauss(P, Cov_0, Cov_1):
    ### grad of P -> tr(Cov_1^{\frac{1}{2}}P^TCov_0PCov_1^{\frac{1}{2})
    sqCov_1 = spl.sqrtm(Cov_1).real
    A = Cov_0 @ P @ sqCov_1
    B = spl.inv(spl.sqrtm(sqCov_1 @ P.T @ A)).real
    return A @ B @ sqCov_1


def grad(P, weights, mu, nu):
    ### grad of P -> \sum_{k,l} weights[k,l]W_2^2(\mu_k,P_{\#}\nu_l)
    return 2 * sum(
        [
            weights[k, l]
            * (
                -np.outer(mu.comp_mean[k], nu.comp_mean[l])
                + P @ np.outer(nu.comp_mean[l], nu.comp_mean[l])
                - grad_gauss(P, mu.comp_cov[k], nu.comp_cov[l])
            )
            for k in range(mu.K)
            for l in range(nu.K)
        ]
    )


def proj_gradient_descent(P, weights, mu, nu, alpha, n_iter=150):
    ### projected gradient descent on the Stiefel manifold
    loss = [
        sum(
            [
                weights[k, l]
                * GaussianW2(0, 0, mu.comp_cov[k], P @ nu.comp_cov[l] @ P.T)
                for k in range(mu.K)
                for l in range(nu.K)
            ]
        )
    ]
    for k in range(n_iter):
        P = proj_stiefel(P - alpha * grad(P, weights, mu, nu))
        loss.append(
            sum(
                [
                    weights[k, l]
                    * GaussianW2(0, 0, mu.comp_cov[k], P @ nu.comp_cov[l] @ P.T)
                    for k in range(mu.K)
                    for l in range(nu.K)
                ]
            )
        )
    return P, loss


def T_map_Gaussian(x, mu, nu):
    ### Transport map between two Gaussian distributions mu and nu
    invcov0 = spl.inv(spl.sqrtm(mu.cov).real)
    sqcov0 = spl.sqrtm(mu.cov).real
    A = invcov0 @ spl.sqrtm(sqcov0 @ nu.cov @ sqcov0) @ invcov0
    if len(x.shape) == 1:
        return nu.mean + A @ (x - mu.mean)
    else:
        return nu.mean + np.einsum("ij,bj -> bi", A, x - mu.mean)


def T_map_Gaussian_rand(x, mu, nu, idx_list):
    ### Transport map between two Gaussian distributions mu and nu (utils for T_rand)
    invcov0 = [spl.inv(spl.sqrtm(mu.comp[k].cov).real) for k in range(mu.K)]
    sqcov0 = [spl.sqrtm(mu.comp[k].cov).real for k in range(mu.K)]
    A_list = [
        [
            invcov0[k] @ spl.sqrtm(sqcov0[k] @ nu.comp[l].cov @ sqcov0[k]) @ invcov0[k]
            for l in range(nu.K)
        ]
        for k in range(mu.K)
    ]
    return np.array(
        [
            nu.comp[idx[1]].mean
            + A_list[idx[0]][idx[1]] @ (x[i] - mu.comp[idx[0]].mean)
            for i, idx in enumerate(idx_list)
        ]
    )


def T_mean(X, mu, nu, P, weights):
    mu_P_centered = gmm_transform(mu, P=P.T, b=-P.T @ mu.mean())
    nu_centered = gmm_transform(nu, b=-nu.mean())
    pdf_comp = [mu.comp[k].pdf(X) for k in range(mu.K)]
    ipdf_mu = 1.0 / mu.pdf(X)
    if len(X.shape) == 1:
        X_P = P.T @ (X - mu.mean())
        Y = (
            sum(
                [
                    weights[k, l]
                    * pdf_comp[k]
                    * T_map_Gaussian(X_P, mu_P_centered.comp[k], nu_centered.comp[l])
                    for k in range(mu.K)
                    for l in range(nu.K)
                ]
            )
            * ipdf_mu
        )
    else:
        X_P = np.einsum("ij,bj->bi", P.T, X - mu.mean())
        Y = np.einsum(
            "bi,b -> bi",
            sum(
                [
                    weights[k, l]
                    * np.einsum(
                        "bi,b -> bi",
                        T_map_Gaussian(X_P, mu_P_centered.comp[k], nu_centered.comp[l]),
                        pdf_comp[k],
                    )
                    for k in range(mu.K)
                    for l in range(nu.K)
                ]
            ),
            ipdf_mu,
        )
    return Y + nu.mean()


def T_rand(x, mu, nu, P, weights):
    mu_P_centered = gmm_transform(mu, P=P.T, b=-P.T @ mu.mean())
    nu_centered = gmm_transform(nu, b=-nu.mean())
    rng = np.random.default_rng()
    probs = np.array(
        [
            weights[k, l] * mu.comp[k].pdf(x) / mu.pdf(x)
            for k in range(mu.K)
            for l in range(nu.K)
        ]
    )
    idx_list = [
        rng.choice([[k, l] for k in range(mu.K) for l in range(nu.K)], p=prob)
        for prob in probs.T
    ]
    return T_map_Gaussian_rand(x, mu_P_centered, nu_centered, idx_list)


## Main algo


def MGW2_GM(mu, nu):
    ### core method between two Gaussian mixtures (returns a distance)
    return ot.gromov.gromov_wasserstein2(
        mu.pdist(GaussianW2),
        nu.pdist(GaussianW2),
        mu.weights,
        nu.weights,
        loss_fun="square_loss",
    )


def aMGW2_GM(mu, nu, n_step=10, reg_init=1, beta=0.95, verbose=False):
    ### core method with annealed scheme
    reg = reg_init
    weights = np.outer(mu.weights, nu.weights)
    for k in range(n_step):
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
    ### MGW2 between two clouds of points (returns a distance)
    mix = sklmi.GaussianMixture(n_components=n_components)
    mix.fit(X)
    mu = GaussianMixture(mix.weights_, mix.means_, mix.covariances_)
    del mix
    mix = sklmi.GaussianMixture(n_components=n_components)
    mix.fit(Y)
    nu = GaussianMixture(mix.weights_, mix.means_, mix.covariances_)
    del mix
    if annealing:
        return aMGW2_GM(mu, nu)
    else:
        return MGW2_GM(mu, nu)


def MGW2_GM_coup(mu, nu):
    ### core method between two Gaussian Mixtures (return a coupling between Gaussian components)
    return ot.gromov.gromov_wasserstein(
        mu.pdist(GaussianW2),
        nu.pdist(GaussianW2),
        mu.weights,
        nu.weights,
        loss_fun="square_loss",
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
    """
    MGW2 between two clouds of points (return a map between the points)
    If points = False, return the direct output of the T_mean or T_rand map (not in Y),
    if points = True, return the idxs of the points in Y where the points in X are transported at.
    """
    if verbose:
        print("fitting mixture 1")
    mix = sklmi.GaussianMixture(n_components=n_components)
    mix.fit(X)
    mu = GaussianMixture(mix.weights_, mix.means_, mix.covariances_)
    del mix
    if verbose:
        print("fitting mixture 2")
    mix = sklmi.GaussianMixture(n_components=n_components)
    mix.fit(Y)
    nu = GaussianMixture(mix.weights_, mix.means_, mix.covariances_)
    del mix
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
            [
                weights[k, l] * np.outer(mu.comp_mean[k], nu.comp_mean[l])
                for k in range(mu.K)
                for l in range(nu.K)
            ]
        )
    )
    P, loss = proj_gradient_descent(P, weights, mu, nu, 1)
    if method == "T_mean":
        Z = T_mean(X, mu, nu, P, weights)
    elif method == "T_rand":
        Z = T_rand(X, mu, nu, P, weights)
    if return_both:
        idx = nbrs.kneighbors(Z, return_distance=False).ravel()
        return idx, Z
    elif points:
        nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(Y)
        idx = nbrs.kneighbors(Z, return_distance=False).ravel()
        return idx
    else:
        return Z


def aMGW2_GM_coup(mu, nu, n_step=10, reg_init=1, beta=0.95, verbose=False):
    ### core method with annealed scheme
    reg = reg_init
    weights = np.outer(mu.weights, nu.weights)
    for k in range(n_step):
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
):
    ### core method between two Gaussian mixtures mu and nu

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
        del mu_P
        if not loss_only:
            weights = ot.emd(mu.weights, nu.weights, M)
        else:
            weights = w
        loss = np.sum(weights * M)
        if loss_only:
            return loss
        else:
            return weights, loss

    def initialize(P, mu, nu, symmetry=False):
        P_list = P @ [np.diag(a) for a in itertools.product([1, -1], repeat=P.shape[1])]
        weights_list, loss_list = [], []
        for k in range(len(P_list)):
            weights, loss = compute_weights_loss(P_list[k], mu, nu)
            weights_list.append(weights)
            loss_list.append(loss)
        idx = np.argmin(loss_list)
        P = P_list[idx]
        weights = weights_list[idx]
        loss = loss_list[idx]
        if symmetry:
            P_list = [P[:, a] for a in itertools.permutations(range(P.shape[1]))]
            weights_list, loss_list = [], []
            for k in range(len(P_list)):
                weights, loss = compute_weights_loss(P_list[k], mu, nu)
                weights_list.append(weights)
                loss_list.append(loss)
            idx = np.argmin(loss_list)
            P = P_list[idx]
            weights = weights_list[idx]
            loss = loss_list[idx]

        return P, weights, loss

    if type(init) is str:
        s0, P0 = spl.eigh(mu.cov())
        s1, P1 = spl.eigh(nu.cov())
        P0 = P0[:, ::-1]
        P1 = P1[:, ::-1]

    mu = gmm_transform(mu, b=-mu.mean())
    nu = gmm_transform(nu, b=-nu.mean())

    if type(init) is str:
        if init == "Gaussian":
            P = P0 @ I(mu.dim, nu.dim) @ P1.T
        elif init == "random":
            P = proj_stiefel(np.random.rand(mu.dim, nu.dim))
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
    # print('initialization, loss = ' + str(loss))
    n_iter = 0
    while n_iter < n_iter_max and np.abs(loss - loss_old) > eps:
        if verbose:
            print("iteration " + str(n_iter) + ": loss = " + str(loss))
        loss_old = loss
        P, _ = proj_gradient_descent(P, weights, mu, nu, alpha, n_iter=n_iter_P)
        weights, loss = compute_weights_loss(P, mu, nu)
        # print('iteration: ' + str(k+1) + ', loss = ' + str(loss))
        n_iter += 1

    return P, weights, loss


def EW2(X, Y, a=None, b=None, eps=1e-3, n_iter_max=10000, verbose=False):
    ### EW2 between two clouds of points (https://arxiv.org/pdf/1806.09277.pdf)
    m = X.shape[1]
    n = Y.shape[1]
    if a is None:
        a = ot.unif(X.shape[0])
    if b is None:
        b = ot.unif(Y.shape[0])
    P = proj_stiefel(np.random.rand(m, n))
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
):
    ### EW2 with annealed scheme (https://arxiv.org/pdf/1806.09277.pdf)
    m = X.shape[1]
    n = Y.shape[1]
    P = proj_stiefel(np.random.rand(m, n))
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


def MEW2(X, Y, n_components=20, annealing=True, n_iter_annealing=10, beta=0.99):
    ### MEW2 between two clouds of points (returns a distance)
    mix = sklmi.GaussianMixture(n_components=n_components)
    mix.fit(X)
    mu = GaussianMixture(mix.weights_, mix.means_, mix.covariances_)
    del mix
    mix = sklmi.GaussianMixture(n_components=n_components)
    mix.fit(Y)
    nu = GaussianMixture(mix.weights_, mix.means_, mix.covariances_)
    del mix
    if annealing:
        P, weights, loss = aEW2(
            mu.comp_mean,
            nu.comp_mean,
            n_iter_max=n_iter_annealing,
            beta=0.99,
            verbose=False,
        )
        P, weights, loss = MEW2_GM(
            mu,
            nu,
            alpha=0.01,
            eps=1e-3,
            n_iter_P=150,
            init=P,
            init_phase=False,
            symmetry=False,
            verbose=False,
        )
    else:
        P, weights, loss = MEW2_GM(
            mu,
            nu,
            alpha=0.01,
            eps=1e-3,
            n_iter_P=150,
            init_phase=True,
            symmetry=True,
            verbose=False,
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
):
    """
    MEW2 between two clouds of points (return a map between the points)
    If points = False, return the direct output of the T_mean or T_rand map (not in Y),
    if points = True, return the idxs of the points in Y where the points in X are transported at.
    """
    mix = sklmi.GaussianMixture(n_components=n_components)
    mix.fit(X)
    mu = GaussianMixture(mix.weights_, mix.means_, mix.covariances_)
    del mix
    mix = sklmi.GaussianMixture(n_components=n_components)
    mix.fit(Y)
    nu = GaussianMixture(mix.weights_, mix.means_, mix.covariances_)
    del mix
    if annealing:
        P, weights, loss = aEW2(
            mu.comp_mean,
            nu.comp_mean,
            n_iter_max=n_iter_annealing,
            beta=0.99,
            verbose=False,
        )
        P, weights, loss = MEW2_GM(
            mu,
            nu,
            alpha=0.01,
            eps=1e-3,
            n_iter_P=150,
            init=P,
            init_phase=False,
            symmetry=False,
            verbose=False,
        )
    else:
        P, weights, loss = MEW2_GM(
            mu,
            nu,
            alpha=0.01,
            eps=1e-3,
            n_iter_P=150,
            init_phase=True,
            symmetry=True,
            verbose=False,
        )
    if method == "T_mean":
        Z = T_mean(X, mu, nu, P, weights)
    elif method == "T_rand":
        Z = T_rand(X, mu, nu, P, weights)
    if return_both:
        nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(Y)
        idx = nbrs.kneighbors(Z, return_distance=False).ravel()
        return idx, Z
    elif points:
        nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(Y)
        idx = nbrs.kneighbors(Z, return_distance=False).ravel()
        return idx
    else:
        return Z


def to_numpy(x):
    """Convert list/array/tensor to numpy."""
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
    """
    Entropic partial OT via dummy node + Sinkhorn

    Parameters
    ----------
    a : (n,) array
        source weights (sum <= 1)
    b : (m,) array
        target weights (sum <= 1)
    M : (n, m) array
        cost matrix
    Lambda : float
        Lagrange multiplier (cost shift)
    reg : float
        entropic regularization
    numItermax : int
        max Sinkhorn iterations
    stopThr : float
        Sinkhorn stopping threshold
    remove_dummy : bool
        if True, return coupling without dummy row/column

    Returns
    -------
    gamma : (n, m) array or (n+1, m+1) array
        transport plan
    """

    # --- convert ---
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    M = np.asarray(M, dtype=float)

    # --- shift cost (Lagrangian trick) ---
    M = M - 2.0 * Lambda

    n, m = M.shape

    # --- extend marginals (dummy mass = 1) ---
    a_ext = np.append(a, 1.0)
    b_ext = np.append(b, 1.0)

    # --- extend cost matrix ---
    M_ext = np.zeros((n + 1, m + 1))
    M_ext[:n, :m] = M
    # dummy row / column are zero cost by default

    # --- Sinkhorn ---
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

    # --- remove dummy if requested ---
    if remove_dummy:
        return gamma_ext[:-1, :-1]

    return gamma_ext


def densite_theorique2d(mu, Sigma, alpha, x):
    """https://github.com/judelo/gmmot.git からの複製"""
    # compute the 2D GMM density with parameters (mu, Sigma) and weights alpha at x
    K = mu.shape[0]
    alpha = alpha.reshape(1, K)
    y = 0
    for j in range(K):
        y += alpha[0, j] * sps.multivariate_normal.pdf(
            x, mean=mu[j, :], cov=Sigma[j, :, :]
        )
    return y


def display_gmm(gmm, n=200, ax=0, bx=1, ay=0, by=1, cmap="gnuplot", axis=None):
    if axis is None:
        axis = plt.gca()

    K, pi, mu, S = gmm

    x = np.linspace(ax, bx, n)
    y = np.linspace(ay, by, n)
    X, Y = np.meshgrid(x, y)
    XX = np.column_stack([X.ravel(), Y.ravel()])
    Z = densite_theorique2d(mu, S, pi, XX).reshape(X.shape)

    Zmax = Z.max()
    levels = np.logspace(np.log10(Zmax * 1e-3), np.log10(Zmax), 8)

    norm = LogNorm(vmin=Zmax * 1e-3, vmax=Zmax)

    axis.contourf(X, Y, Z, levels=levels, cmap=cmap, norm=norm)
    #   axis.contour (X, Y, Z, levels=levels, colors='k', linewidths=0.3)
    #   axis.scatter(mu[:,0], mu[:,1], c='white', s=8, edgecolors='k')
    axis.set_aspect("equal")


def partial_wasserstein_lagrange_entropic(
    a, b, M, Lambda=None, epsilon=1e-2, nb_dummies=1, log=False, **kwargs
):
    """
    Entropic Lagrangian Partial Optimal Transport (Cuturi et al., 2016)
    Implemented with inequality-constrained Sinkhorn.
    Arguments:
        a, b : probability weights (sum <= 1)
        M    : cost matrix
        Lambda: Lagrange multiplier (lambda)
        epsilon : entropic regularization
    """

    # --- convert to numpy ---
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    M = np.asarray(M, float)

    # --- default λ ---
    if Lambda is None:
        Lambda = float(np.max(M)) + 1.0

    # --- modified cost ---
    C = M - Lambda

    # --- Gibbs kernel ---
    K = np.exp(-C / epsilon)  # entropic kernel

    # --- initialize dual variables ---
    u = np.ones_like(a)
    v = np.ones_like(b)

    # --- inequality-constrained Sinkhorn iterations ---
    for _ in range(2000):
        u_prev = u.copy()

        # inequality constraint: row sums <= a
        Ku = K @ v
        u = np.minimum(a / (Ku + 1e-300), 1.0)

        # inequality constraint: col sums <= b
        Kv = K.T @ u
        v = np.minimum(b / (Kv + 1e-300), 1.0)

        # convergence check
        if np.linalg.norm(u - u_prev) < 1e-14:
            break

    # --- transport plan ---
    gamma = np.diag(u) @ K @ np.diag(v)
    print(gamma)
    print(np.sum(gamma))

    if log:
        return gamma, {"transported_mass": np.sum(gamma), "cost": np.sum(gamma * M)}
    else:
        return gamma


def normalize_cost(C):
    m = np.max(C)
    if m > 0:
        C = C / m
    return C


def compute_T_X_to_Z(
    X, Y, n_components_X, n_components_Y, epsilon=1e-2, Lambda=None, nb_dummies=1
):
    # 1. Fit separate GMMs
    mixX = skGaussianMixture(n_components=n_components_X, covariance_type="full").fit(X)
    mixY = skGaussianMixture(n_components=n_components_Y, covariance_type="full").fit(Y)

    a = mixX.weights_
    b = mixY.weights_

    Kx = len(a)
    Ky = len(b)

    # 2. Gaussian-W2 cost

    def GaussianW2(mean0, cov0, mean1, cov1):
        mean0 = np.array(mean0)
        mean1 = np.array(mean1)
        cov0 = np.array(cov0)
        cov1 = np.array(cov1)

        sqrt_cov1 = sqrtm(cov1).real
        cross_term = sqrtm(sqrt_cov1 @ cov0 @ sqrt_cov1).real

        return (
            np.sum((mean0 - mean1) ** 2)
            + np.trace(cov0)
            + np.trace(cov1)
            - 2 * np.trace(cross_term)
        )

    M = np.zeros((Kx, Ky))
    for k in range(Kx):
        for l in range(Ky):
            M[k, l] = GaussianW2(
                mixX.means_[k],
                mixX.covariances_[k],
                mixY.means_[l],
                mixY.covariances_[l],
            )

    M = M / np.max(M)

    # 3. Partial OT
    res = partial_wasserstein_lagrange_entropic(
        a, b, M, Lambda=Lambda, epsilon=epsilon, nb_dummies=nb_dummies, log=True
    )
    gamma, _ = res  # shape (Kx, Ky)

    # 4. Barycentric projection (correct version)
    Z = np.zeros_like(X)
    d = X.shape[1]

    for i, x in enumerate(X):
        # ---- compute Gaussian densities (no weights!) ----
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

        # ---- mixture density ----
        denom = np.sum(a * p) + 1e-300

        # ----  π_k(x) ----
        pi_x = (a * p) / denom

        # ---- barycentric projection ----
        Tb = np.zeros_like(x)

        for k in range(Kx):
            if a[k] < 1e-15:
                continue  # avoid division by zero

            for l in range(Ky):
                Ak_l = compute_monge_map_matrix(
                    mixX.covariances_[k], mixY.covariances_[l]
                )
                Tkl = mixY.means_[l] + Ak_l @ (x - mixX.means_[k])

                Tb += (gamma[k, l] / a[k]) * pi_x[k] * Tkl

        Z[i] = Tb

    return Z


def compute_monge_map_matrix(CovK, CovL):
    """计算两个高斯分布之间的最优传输映射矩阵 A"""
    sqrtK = sqrtm(CovK).real
    inv_sqrtK = inv(sqrtK)
    # 核心公式: A = Σk^{-1/2} (Σk^{1/2} Σl Σk^{1/2})^{1/2} Σk^{-1/2}
    core = sqrtm(sqrtK @ CovL @ sqrtK).real
    return inv_sqrtK @ core @ inv_sqrtK


def compute_T_X_to_Z_C(
    X, Y, n_components_X, n_components_Y, epsilon=1e-2, Lambda=1e-1, nb_dummies=1
):
    # --- 1. 拟合源域和目标域的 GMM ---
    mixX = skGaussianMixture(
        n_components=n_components_X, covariance_type="full", random_state=42
    ).fit(X)
    mixY = skGaussianMixture(
        n_components=n_components_Y, covariance_type="full", random_state=42
    ).fit(Y)

    a = mixX.weights_  # 论文中的 a_k
    b = mixY.weights_  # 论文中的 b_l
    Kx, Ky = len(a), len(b)
    d = X.shape[1]

    # --- 2. 构建代价矩阵 M (Gaussian W2 Cost) ---
    def GaussianW2(mean0, cov0, mean1, cov1):
        """计算两个高斯分量之间的 Wasserstein-2 距离 (作为代价矩阵 M 的元素)"""
        mean0 = np.array(mean0)
        mean1 = np.array(mean1)
        sqrt_cov1 = sqrtm(cov1).real
        cross_term = sqrtm(sqrt_cov1 @ cov0 @ sqrt_cov1).real

        val = (
            np.sum((mean0 - mean1) ** 2)
            + np.trace(cov0)
            + np.trace(cov1)
            - 2 * np.trace(cross_term)
        )
        return max(0, val)

    M = np.zeros((Kx, Ky))
    for k in range(Kx):
        for l in range(Ky):
            M[k, l] = GaussianW2(
                mixX.means_[k],
                mixX.covariances_[k],
                mixY.means_[l],
                mixY.covariances_[l],
            )

    # 归一化代价矩阵以提高数值稳定性
    M = M / np.max(M)

    # --- 3. 运行 Partial OT (得到传输方案 omega) ---
    # 注意：这里调用你环境中的 partial_wasserstein_lagrange_entropic 函数
    # 该函数返回的 gamma 对应论文中的 omega^{\epsilon, \lambda}
    res = partial_wasserstein_lagrange_entropic(
        a, b, M, Lambda=Lambda, epsilon=epsilon, nb_dummies=nb_dummies, log=True
    )
    gamma, _ = res  # 形状为 (Kx, Ky)

    # --- 4. 预计算局部 Monge Map 的算子 A_kl ---
    A_matrices = {}
    for k in range(Kx):
        for l in range(Ky):
            # 只有在有质量传输的路径上计算变换，节省开销
            if gamma[k, l] > 1e-10:
                A_matrices[(k, l)] = compute_monge_map_matrix(
                    mixX.covariances_[k], mixY.covariances_[l]
                )

    # --- 5. 执行重心投影映射 ---
    Z = np.zeros_like(X)

    for i, x in enumerate(X):
        # 计算当前样本在源 GMM 各个分量下的似然密度 p_{\mu_k}(x)
        p_vals = np.zeros(Kx)
        for k in range(Kx):
            diff = x - mixX.means_[k]
            # 使用精准的高斯 PDF 公式
            inv_cov = inv(mixX.covariances_[k])
            exponent = -0.5 * diff @ inv_cov @ diff
            norm_const = np.sqrt((2 * np.pi) ** d * det(mixX.covariances_[k]))
            p_vals[k] = np.exp(exponent) / norm_const

        # 对应论文公式 (21) 的分子与分母
        numerator = np.zeros(d)
        denominator = 0.0

        for k in range(Kx):
            for l in range(Ky):
                # 分母累计: \sum gamma_{kl} * p_k(x)
                denominator += gamma[k, l] * p_vals[k]

                if (k, l) in A_matrices:
                    # 局部变换: T^{k,l}_{W2}(x) = \mu_l + A_{kl}(x - \mu_k)
                    Tkl = mixY.means_[l] + A_matrices[(k, l)] @ (x - mixX.means_[k])

                    # 分子累计: \sum \omega_{kl} * p_k(x) * Tkl
                    numerator += gamma[k, l] * p_vals[k] * Tkl

        # 计算重心投影结果: T_b(x)
        Z[i] = numerator / (denominator + 1e-300)

    return Z


def sample_from_gmm(alpha, means, covs, N):
    """
    alpha: (K,) weights
    means: (K, d)
    covs:  (K, d, d)
    N: number of samples

    return: (N, d) sampled points
    """
    K, d = means.shape
    samples = np.zeros((N, d))

    # どの成分からサンプルするか
    comp_ids = np.random.choice(K, size=N, p=alpha)

    # 成分ごとに多変量正規分布から生成
    for k in range(K):
        idx = np.where(comp_ids == k)[0]
        if len(idx) > 0:
            samples[idx] = np.random.multivariate_normal(
                mean=means[k], cov=covs[k], size=len(idx)
            )
    return samples


def gaussian_w2_squared(mean_0,mean_1,Cov_0,Cov_1):
    ### squared W_2 distance between Gaussian distributions (this returns W_2^2, not W_2)
    sqCov_1 = spl.sqrtm(Cov_1).real
    return spl.norm(mean_0 - mean_1)**2 + np.trace(Cov_0) + np.trace(Cov_1) - 2*np.trace(spl.sqrtm(sqCov_1@Cov_0@sqCov_1).real)

def gaussian_cost_matrix(gmm):
    K = gmm.K
    C = np.zeros((K,K))
    for i in range(K):
        for j in range(K):
            C[i,j] = gaussian_w2_squared(gmm.comp_mean[i], gmm.comp_mean[j],
                                gmm.comp_cov[i],  gmm.comp_cov[j])
    return C


def validate_partial_coupling(gamma, a, b, tol=1e-8):
    ### check that gamma is a feasible (non-negative, sub-marginal) partial coupling
    gamma = np.asarray(gamma, dtype=float)
    if np.any(gamma < -tol):
        raise ValueError("Partial coupling Gamma has negative entries.")
    if np.any(gamma.sum(axis=1) > np.asarray(a) + tol):
        raise ValueError("Partial coupling violates the source marginal a (Gamma 1 <= a).")
    if np.any(gamma.sum(axis=0) > np.asarray(b) + tol):
        raise ValueError("Partial coupling violates the target marginal b (Gamma^T 1 <= b).")
    Z = float(gamma.sum())
    if not (0.0 < Z <= 1.0 + tol):
        raise ValueError(f"Z_lambda = {Z} is out of the expected (0, 1] range.")
    return Z


def matched_statistics(gamma, mu, nu, mass_tol=1e-14):
    ### Z_lambda, row/col matched masses, and matched means m0_lambda, m1_lambda
    gamma = np.asarray(gamma, dtype=float)
    Z = float(gamma.sum())
    if not np.isfinite(Z) or Z <= mass_tol:
        raise ValueError("The partial coupling has zero matched mass (Z_lambda <= 0); "
                          "barycentric map is undefined.")
    row_mass = gamma.sum(axis=1)   # r_k = sum_l Gamma[k,l]
    col_mass = gamma.sum(axis=0)   # s_l = sum_k Gamma[k,l]
    m0 = (row_mass[:, None] * mu.comp_mean).sum(axis=0) / Z
    m1 = (col_mass[:, None] * nu.comp_mean).sum(axis=0) / Z
    return Z, row_mass, col_mass, m0, m1


def alignment_loss(P, gamma, mu_c, nu_c):
    ### J_Gamma(P) = sum_{k,l} Gamma[k,l] * W2^2(centered_mu_k, P_# centered_nu_l)
    ### mu_c, nu_c must already be centered on the MATCHED means m0_lambda, m1_lambda
    value = 0.0
    for k in range(mu_c.K):
        for l in range(nu_c.K):
            if gamma[k, l] == 0.0:
                continue
            target_mean = P @ nu_c.comp_mean[l]
            target_cov = P @ nu_c.comp_cov[l] @ P.T
            value += gamma[k, l] * gaussian_w2_squared(
                mu_c.comp_mean[k], target_mean, mu_c.comp_cov[k], target_cov
            )
    return float(value)


def alignment_grad(P, gamma, mu_c, nu_c):
    ### gradient of J_Gamma(P); reuses the existing grad_gauss/grad formulas,
    ### but on matched-centered GMMs so it matches alignment_loss exactly.
    return grad(P, gamma, mu_c, nu_c)


def partial_projected_gradient_descent(P0, gamma, mu_c, nu_c, step_size=1.0, max_iter=150, tol=1e-8):
    ### projected gradient descent on the Stiefel manifold for J_Gamma(P);
    ### the recorded loss and the update direction are the same functional.
    P = P0.copy()
    losses = [alignment_loss(P, gamma, mu_c, nu_c)]
    for _ in range(max_iter):
        G = alignment_grad(P, gamma, mu_c, nu_c)
        P_next = proj_stiefel(P - step_size * G)
        losses.append(alignment_loss(P_next, gamma, mu_c, nu_c))
        if np.linalg.norm(P_next - P) <= tol * (1.0 + np.linalg.norm(P)):
            P = P_next
            break
        P = P_next
    return P, losses


def partial_mgw_barycentric_map(X, mu, nu, P, gamma, m0_match, m1_match, density_tol=None):
    """
    Partial barycentric map T_b^{GW,lambda}(x), normalized by the MATCHED
    source density (not the full mixture density mu.pdf(x)):

        T_b^{GW,lambda}(x) =
            [ sum_{k,l} Gamma[k,l] p_mu_k(x) R_kl(x) ]
          / [ sum_{k,l} Gamma[k,l] p_mu_k(x) ]
          = [ sum_k  r_k p_mu_k(x)  R_k(x) ] / [ sum_k r_k p_mu_k(x) ]

    where R_kl is built from the optimal Gaussian map between the
    matched-centered mu_k and the P-embedded, matched-centered nu_l.
    Raises FloatingPointError if the matched source density underflows to
    zero, instead of silently perturbing gamma.
    """
    X = np.asarray(X, dtype=float)
    single = X.ndim == 1
    X2 = X[None, :] if single else X

    mu_P_centered = gmm_transform(mu, P=P.T, b=-P.T @ m0_match)
    nu_centered = gmm_transform(nu, b=-m1_match)

    component_pdf = np.vstack([mu.comp[k].pdf(X2) for k in range(mu.K)])  # (K, n)
    row_mass = gamma.sum(axis=1)  # r_k
    denominator = (row_mass[:, None] * component_pdf).sum(axis=0)

    if density_tol is None:
        density_tol = np.finfo(float).tiny
    if np.any(denominator <= density_tol):
        raise FloatingPointError(
            "Matched source density underflowed to zero at one or more query points; "
            "the barycentric map is not defined there."
        )

    X_P = np.einsum('ij,bj->bi', P.T, X2 - m0_match)

    numerator = np.zeros((X2.shape[0], nu.dim), dtype=float)
    for k in range(mu.K):
        for l in range(nu.K):
            if gamma[k, l] == 0.0:
                continue
            R = T_map_Gaussian(X_P, mu_P_centered.comp[k], nu_centered.comp[l])
            numerator += gamma[k, l] * component_pdf[k][:, None] * R

    mapped = numerator / denominator[:, None] + m1_match
    return mapped[0] if single else mapped


def pMGW2_coup(
    X, Y, Lambda, n_components_X, n_components_Y,
    solver_tol=1e-12, step_size=1.0, max_alignment_iter=150,
    normalize_costs=True, points=True, return_both=False, verbose=False,
):
    """
    Partial mixture Gromov-Wasserstein coupling and (partial) barycentric map.

    Corrections applied vs. the original version (see correction guide):
      - Gamma (output of partial_gromov_ver1) is used as-is everywhere:
        no "Gamma + eps" perturbation, and the SAME Gamma is used for the
        alignment step and for the barycentric map.
      - centering uses matched statistics (Gamma's row/column masses),
        not the full-mixture means mu.mean()/nu.mean().
      - the alignment loss and its gradient are the same functional J_Gamma(P).
      - the barycentric map is normalized by the matched source density.
      - `annealing` / `n_iter_annealing` are removed (unused on this path);
        `reg` is renamed `solver_tol` since it is the solver's stopping
        tolerance, not an entropic regularization parameter.
      - T_rand is not used on the partial path: for a partial coupling its
        per-point assignment probabilities do not sum to 1 (see guide 4.6).

    Lambda is interpreted on the NORMALIZED cost scale (i.e. after C1, C2
    are divided by M = max(C1.max(), C2.max())), when normalize_costs=True.
    If you have a Lambda tuned for the raw (non-normalized) cost scale,
    convert it first: Lambda_normalized = Lambda_raw / M**2.
    """
    # 1. Fit GMMs
    mix = sklmi.GaussianMixture(n_components=n_components_X)
    mix.fit(X)
    mu = GaussianMixture(mix.weights_, mix.means_, mix.covariances_)
    del mix
    mix = sklmi.GaussianMixture(n_components=n_components_Y)
    mix.fit(Y)
    nu = GaussianMixture(mix.weights_, mix.means_, mix.covariances_)
    del mix

    # 2. Intra-mixture W2^2 cost matrices, with a zero-guard on normalization
    C1 = gaussian_cost_matrix(mu)
    C2 = gaussian_cost_matrix(nu)
    M = max(float(C1.max()), float(C2.max()))
    if normalize_costs and M > 0.0:
        C1_solver, C2_solver = C1 / M, C2 / M
    else:
        C1_solver, C2_solver = C1.copy(), C2.copy()

    a = mu.weights
    b = nu.weights

    if verbose:
        print(f'Deriving partial MGW coupling with Lambda={Lambda}')

    # 3. Solve the non-entropic partial GW coupling (p=q=2, square loss).
    start_time = time.time()
    Gamma = gromov.partial_gromov_ver1(
        C1_solver, C2_solver, a, b,
        Lambda=Lambda,
        nb_dummies=1,
        G0=None,
        thres=1,
        numItermax=None,
        numItermax_gw=1000,
        tol=solver_tol,
        log=False,
        verbose=verbose,
        line_search=True,
    )
    if np.isnan(Gamma).any():
        raise FloatingPointError("Partial GW solver returned NaN entries in Gamma.")

    # feasibility check: Gamma is used as-is from here on, never perturbed
    Z = validate_partial_coupling(Gamma, a, b)
    if verbose:
        print(f"Calculation time: {time.time() - start_time:.6f} sec")
        print(f"Total matched mass Z_lambda = {Z:.6f}")

    # 4. Matched centering (replaces full-mixture means mu.mean()/nu.mean())
    Z, row_mass, col_mass, m0, m1 = matched_statistics(Gamma, mu, nu)

    # 5. Alignment: loss and gradient share the same centered, matched objective
    mu_c = gmm_transform(mu, b=-m0)
    nu_c = gmm_transform(nu, b=-m1)
    cross = sum(
        Gamma[k, l] * np.outer(mu_c.comp_mean[k], nu_c.comp_mean[l])
        for k in range(mu.K) for l in range(nu.K)
    )
    P0 = proj_stiefel(cross)
    P, losses = partial_projected_gradient_descent(
        P0, Gamma, mu_c, nu_c,
        step_size=step_size, max_iter=max_alignment_iter,
    )

    # 6. Partial barycentric map, normalized by the matched source density
    Z_map = partial_mgw_barycentric_map(X, mu, nu, P, Gamma, m0, m1)

    # 7. Nearest-neighbor projection onto the target point cloud
    if return_both or points:
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(Y)
        idx = nbrs.kneighbors(Z_map, return_distance=False).ravel()
        if return_both:
            return idx, Z_map
        return idx
    return Z_map
