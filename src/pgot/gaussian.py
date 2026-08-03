"""Gaussian-mixture models, Gaussian W2 geometry, and transport maps."""

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as spl
from scipy.stats import multivariate_normal

from .reproducibility import get_rng


class Gaussian:
    """Gaussian distribution described by a mean and covariance matrix."""

    def __init__(self, mean, cov):
        self.mean = np.array(mean)
        self.cov = np.array(cov)
        self.dim = len(mean)

    def sample(self, n_samples, rng=None):
        return get_rng(rng).multivariate_normal(self.mean, self.cov, n_samples)

    def pdf(self, x):
        return multivariate_normal.pdf(x, mean=self.mean, cov=self.cov)


class GaussianMixture:
    """Finite mixture of Gaussian distributions."""

    def __init__(self, weights_list, mean_list, cov_list):
        self.comp = [Gaussian(mean_list[k], cov_list[k]) for k in range(len(mean_list))]
        self.comp_mean = np.array(mean_list)
        self.comp_cov = np.array(cov_list)
        self.weights = np.array(weights_list)
        self.dim = len(mean_list[0])
        self.K = len(mean_list)

    def sample(self, n_samples, return_K=False, rng=None):
        """Sample points and, optionally, their component indices."""
        rng = get_rng(rng)
        idx_list = rng.choice(np.arange(self.K), n_samples, p=self.weights)
        samples_list = [
            rng.multivariate_normal(self.comp_mean[i], self.comp_cov[i])
            for i in idx_list
        ]
        if return_K:
            return np.array(samples_list), idx_list
        return np.array(samples_list)

    def mean(self):
        """Return the mean of the Gaussian mixture."""
        return np.average(self.comp_mean, axis=0, weights=self.weights)

    def cov(self):
        """Return the covariance matrix of the Gaussian mixture."""
        second_moment = sum(
            self.weights[k]
            * (self.comp_cov[k] + np.outer(self.comp_mean[k], self.comp_mean[k]))
            for k in range(self.K)
        )
        mean = self.mean()
        return second_moment - np.outer(mean, mean)

    def pdf(self, x):
        return sum(
            self.weights[k]
            * multivariate_normal.pdf(
                x,
                mean=self.comp_mean[k],
                cov=self.comp_cov[k],
            )
            for k in range(self.K)
        )

    def plot_scatter_2d(self, n_samples, T=None, *args, rng=None):
        """Plot samples, optionally after applying a transport map."""
        X, idx_list = self.sample(n_samples, True, rng=rng)
        if T is not None:
            Y = np.array([T(X[k], idx_list[k], *args) for k in range(len(X))])
        else:
            Y = X
        for k in range(self.K):
            Z = Y[idx_list == k]
            plt.scatter(Z[:, 0], Z[:, 1], color=X[idx_list == k][:, 0], s=10)
        plt.axis("equal")

    def pdist(self, f):
        """Compute the pairwise component-distance matrix."""
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
    """Apply the affine transformation ``x -> P x + b`` to a GMM."""
    if P is None:
        P = np.eye(mu.dim)
    if b is None:
        b = np.zeros(P.shape[0])
    mean_list = [P.dot(m) + b for m in mu.comp_mean]
    cov_list = [P @ covariance @ P.T for covariance in mu.comp_cov]
    return GaussianMixture(mu.weights, mean_list, cov_list)


def I(m, n):
    """Return the rectangular matrix ``(I_n 0)^T``."""
    identity = np.zeros((m, n))
    identity[:n, :n] = np.eye(n)
    return identity


def proj_stiefel(A):
    """Project a matrix onto the Stiefel manifold."""
    m, n = A.shape
    U, _, VT = spl.svd(A)
    return U @ I(m, n) @ VT


def gaussian_w2_squared(mean_0, mean_1, Cov_0, Cov_1):
    """Return the squared W2 distance between two Gaussian distributions."""
    sqCov_1 = spl.sqrtm(Cov_1).real
    return (
        spl.norm(mean_0 - mean_1) ** 2
        + np.trace(Cov_0)
        + np.trace(Cov_1)
        - 2 * np.trace(spl.sqrtm(sqCov_1 @ Cov_0 @ sqCov_1).real)
    )


def GaussianW2(mean_0, mean_1, Cov_0, Cov_1):
    """Backward-compatible name for squared Gaussian W2 distance."""
    return gaussian_w2_squared(mean_0, mean_1, Cov_0, Cov_1)


def gaussian_cost_matrix(gmm):
    """Return the intra-mixture squared Gaussian W2 cost matrix."""
    cost = np.zeros((gmm.K, gmm.K))
    for i in range(gmm.K):
        for j in range(gmm.K):
            cost[i, j] = gaussian_w2_squared(
                gmm.comp_mean[i],
                gmm.comp_mean[j],
                gmm.comp_cov[i],
                gmm.comp_cov[j],
            )
    return cost


def gaussian_W(gmm1, gmm2):
    """Return the cross-mixture squared Gaussian W2 cost matrix."""
    cost = np.zeros((gmm1.K, gmm2.K))
    for i in range(gmm1.K):
        for j in range(gmm2.K):
            cost[i, j] = GaussianW2(
                gmm1.comp_mean[i],
                gmm2.comp_mean[j],
                gmm1.comp_cov[i],
                gmm2.comp_cov[j],
            )
    return cost


def grad_gauss(P, Cov_0, Cov_1):
    """Gradient of the covariance contribution to Gaussian W2."""
    sqCov_1 = spl.sqrtm(Cov_1).real
    A = Cov_0 @ P @ sqCov_1
    B = spl.inv(spl.sqrtm(sqCov_1 @ P.T @ A)).real
    return A @ B @ sqCov_1


def grad(P, weights, mu, nu):
    """Gradient of the weighted Gaussian alignment objective."""
    return 2 * sum(
        weights[k, l]
        * (
            -np.outer(mu.comp_mean[k], nu.comp_mean[l])
            + P @ np.outer(nu.comp_mean[l], nu.comp_mean[l])
            - grad_gauss(P, mu.comp_cov[k], nu.comp_cov[l])
        )
        for k in range(mu.K)
        for l in range(nu.K)
    )


def proj_gradient_descent(P, weights, mu, nu, alpha, n_iter=150):
    """Run projected gradient descent on the Stiefel manifold."""
    loss = [
        sum(
            weights[k, l]
            * GaussianW2(0, 0, mu.comp_cov[k], P @ nu.comp_cov[l] @ P.T)
            for k in range(mu.K)
            for l in range(nu.K)
        )
    ]
    for _ in range(n_iter):
        P = proj_stiefel(P - alpha * grad(P, weights, mu, nu))
        loss.append(
            sum(
                weights[k, l]
                * GaussianW2(0, 0, mu.comp_cov[k], P @ nu.comp_cov[l] @ P.T)
                for k in range(mu.K)
                for l in range(nu.K)
            )
        )
    return P, loss


def T_map_Gaussian(x, mu, nu):
    """Apply the optimal W2 map between two Gaussian distributions."""
    invcov0 = spl.inv(spl.sqrtm(mu.cov).real)
    sqcov0 = spl.sqrtm(mu.cov).real
    A = invcov0 @ spl.sqrtm(sqcov0 @ nu.cov @ sqcov0) @ invcov0
    if len(x.shape) == 1:
        return nu.mean + A @ (x - mu.mean)
    return nu.mean + np.einsum("ij,bj -> bi", A, x - mu.mean)


def T_map_Gaussian_rand(x, mu, nu, idx_list):
    """Apply component-wise optimal Gaussian maps selected by ``idx_list``."""
    invcov0 = [spl.inv(spl.sqrtm(mu.comp[k].cov).real) for k in range(mu.K)]
    sqcov0 = [spl.sqrtm(mu.comp[k].cov).real for k in range(mu.K)]
    A_list = [
        [
            invcov0[k]
            @ spl.sqrtm(sqcov0[k] @ nu.comp[l].cov @ sqcov0[k])
            @ invcov0[k]
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
    """Apply the deterministic mixture barycentric transport map."""
    mu_P_centered = gmm_transform(mu, P=P.T, b=-P.T @ mu.mean())
    nu_centered = gmm_transform(nu, b=-nu.mean())
    pdf_comp = [mu.comp[k].pdf(X) for k in range(mu.K)]
    ipdf_mu = 1.0 / mu.pdf(X)
    if len(X.shape) == 1:
        X_P = P.T @ (X - mu.mean())
        Y = (
            sum(
                weights[k, l]
                * pdf_comp[k]
                * T_map_Gaussian(
                    X_P,
                    mu_P_centered.comp[k],
                    nu_centered.comp[l],
                )
                for k in range(mu.K)
                for l in range(nu.K)
            )
            * ipdf_mu
        )
    else:
        X_P = np.einsum("ij,bj->bi", P.T, X - mu.mean())
        Y = np.einsum(
            "bi,b -> bi",
            sum(
                weights[k, l]
                * np.einsum(
                    "bi,b -> bi",
                    T_map_Gaussian(
                        X_P,
                        mu_P_centered.comp[k],
                        nu_centered.comp[l],
                    ),
                    pdf_comp[k],
                )
                for k in range(mu.K)
                for l in range(nu.K)
            ),
            ipdf_mu,
        )
    return Y + nu.mean()


def T_rand(x, mu, nu, P, weights, rng=None):
    """Apply the randomized mixture transport map."""
    mu_P_centered = gmm_transform(mu, P=P.T, b=-P.T @ mu.mean())
    nu_centered = gmm_transform(nu, b=-nu.mean())
    rng = get_rng(rng)
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


def sample_from_gmm(alpha, means, covs, N, rng=None):
    """Sample ``N`` points from a Gaussian mixture."""
    rng = get_rng(rng)
    K, d = means.shape
    samples = np.zeros((N, d))
    comp_ids = rng.choice(K, size=N, p=alpha)
    for k in range(K):
        idx = np.where(comp_ids == k)[0]
        if len(idx) > 0:
            samples[idx] = rng.multivariate_normal(
                mean=means[k],
                cov=covs[k],
                size=len(idx),
            )
    return samples
