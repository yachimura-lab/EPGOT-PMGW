"""Canonical data, solves and plots for paper Figures 1--7.

The notebooks deliberately stay thin: numerical work is performed here once,
then the returned solve records are used for plotting and provenance sidecars.
"""

from __future__ import annotations

from time import perf_counter

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import ot
from lib import gromov
from matplotlib.colors import Normalize
from sklearn.mixture import GaussianMixture as SkGaussianMixture

from .gaussian import GaussianMixture, gaussian_W, gaussian_cost_matrix
from .metadata import solve_id
from .mgw import MGW2_GM_coup, MGW2_coup
from .partial_mgw import (
    barycentric_projection,
    gw_mean_distortion,
    nearest_target_indices,
    pMGW2_coup,
    partial_gw_penalty,
)
from .partial_ot import (
    display_gmm,
    entropic_partial_barycentric_map,
    entropic_partial_coupling,
    entropic_partial_ot,
)
from .reproducibility import DEFAULT_SEED


FIGURE_PARAMETERS = {
    1: {"lambda": 0.3, "epsilon": 0.01},
    2: {
        "lambda": [0.0, 0.125, 0.25, 0.375, 0.5],
        "epsilon": 0.01,
    },
    3: {
        "lambda": [0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5],
        "epsilon": [0.01, 0.11, 0.21],
    },
    4: {
        "lambda_epsilon": [[0.25, 0.03], [0.25, 0.1], [0.5, 0.03]],
        "t": [0.0, 0.25, 0.5, 0.75, 1.0],
    },
    5: {
        "lambda": [0.25, 0.5],
        "epsilon": 0.03,
        "t": [0.2, 0.4, 0.6, 0.8, 1.0],
        "samples_per_mixture": 2000,
    },
    6: {
        "paper_lambda": 0.01,
        "ring_radius": 4.0,
        "source_points": 300,
        "target_ring_points": 300,
        "target_noise_points": 10,
        "source_components": 6,
        "target_components": 7,
    },
    7: {"shared_with_figure": 6},
}


def paper_gaussian_mixtures():
    """Known source and target mixtures from Section 7.1."""
    source = GaussianMixture(
        np.array([0.5, 0.5]),
        np.array([[-8.0, -5.0], [1.0, -8.0]]),
        0.2 * np.array([[[1.0, 0.2], [0.2, 1.0]], [[1.0, -0.1], [-0.1, 1.0]]]),
    )
    target = GaussianMixture(
        np.array([0.45, 0.45, 0.1]),
        np.array([[-3.5, 3.5], [3.5, 3.5], [9.0, 7.5]]),
        0.2
        * np.array(
            [
                [[1.0, -0.1], [-0.1, 1.0]],
                [[1.0, 0.2], [0.2, 1.0]],
                [[1.0, 0.0], [0.0, 1.0]],
            ]
        ),
    )
    return source, target


def normalized_cross_cost(source, target):
    """Cross Gaussian-W2 cost and the scale used by (7.1)."""
    raw = gaussian_W(source, target)
    scale = float(raw.max())
    return raw / scale, scale


def solve_epot_panels(lambdas, epsilons, source=None, target=None):
    """Solve each requested Figure 1--3 panel once with residual logs."""
    if source is None or target is None:
        source, target = paper_gaussian_mixtures()
    cost, cost_scale = normalized_cross_cost(source, target)
    records = []
    for epsilon in np.atleast_1d(epsilons):
        for paper_lambda in np.atleast_1d(lambdas):
            coupling, info = entropic_partial_ot(
                source.weights,
                target.weights,
                cost,
                Lambda=float(paper_lambda),
                reg=float(epsilon),
                numItermax=20_000,
                stopThr=1e-10,
                log=True,
            )
            if not info["converged"]:
                raise RuntimeError(
                    "EPOT panel did not meet its declared marginal tolerance: "
                    f"lambda={paper_lambda}, epsilon={epsilon}, {info}"
                )
            parameters = {
                "paper_lambda": float(paper_lambda),
                "solver_lambda": float(paper_lambda),
                "epsilon": float(epsilon),
            }
            records.append(
                {
                    **parameters,
                    **info,
                    "coupling": coupling,
                    "solve_id": solve_id(info["solver"], coupling, parameters),
                    "cost_normalization_scale": cost_scale,
                }
            )
    return records


def _draw_mixtures(axis, source, target, limits=(-12, 12)):
    lo, hi = limits
    display_gmm(
        [source.K, source.weights, source.comp_mean, source.comp_cov],
        n=180,
        ax=lo,
        bx=hi,
        ay=lo,
        by=hi,
        cmap="Reds",
        axis=axis,
    )
    display_gmm(
        [target.K, target.weights, target.comp_mean, target.comp_cov],
        n=180,
        ax=lo,
        bx=hi,
        ay=lo,
        by=hi,
        cmap="Blues",
        axis=axis,
    )
    axis.scatter(*source.comp_mean.T, c="black", s=18, zorder=4)
    axis.scatter(*target.comp_mean.T, c="black", s=18, zorder=4)
    axis.set_xlim(lo, hi)
    axis.set_ylim(lo, hi)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xticks([])
    axis.set_yticks([])


def _draw_coupling(axis, source, target, coupling, norm, cmap, threshold=1e-10):
    for i in range(source.K):
        for j in range(target.K):
            weight = float(coupling[i, j])
            if weight <= threshold:
                continue
            start, end = source.comp_mean[i], target.comp_mean[j]
            axis.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color=cmap(norm(weight)),
                linewidth=0.4 + 7.0 * weight / max(norm.vmax, 1e-15),
                alpha=1.0,
                zorder=3,
            )


def _plot_epot_grid(records, nrows, ncols, figsize):
    source, target = paper_gaussian_mixtures()
    vmax = max(float(np.max(record["coupling"])) for record in records)
    norm = Normalize(vmin=0.0, vmax=vmax)
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    for axis, record in zip(axes.ravel(), records):
        _draw_mixtures(axis, source, target)
        _draw_coupling(axis, source, target, record["coupling"], norm, cmap)
        axis.set_title(
            rf"$\lambda={record['paper_lambda']:g}$, "
            rf"$\varepsilon={record['epsilon']:g}$"
            "\n"
            rf"$Z_\lambda={record['matched_mass']:.4f}$",
            fontsize=10,
        )
    scalar = cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    fig.colorbar(
        scalar,
        ax=list(axes.ravel()),
        orientation="horizontal",
        fraction=0.045,
        pad=0.05,
        label=r"coupling weight $\omega_{ij}^{\varepsilon,\lambda}$",
    )
    return fig


def make_figure1(records=None):
    """Figure 1: continuous coupling weights and horizontal colorbar."""
    records = records or solve_epot_panels([0.3], [0.01])
    return _plot_epot_grid(records, 1, 1, (7, 7))


def make_figure2(records=None):
    """Figure 2: the exact 1-by-5 lambda sweep from the paper."""
    params = FIGURE_PARAMETERS[2]
    records = records or solve_epot_panels(params["lambda"], [params["epsilon"]])
    return _plot_epot_grid(records, 1, 5, (22, 5))


def make_figure3(records=None):
    """Figure 3: 3 epsilon rows by 7 lambda columns."""
    params = FIGURE_PARAMETERS[3]
    records = records or solve_epot_panels(params["lambda"], params["epsilon"])
    return _plot_epot_grid(records, 3, 7, (22, 12))


def solve_figure5(source_points, source, target, lambdas=(0.25, 0.5), epsilon=0.03):
    """Solve Figure 5 once per lambda and reuse each coupling for its map."""
    records = []
    for paper_lambda in lambdas:
        coupling, info = entropic_partial_coupling(
            source, target, epsilon=epsilon, Lambda=paper_lambda, log=True
        )
        mapped = entropic_partial_barycentric_map(
            source_points,
            source,
            target,
            epsilon=epsilon,
            Lambda=paper_lambda,
            coupling=coupling,
        )
        parameters = {"paper_lambda": paper_lambda, "epsilon": epsilon}
        records.append(
            {
                **parameters,
                **info,
                "coupling": coupling,
                "mapped": mapped,
                "solve_id": solve_id(info["solver"], coupling, parameters),
            }
        )
    return records


def make_figure5(source_points, target_points, records, times=None):
    """Figure 5's canonical two-row composite layout."""
    times = np.asarray(times if times is not None else FIGURE_PARAMETERS[5]["t"])
    fig, axes = plt.subplots(len(records), len(times), figsize=(20, 8), squeeze=False)
    for row, record in enumerate(records):
        for col, t in enumerate(times):
            axis = axes[row, col]
            interpolated = (1.0 - t) * source_points + t * record["mapped"]
            axis.scatter(*source_points.T, s=3, c="red", alpha=0.18)
            axis.scatter(*target_points.T, s=3, c="blue", alpha=0.18)
            axis.scatter(*interpolated.T, s=4, c="green", alpha=0.65)
            axis.set_xlim(-12, 12)
            axis.set_ylim(-12, 12)
            axis.set_aspect("equal", adjustable="box")
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(rf"$t={t:.1f}$")
            if col == 0:
                axis.set_ylabel(
                    rf"$\lambda={record['paper_lambda']}$"
                    "\n"
                    rf"$Z_\lambda={record['matched_mass']:.4f}$"
                )
    fig.tight_layout()
    return fig


def generate_figure67_data(
    *,
    seed=DEFAULT_SEED,
    components=6,
    points=300,
    sigma=0.2,
    radius=4.0,
    height=10.0,
    noise_points=10,
    noise_radius=0.3,
):
    """Generate independent, positive-definite ring samples for Figures 6--7."""
    if points % components:
        raise ValueError("points must be divisible by components")
    rng_source = np.random.default_rng(seed)
    rng_target = np.random.default_rng(seed + 1)
    rng_noise = np.random.default_rng(seed + 2)
    theta = np.linspace(0, 2 * np.pi, components, endpoint=False)
    source_means = np.stack(
        [radius * np.cos(theta), radius * np.sin(theta), np.zeros(components)], axis=1
    )
    target_means = source_means + np.array([0.0, 0.0, height])
    covariance = sigma**2 * np.eye(3)

    def sample_ring(rng, means):
        return np.vstack(
            [
                rng.multivariate_normal(mean, covariance, points // components)
                for mean in means
            ]
        )

    source = sample_ring(rng_source, source_means)
    target_ring = sample_ring(rng_target, target_means)
    noise_center = np.array([2.5 * radius, 0.0, height])
    directions = rng_noise.normal(size=(noise_points, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = noise_radius * rng_noise.uniform(0, 1, noise_points) ** (1 / 3)
    noise = noise_center + directions * radii[:, None]
    return {
        "source": source,
        "target_ring": target_ring,
        "noise": noise,
        "target": np.vstack([target_ring, noise]),
        "source_means": source_means,
        "target_means": target_means,
        "covariance": covariance,
        "seed": seed,
        "radius": radius,
        "height": height,
        "noise_radius": noise_radius,
        "noise_center": noise_center,
    }


def fit_figure67_problem(data, random_state=DEFAULT_SEED):
    """Fit the shared 6/7 GMMs and construct both normalized cost pairs."""
    source_points, target_points = data["source"], data["target"]

    def fit(points, components):
        model = SkGaussianMixture(
            n_components=components,
            covariance_type="full",
            random_state=random_state,
        ).fit(points)
        return GaussianMixture(model.weights_, model.means_, model.covariances_)

    source_gmm = fit(source_points, 6)
    target_gmm = fit(target_points, 7)
    component_source = gaussian_cost_matrix(source_gmm)
    component_target = gaussian_cost_matrix(target_gmm)
    component_scale = float(max(component_source.max(), component_target.max()))
    point_source = ot.dist(source_points, source_points)
    point_target = ot.dist(target_points, target_points)
    point_scale = float(max(point_source.max(), point_target.max()))
    return {
        **data,
        "source_gmm": source_gmm,
        "target_gmm": target_gmm,
        "component_source_cost": component_source,
        "component_target_cost": component_target,
        "component_source_cost_normalized": component_source / component_scale,
        "component_target_cost_normalized": component_target / component_scale,
        "component_cost_scale": component_scale,
        "point_source_cost_normalized": point_source / point_scale,
        "point_target_cost_normalized": point_target / point_scale,
        "point_cost_scale": point_scale,
        "fit_random_state": random_state,
    }


def solve_figure67(problem, paper_lambda=0.01):
    """Solve each method once; Figures 6 and 7 consume the same maps."""
    source, target = problem["source"], problem["target"]
    source_gmm, target_gmm = problem["source_gmm"], problem["target_gmm"]
    point_source = problem["point_source_cost_normalized"]
    point_target = problem["point_target_cost_normalized"]
    component_source = problem["component_source_cost_normalized"]
    component_target = problem["component_target_cost_normalized"]
    p = ot.unif(len(source))
    q = ot.unif(len(target))

    started = perf_counter()
    coupling_gw = ot.gromov.gromov_wasserstein(
        point_source, point_target, p, q, loss_fun="square_loss"
    )
    runtime_gw = perf_counter() - started
    started = perf_counter()
    coupling_mgw = MGW2_GM_coup(
        source_gmm, target_gmm, C1=component_source, C2=component_target
    )
    runtime_mgw_coupling = perf_counter() - started

    point_distortion = gw_mean_distortion(point_source, point_target, coupling_gw)
    lambda_tilde = paper_lambda / point_distortion
    lambda_point = partial_gw_penalty(
        lambda_tilde, point_source, point_target, coupling_gw
    )
    lambda_component = partial_gw_penalty(
        lambda_tilde, component_source, component_target, coupling_mgw
    )

    started = perf_counter()
    coupling_pgw = gromov.partial_gromov_ver1(
        point_source,
        point_target,
        p,
        q,
        Lambda=lambda_point,
        numItermax_gw=100,
        tol=1e-12,
        verbose=False,
    )
    runtime_pgw = perf_counter() - started
    map_gw, mask_gw = barycentric_projection(coupling_gw, target)
    map_pgw, mask_pgw = barycentric_projection(coupling_pgw, target)
    index_gw = nearest_target_indices(map_gw, target)
    index_pgw = nearest_target_indices(map_pgw, target)

    (index_mgw, map_mgw), info_mgw = MGW2_coup(
        source,
        target,
        mu=source_gmm,
        nu=target_gmm,
        C1=component_source,
        C2=component_target,
        coupling=coupling_mgw,
        method="T_mean",
        return_both=True,
        log=True,
    )
    (index_pmgw, map_pmgw), info_pmgw = pMGW2_coup(
        source,
        target,
        Lambda=lambda_component,
        n_components_X=6,
        n_components_Y=7,
        mu=source_gmm,
        nu=target_gmm,
        C1=problem["component_source_cost"],
        C2=problem["component_target_cost"],
        solver_tol=1e-12,
        return_both=True,
        verbose=False,
        log=True,
    )
    method_parameters = {
        "paper_lambda": paper_lambda,
        "lambda_tilde": lambda_tilde,
        "point_solver_lambda": lambda_point,
        "component_solver_lambda": lambda_component,
    }
    ids = {
        "GW2": solve_id("POT gromov_wasserstein", coupling_gw),
        "PGW2": solve_id(
            "PGW_Metric partial_gromov_ver1", coupling_pgw, method_parameters
        ),
        "MGW2": solve_id("POT gromov_wasserstein", coupling_mgw),
        "pMGW2": solve_id(
            "PGW_Metric partial_gromov_ver1", info_pmgw["coupling"], method_parameters
        ),
    }
    return {
        "paper_lambda": paper_lambda,
        "lambda_tilde": lambda_tilde,
        "lambda_point": lambda_point,
        "lambda_component": lambda_component,
        "point_mean_distortion": point_distortion,
        "component_mean_distortion": gw_mean_distortion(
            component_source, component_target, coupling_mgw
        ),
        "coupling_gw": coupling_gw,
        "coupling_pgw": coupling_pgw,
        "coupling_mgw": coupling_mgw,
        "coupling_pmgw": info_pmgw["coupling"],
        "map_gw": map_gw,
        "map_pgw": map_pgw,
        "map_mgw": map_mgw,
        "map_pmgw": map_pmgw,
        "mask_gw": mask_gw,
        "mask_pgw": mask_pgw,
        "index_gw": index_gw,
        "index_pgw": index_pgw,
        "index_mgw": index_mgw,
        "index_pmgw": index_pmgw,
        "solve_ids": ids,
        "runtime_seconds": {
            "GW2": runtime_gw,
            "PGW2": runtime_pgw,
            "MGW2_coupling": runtime_mgw_coupling,
            "MGW2_map": info_mgw["runtime_seconds"],
            "pMGW2": info_pmgw["runtime_seconds"],
        },
        "info_mgw": info_mgw,
        "info_pmgw": info_pmgw,
    }


def _equal_3d_axes(axis, points, margin=0.1):
    minimum, maximum = points.min(axis=0), points.max(axis=0)
    centre = 0.5 * (minimum + maximum)
    half = 0.5 * (maximum - minimum).max() * (1.0 + margin)
    axis.set_xlim(centre[0] - half, centre[0] + half)
    axis.set_ylim(centre[1] - half, centre[1] + half)
    axis.set_zlim(centre[2] - half, centre[2] + half)
    axis.set_box_aspect([1, 1, 1])


def _draw_correspondences(axis, data, targets, title, source_mask=None):
    source, target = data["source"], data["target"]
    axis.scatter(*source.T, s=18, c="blue", label="source", alpha=0.8)
    axis.scatter(*target.T, s=18, c="red", label="target", alpha=0.8)
    rows = np.arange(len(source)) if source_mask is None else np.flatnonzero(source_mask)
    for i, target_point in zip(rows, targets):
        axis.plot(
            [source[i, 0], target_point[0]],
            [source[i, 1], target_point[1]],
            [source[i, 2], target_point[2]],
            color="orange",
            alpha=0.3,
            linewidth=0.7,
        )
    axis.set_title(title)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")
    _equal_3d_axes(axis, np.vstack([source, target]))


def make_figure6(data, solved):
    """Nearest-target assignments derived from the four barycentric maps."""
    target = data["target"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 14), subplot_kw={"projection": "3d"})
    _draw_correspondences(
        axes[0, 0], data, target[solved["index_gw"]], "(a) GW2", solved["mask_gw"]
    )
    _draw_correspondences(
        axes[0, 1], data, target[solved["index_pgw"]], "(b) PGW2", solved["mask_pgw"]
    )
    _draw_correspondences(
        axes[1, 0], data, target[solved["index_mgw"]], "(c) MGW2"
    )
    _draw_correspondences(
        axes[1, 1], data, target[solved["index_pmgw"]], "(d) pMGW2"
    )
    fig.tight_layout()
    return fig


def make_figure7(data, solved):
    """Raw mixture barycentric maps from the same MGW/pMGW solves as Figure 6."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), subplot_kw={"projection": "3d"})
    _draw_correspondences(axes[0], data, solved["map_mgw"], "(a) MGW2")
    _draw_correspondences(axes[1], data, solved["map_pmgw"], "(b) pMGW2")
    fig.tight_layout()
    return fig
