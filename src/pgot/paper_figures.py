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
)
from .partial_ot import (
    display_gmm,
    entropic_partial_barycentric_map,
    entropic_partial_coupling,
    entropic_partial_ot,
)
from .reproducibility import DEFAULT_SEED


def solve_epot_panels(lambdas, epsilons, source, target, cost, cost_scale):
    """Solve each requested Figure 1--3 panel once with residual logs.

    ``cost`` is the normalized component cost matrix (7.1) and ``cost_scale``
    the constant it was divided by; the caller builds both so that the
    normalization stays visible where the experiment is specified.
    """
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


def _plot_epot_grid(records, shape, figsize, source, target):
    nrows, ncols = shape
    if nrows * ncols != len(records):
        raise ValueError(
            f"grid {nrows}x{ncols} does not match {len(records)} solved panels"
        )
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
        label=r"coupling weight $\omega_{kl}^{\varepsilon,\lambda}$",
    )
    return fig


def make_figure1(records, source, target):
    """Figure 1: continuous coupling weights and horizontal colorbar."""
    return _plot_epot_grid(records, (1, 1), (7, 7), source, target)


def make_figure2(records, source, target):
    """Figure 2: the single-row lambda sweep from the paper."""
    return _plot_epot_grid(records, (1, len(records)), (22, 5), source, target)


def make_figure3(records, source, target, shape):
    """Figure 3: one row per epsilon and one column per lambda.

    ``shape`` comes from the caller's own parameter lists, so the grid cannot
    silently disagree with the number of solved panels.
    """
    return _plot_epot_grid(records, shape, (22, 12), source, target)


def solve_figure5(source_points, source, target, lambdas, epsilon):
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


def make_figure5(source_points, target_points, records, times):
    """Figure 5's canonical two-row composite layout."""
    times = np.asarray(times)
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


def solve_point_cloud_matching(
    problem,
    coupling_gw,
    coupling_mgw,
    lambda_point,
    lambda_component,
    method_parameters,
    runtime_gw,
    runtime_mgw_coupling,
):
    """Solve each partial method once; Figures 6 and 7 consume the same maps.

    The balanced couplings are supplied by the caller, and serve twice: they
    are Figures 6 and 7's balanced panels, and they are the starting points of
    the two partial solves. Partial GW is non-convex, and the solver's default
    start is the independent coupling; on the component problem that start
    leaves the solver at the empty coupling for the penalty of Section 7.2,
    even though the matching found from the balanced start has a strictly
    better objective. Passing them in also keeps the promise that every method
    is solved exactly once. ``method_parameters`` is recorded in the solve
    identifiers, and the two balanced runtimes are recorded alongside the
    partial ones so the sidecar still times every solve.
    """
    source, target = problem["source"], problem["target"]
    source_gmm, target_gmm = problem["source_gmm"], problem["target_gmm"]
    point_source = problem["point_source_cost_normalized"]
    point_target = problem["point_target_cost_normalized"]
    component_source = problem["component_source_cost_normalized"]
    component_target = problem["component_target_cost_normalized"]
    p = ot.unif(len(source))
    q = ot.unif(len(target))


    started = perf_counter()
    coupling_pgw = gromov.partial_gromov_ver1(
        point_source,
        point_target,
        p,
        q,
        Lambda=lambda_point,
        G0=np.asarray(coupling_gw, dtype=float).copy(),
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
        n_components_X=source_gmm.K,
        n_components_Y=target_gmm.K,
        mu=source_gmm,
        nu=target_gmm,
        C1=problem["component_source_cost"],
        C2=problem["component_target_cost"],
        init_coupling=coupling_mgw,
        solver_tol=1e-12,
        return_both=True,
        verbose=False,
        log=True,
    )
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
        **method_parameters,
        "lambda_point": lambda_point,
        "lambda_component": lambda_component,
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


# Panel titles spell the method out. The sidecars keep the short GW2/PGW2/
# MGW2/pMGW2 keys as solve identifiers, but a reader of the figure has only
# the title, and "pMGW2" does not say which of the four cells of the
# balanced/partial by point/mixture comparison a panel is.
BALANCED_GW = "Balanced GW"
PARTIAL_GW = "Partial GW"
BALANCED_MIXTURE_GW = "Balanced mixture GW"
PARTIAL_MIXTURE_GW = "Partial mixture GW"


def make_figure6(data, solved):
    """Nearest-target assignments derived from the four barycentric maps."""
    target = data["target"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 14), subplot_kw={"projection": "3d"})
    _draw_correspondences(
        axes[0, 0],
        data,
        target[solved["index_gw"]],
        f"(a) {BALANCED_GW}",
        solved["mask_gw"],
    )
    _draw_correspondences(
        axes[0, 1],
        data,
        target[solved["index_pgw"]],
        f"(b) {PARTIAL_GW}",
        solved["mask_pgw"],
    )
    _draw_correspondences(
        axes[1, 0], data, target[solved["index_mgw"]], f"(c) {BALANCED_MIXTURE_GW}"
    )
    _draw_correspondences(
        axes[1, 1], data, target[solved["index_pmgw"]], f"(d) {PARTIAL_MIXTURE_GW}"
    )
    fig.tight_layout()
    return fig


def make_figure7(data, solved):
    """Raw mixture barycentric maps from the same solves as Figure 6.

    The two panels are the balanced and the partial mixture GW maps, named
    as in Figure 6 so the same method is called the same thing in both.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), subplot_kw={"projection": "3d"})
    _draw_correspondences(
        axes[0], data, solved["map_mgw"], f"(a) {BALANCED_MIXTURE_GW}"
    )
    _draw_correspondences(
        axes[1], data, solved["map_pmgw"], f"(b) {PARTIAL_MIXTURE_GW}"
    )
    fig.tight_layout()
    return fig
