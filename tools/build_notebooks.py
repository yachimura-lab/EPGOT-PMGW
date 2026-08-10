"""Rebuild the four canonical paper notebooks from their source cells.

The notebooks are build artifacts: this file is the single place where their
cells are written. Section 7 parameters live in the emitted cells, not in
`pgot`, so the published page can be checked against the paper directly.
Edit this file and re-run it rather than editing a notebook by hand.
"""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "docs" / "notebook"

CELLS = {
    "figure1_2_3.ipynb": [
        ("markdown", r'''
# Paper Figures 1–3 — entropic partial OT

Each paper figure has one canonical solve cell and one unique output.
The paper penalty is passed unchanged to `entropic_partial_ot`, whose
real-real cost is `M - 2*lambda`. The JSON sidecars record the extended
marginal residuals and the exact stopping condition for every panel.
'''),
        ("code", r'''
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from pgot import set_reproducible

set_reproducible()
OUTPUT_DIR = Path("outputs")
METADATA_DIR = Path("metadata")
OUTPUT_DIR.mkdir(exist_ok=True)
METADATA_DIR.mkdir(exist_ok=True)
'''),
        ("code", r'''
import matplotlib.pyplot as plt
import numpy as np

from pgot import (
    GaussianMixture,
    gaussian_W,
    make_figure1,
    make_figure2,
    make_figure3,
    solve_epot_panels,
    write_figure_sidecar,
)

# ---------------------------------------------------------------
# Section 7.1 mixtures. These values are the experiment: they are
# written here, not imported, so that the published page can be
# checked against the paper directly.
# ---------------------------------------------------------------
source = GaussianMixture(
    np.array([0.5, 0.5]),                                    # alpha_A
    np.array([[-8.0, -5.0], [1.0, -8.0]]),                   # m_A^(1), m_A^(2)
    0.2 * np.array([[[1.0, 0.2], [0.2, 1.0]],                # Sigma_A^(1)
                    [[1.0, -0.1], [-0.1, 1.0]]]),            # Sigma_A^(2)
)
target = GaussianMixture(
    np.array([0.45, 0.45, 0.1]),                             # alpha_B
    np.array([[-3.5, 3.5], [3.5, 3.5], [9.0, 7.5]]),         # m_B^(1..3)
    0.2 * np.array([[[1.0, -0.1], [-0.1, 1.0]],              # Sigma_B^(1)
                    [[1.0, 0.2], [0.2, 1.0]],                # Sigma_B^(2)
                    [[1.0, 0.0], [0.0, 1.0]]]),              # Sigma_B^(3), isotropic
)

# Normalized component cost (7.1): squared Gaussian W2 divided by its maximum.
cross_cost = gaussian_W(source, target)
cost_scale = float(cross_cost.max())
cost = cross_cost / cost_scale

gmm_inputs = {
    "source": {
        "weights": source.weights,
        "means": source.comp_mean,
        "covariances": source.comp_cov,
    },
    "target": {
        "weights": target.weights,
        "means": target.comp_mean,
        "covariances": target.comp_cov,
    },
    "cost": "squared Gaussian W2 divided by its cross-matrix maximum",
    "cost_scale": cost_scale,
}
'''),
        ("markdown", r'''
## Figure 1

Continuous coupling-weight colormap with a horizontal colorbar. Component
labels, pair categories, and the pair legend are intentionally absent.
'''),
        ("code", r'''
# Section 7.1.1: lambda = 0.3, epsilon = 0.01
FIGURE1_LAMBDA = 0.3
FIGURE1_EPSILON = 0.01
figure1_records = solve_epot_panels(
    [FIGURE1_LAMBDA], [FIGURE1_EPSILON], source, target, cost, cost_scale
)
figure1 = make_figure1(figure1_records, source, target)
figure1_path = OUTPUT_DIR / "figure1.pdf"
figure1.savefig(figure1_path, format="pdf", bbox_inches="tight")
write_figure_sidecar(
    METADATA_DIR / "figure1.json",
    figure=1,
    outputs=[figure1_path],
    panels=figure1_records,
    inputs=gmm_inputs,
    seeds={"sampling": None, "fit": None},
    notes="Known GMMs; no sampling or fitting.",
)
plt.show()
'''),
        ("markdown", r'''
## Figure 2

The exact paper sweep: one row and five lambda values at epsilon=0.01.
'''),
        ("code", r'''
# Section 7.1.2: epsilon = 0.01, five lambda values including lambda = 0
FIGURE2_LAMBDAS = [0.0, 0.125, 0.25, 0.375, 0.5]
FIGURE2_EPSILON = 0.01
figure2_records = solve_epot_panels(
    FIGURE2_LAMBDAS, [FIGURE2_EPSILON], source, target, cost, cost_scale
)
figure2 = make_figure2(figure2_records, source, target)
figure2_path = OUTPUT_DIR / "figure2.pdf"
figure2.savefig(figure2_path, format="pdf", bbox_inches="tight")
write_figure_sidecar(
    METADATA_DIR / "figure2.json",
    figure=2,
    outputs=[figure2_path],
    panels=figure2_records,
    inputs=gmm_inputs,
    seeds={"sampling": None, "fit": None},
    notes="Exact 1 x 5 paper sweep.",
)
print([(r["paper_lambda"], round(r["matched_mass"], 7)) for r in figure2_records])
plt.show()
'''),
        ("markdown", r'''
## Figure 3

Seven lambda columns and three epsilon rows. The color scale is the maximum
coupling weight across these 21 solves—not the unrelated cost maximum.
A stalled Sinkhorn panel is refined on the same convex entropic objective,
and the sidecar records both the fallback and final marginal residuals.
'''),
        ("code", r'''
# Section 7.1.3: seven lambda columns and three epsilon rows
FIGURE3_LAMBDAS = [0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5]
FIGURE3_EPSILONS = [0.01, 0.11, 0.21]
figure3_records = solve_epot_panels(
    FIGURE3_LAMBDAS, FIGURE3_EPSILONS, source, target, cost, cost_scale
)
figure3 = make_figure3(
    figure3_records, source, target, (len(FIGURE3_EPSILONS), len(FIGURE3_LAMBDAS))
)
figure3_path = OUTPUT_DIR / "figure3.pdf"
figure3.savefig(figure3_path, format="pdf", bbox_inches="tight")
write_figure_sidecar(
    METADATA_DIR / "figure3.json",
    figure=3,
    outputs=[figure3_path],
    panels=figure3_records,
    inputs=gmm_inputs,
    seeds={"sampling": None, "fit": None},
    notes="3 x 7 paper sweep; shared coupling-weight color scale.",
)
for record in figure3_records:
    print(
        f"lambda={record['paper_lambda']:.4f}, epsilon={record['epsilon']:.2f}, "
        f"mass={record['matched_mass']:.7f}, solver={record['solver']}, "
        f"residual={max(record['row_residual'], record['column_residual']):.2e}"
    )
plt.show()
'''),
    ],
    "figure4.ipynb": [
        ("markdown", r'''
# Paper Figure 4 — partial displacement interpolation

The component coupling is solved once per `(lambda, epsilon)` row and reused
for all five interpolation times. Coupling weights are neither thresholded nor
renormalized, so each density integrates to the partial mass `Z_lambda`.
'''),
        ("code", r'''
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from pgot import set_reproducible

set_reproducible()
OUTPUT_DIR = Path("outputs")
METADATA_DIR = Path("metadata")
OUTPUT_DIR.mkdir(exist_ok=True)
METADATA_DIR.mkdir(exist_ok=True)
'''),
        ("code", r'''
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from pgot import (
    GaussianMixture,
    entropic_partial_coupling,
    entropic_partial_displacement_interpolation,
    solve_id,
    submixture_pdf,
    write_figure_sidecar,
)

# ---------------------------------------------------------------
# Section 7.1 mixtures. These values are the experiment: they are
# written here, not imported, so that the published page can be
# checked against the paper directly.
# ---------------------------------------------------------------
source = GaussianMixture(
    np.array([0.5, 0.5]),                                    # alpha_A
    np.array([[-8.0, -5.0], [1.0, -8.0]]),                   # m_A^(1), m_A^(2)
    0.2 * np.array([[[1.0, 0.2], [0.2, 1.0]],                # Sigma_A^(1)
                    [[1.0, -0.1], [-0.1, 1.0]]]),            # Sigma_A^(2)
)
target = GaussianMixture(
    np.array([0.45, 0.45, 0.1]),                             # alpha_B
    np.array([[-3.5, 3.5], [3.5, 3.5], [9.0, 7.5]]),         # m_B^(1..3)
    0.2 * np.array([[[1.0, -0.1], [-0.1, 1.0]],              # Sigma_B^(1)
                    [[1.0, 0.2], [0.2, 1.0]],                # Sigma_B^(2)
                    [[1.0, 0.0], [0.0, 1.0]]]),              # Sigma_B^(3), isotropic
)
axis_values = np.linspace(-12, 12, 400)
GX, GY = np.meshgrid(axis_values, axis_values)
grid_points = np.column_stack([GX.ravel(), GY.ravel()])

def log_levels(maximum, count=7):
    return np.logspace(np.log10(maximum * 1e-3), np.log10(maximum), count)

def draw_density(axis, values, levels, cmap):
    positive = np.maximum(values.reshape(GX.shape), np.finfo(float).tiny)
    axis.contourf(
        GX, GY, positive, levels=levels, cmap=cmap,
        norm=LogNorm(vmin=levels[0], vmax=levels[-1]), alpha=1,
    )
    axis.set_aspect("equal", adjustable="box")
'''),
        ("code", r'''
# Section 7.1.4: three (lambda, epsilon) rows and five interpolation times
configurations = [[0.25, 0.03], [0.25, 0.1], [0.5, 0.03]]
times = [0.0, 0.25, 0.5, 0.75, 1.0]
row_solves = []
for paper_lambda, epsilon in configurations:
    coupling, info = entropic_partial_coupling(
        source, target, epsilon=epsilon, Lambda=paper_lambda, log=True
    )
    row_solves.append(
        {
            "paper_lambda": paper_lambda,
            "epsilon": epsilon,
            "coupling": coupling,
            "solve_id": solve_id(
                info["solver"], coupling,
                {"paper_lambda": paper_lambda, "epsilon": epsilon},
            ),
            **info,
        }
    )
    print(
        f"lambda={paper_lambda}, epsilon={epsilon}, "
        f"matched mass={coupling.sum():.7f}"
    )
'''),
        ("code", r'''
density_source = source.pdf(grid_points)
density_target = target.pdf(grid_points)
panels = {}
panel_records = []
for row, solve in enumerate(row_solves):
    for column, t in enumerate(times):
        weights, means, covariances = entropic_partial_displacement_interpolation(
            source,
            target,
            t,
            epsilon=solve["epsilon"],
            Lambda=solve["paper_lambda"],
            coupling=solve["coupling"],
        )
        panels[(row, column)] = submixture_pdf(
            grid_points, weights, means, covariances
        )
        panel_records.append(
            {
                "row": row,
                "column": column,
                "t": t,
                "paper_lambda": solve["paper_lambda"],
                "solver_lambda": solve["paper_lambda"],
                "epsilon": solve["epsilon"],
                "matched_mass": solve["matched_mass"],
                "row_residual": solve["row_residual"],
                "column_residual": solve["column_residual"],
                "runtime_seconds": solve["runtime_seconds"],
                "solver": solve["solver"],
                "solve_id": solve["solve_id"],
                "coupling": solve["coupling"],
            }
        )

levels_source = log_levels(density_source.max())
levels_target = log_levels(density_target.max())
levels_interpolation = log_levels(max(value.max() for value in panels.values()))
'''),
        ("code", r'''
figure4, axes = plt.subplots(3, 5, figsize=(22, 12))
for row, solve in enumerate(row_solves):
    for column, t in enumerate(times):
        axis = axes[row, column]
        draw_density(axis, density_source, levels_source, "Reds")
        draw_density(axis, density_target, levels_target, "Blues")
        draw_density(axis, panels[(row, column)], levels_interpolation, "viridis")
        axis.set_xlim(-12, 12)
        axis.set_ylim(-12, 12)
        if row == 0:
            axis.set_title(f"t={t:.2f}")
        if column == 0:
            axis.set_ylabel(
                rf"$\lambda={solve['paper_lambda']}$, "
                rf"$\varepsilon={solve['epsilon']}$"
                "\n"
                rf"$Z_\lambda={solve['matched_mass']:.4f}$"
            )
        axis.set_xticks([])
        axis.set_yticks([])
figure4.tight_layout()
figure4_path = OUTPUT_DIR / "figure4.pdf"
figure4.savefig(figure4_path, format="pdf", bbox_inches="tight")
write_figure_sidecar(
    METADATA_DIR / "figure4.json",
    figure=4,
    outputs=[figure4_path],
    panels=panel_records,
    inputs={
        "source_weights": source.weights,
        "source_means": source.comp_mean,
        "source_covariances": source.comp_cov,
        "target_weights": target.weights,
        "target_means": target.comp_mean,
        "target_covariances": target.comp_cov,
        "density_grid": {"limits": [-12, 12], "points_per_axis": 400},
        "shared_density_levels": levels_interpolation,
    },
    seeds={"sampling": None, "fit": None},
    notes="Equation (4.17) closed form; (4.18) sub-probability mass.",
)
plt.show()
'''),
    ],
    "figure5.ipynb": [
        ("markdown", r'''
# Paper Figure 5 — barycentric projection map

The known Section 7.1 mixtures are used directly. Samples are only evaluation
points. Each lambda is solved once, and the exact same coupling is reused for
its map and all five interpolation panels in the two-row paper layout. The
figure is saved as a PNG at its native resolution without a DPI override.
'''),
        ("code", r'''
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from pgot import set_reproducible

set_reproducible()
OUTPUT_DIR = Path("outputs")
METADATA_DIR = Path("metadata")
OUTPUT_DIR.mkdir(exist_ok=True)
METADATA_DIR.mkdir(exist_ok=True)
'''),
        ("code", r'''
import matplotlib.pyplot as plt
import numpy as np

from pgot import (
    DEFAULT_SEED,
    GaussianMixture,
    make_figure5,
    sample_from_gmm,
    solve_figure5,
    write_figure_sidecar,
)

# ---------------------------------------------------------------
# Section 7.1 mixtures. These values are the experiment: they are
# written here, not imported, so that the published page can be
# checked against the paper directly.
# ---------------------------------------------------------------
source = GaussianMixture(
    np.array([0.5, 0.5]),                                    # alpha_A
    np.array([[-8.0, -5.0], [1.0, -8.0]]),                   # m_A^(1), m_A^(2)
    0.2 * np.array([[[1.0, 0.2], [0.2, 1.0]],                # Sigma_A^(1)
                    [[1.0, -0.1], [-0.1, 1.0]]]),            # Sigma_A^(2)
)
target = GaussianMixture(
    np.array([0.45, 0.45, 0.1]),                             # alpha_B
    np.array([[-3.5, 3.5], [3.5, 3.5], [9.0, 7.5]]),         # m_B^(1..3)
    0.2 * np.array([[[1.0, -0.1], [-0.1, 1.0]],              # Sigma_B^(1)
                    [[1.0, 0.2], [0.2, 1.0]],                # Sigma_B^(2)
                    [[1.0, 0.0], [0.0, 1.0]]]),              # Sigma_B^(3), isotropic
)

# Section 7.1.5: 2000 samples from each mixture, used only as evaluation
# points for the map and as the scatter overlay.
SAMPLES_PER_MIXTURE = 2000
source_points = sample_from_gmm(
    source.weights, source.comp_mean, source.comp_cov, SAMPLES_PER_MIXTURE,
    rng=np.random.default_rng(DEFAULT_SEED + 10),
)
target_points = sample_from_gmm(
    target.weights, target.comp_mean, target.comp_cov, SAMPLES_PER_MIXTURE,
    rng=np.random.default_rng(DEFAULT_SEED + 11),
)
'''),
        ("code", r'''
# Section 7.1.5: lambda in {0.25, 0.5} at a fixed epsilon = 0.03
FIGURE5_LAMBDAS = [0.25, 0.5]
FIGURE5_EPSILON = 0.03
FIGURE5_TIMES = [0.2, 0.4, 0.6, 0.8, 1.0]
figure5_records = solve_figure5(
    source_points,
    source,
    target,
    lambdas=FIGURE5_LAMBDAS,
    epsilon=FIGURE5_EPSILON,
)
for record in figure5_records:
    print(
        f"lambda={record['paper_lambda']}, epsilon={record['epsilon']}, "
        f"matched mass={record['matched_mass']:.7f}, solve={record['solve_id'][:12]}"
    )
'''),
        ("code", r'''
figure5 = make_figure5(
    source_points, target_points, figure5_records, times=FIGURE5_TIMES
)
figure5_path = OUTPUT_DIR / "figure5.png"
figure5.savefig(figure5_path, format="png", bbox_inches="tight")
write_figure_sidecar(
    METADATA_DIR / "figure5.json",
    figure=5,
    outputs=[figure5_path],
    panels=figure5_records,
    inputs={
        "source_gmm": {
            "weights": source.weights,
            "means": source.comp_mean,
            "covariances": source.comp_cov,
        },
        "target_gmm": {
            "weights": target.weights,
            "means": target.comp_mean,
            "covariances": target.comp_cov,
        },
        "source_evaluation_points": source_points,
        "target_plot_points": target_points,
        "times": FIGURE5_TIMES,
    },
    seeds={
        "source_sampling": DEFAULT_SEED + 10,
        "target_sampling": DEFAULT_SEED + 11,
        "fit": None,
    },
    notes="Known GMM parameters; canonical dummy-point EPOT and matched-density map (6.2).",
)
plt.show()
'''),
    ],
    "PMGW_corrected.ipynb": [
        ("markdown", r'''
# Paper Figures 6–7 — point and mixture GW

Source and target rings are independent samples from radius-4 GMMs; ten points
from a separate ball distribution are added to the target. The 6/7-component
fit and its W2 matrices are shared. Every method is solved exactly once:
Figure 7 displays the raw mixture barycentric maps, and Figure 6 applies nearest
target assignment to those same maps.

The dimensionality claim is limited to the coupling solve (300×310 versus 6×7).
'''),
        ("code", r'''
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from pgot import set_reproducible

set_reproducible()
OUTPUT_DIR = Path("outputs")
METADATA_DIR = Path("metadata")
OUTPUT_DIR.mkdir(exist_ok=True)
METADATA_DIR.mkdir(exist_ok=True)
'''),
        ("code", r'''
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import ot
from sklearn.mixture import GaussianMixture as SkGaussianMixture

from pgot import (
    DEFAULT_SEED,
    GaussianMixture,
    MGW2_GM_coup,
    coupling_marginal_diagnostics,
    gaussian_cost_matrix,
    gw_mean_distortion,
    make_figure6,
    make_figure7,
    solve_point_cloud_matching,
    write_figure_sidecar,
)

# ---------------------------------------------------------------
# Section 7.2 point clouds. The source ring lies in the plane z = 0
# and the target ring is its translate into z = HEIGHT. The target
# additionally carries a cluster of noise points, which the balanced
# methods have to match and the partial methods may leave unmatched.
# ---------------------------------------------------------------
COMPONENTS = 6            # six equally weighted components per ring
POINTS = 300              # points sampled from each ring
SIGMA = 0.2               # component standard deviation, covariance 0.04 I_3
RADIUS = 4.0              # ring radius, identical for source and target
HEIGHT = 10.0             # the target ring lies in the plane z = HEIGHT
NOISE_POINTS = 10         # outlying points added to the target
NOISE_CENTRE = np.array([2.5 * RADIUS, 0.0, HEIGHT])  # centre of the noise ball
NOISE_RADIUS = 0.3        # radius of that ball

# Independent generators: the source and the target ring must be separate
# samples, not an affine image of one another.
rng_source = np.random.default_rng(DEFAULT_SEED)
rng_target = np.random.default_rng(DEFAULT_SEED + 1)
rng_noise = np.random.default_rng(DEFAULT_SEED + 2)

theta = np.linspace(0, 2 * np.pi, COMPONENTS, endpoint=False)
source_means = np.stack(
    [RADIUS * np.cos(theta), RADIUS * np.sin(theta), np.zeros(COMPONENTS)], axis=1
)
target_means = source_means + np.array([0.0, 0.0, HEIGHT])
covariance = SIGMA**2 * np.eye(3)


def sample_ring(rng, means):
    return np.vstack(
        [rng.multivariate_normal(mean, covariance, POINTS // COMPONENTS)
         for mean in means]
    )


source_points = sample_ring(rng_source, source_means)
target_ring = sample_ring(rng_target, target_means)

# Noise: uniform in a ball, a different distribution from the ring components.
directions = rng_noise.normal(size=(NOISE_POINTS, 3))
directions /= np.linalg.norm(directions, axis=1, keepdims=True)
noise_radii = NOISE_RADIUS * rng_noise.uniform(0, 1, NOISE_POINTS) ** (1 / 3)
noise_points = NOISE_CENTRE + directions * noise_radii[:, None]

figure67_data = {
    "source": source_points,
    "target_ring": target_ring,
    "noise": noise_points,
    "target": np.vstack([target_ring, noise_points]),
    "source_means": source_means,
    "target_means": target_means,
    "covariance": covariance,
    "seed": DEFAULT_SEED,
    "radius": RADIUS,
    "height": HEIGHT,
    "noise_radius": NOISE_RADIUS,
    "noise_center": NOISE_CENTRE,
}

print(
    "source/target radii:",
    np.hypot(*figure67_data["source"][:, :2].T).mean(),
    np.hypot(*figure67_data["target_ring"][:, :2].T).mean(),
)
print("source z standard deviation:", figure67_data["source"][:, 2].std())
'''),
        ("code", r'''
# ---------------------------------------------------------------
# One fit shared by every mixture method, and both cost pairs.
# Section 7.2 fits six components to the source and seven to the
# target, and normalizes every cost matrix before solving.
# ---------------------------------------------------------------
SOURCE_COMPONENTS = 6
TARGET_COMPONENTS = 7


def fit_mixture(points, components):
    model = SkGaussianMixture(
        n_components=components,
        covariance_type="full",
        random_state=DEFAULT_SEED,
    ).fit(points)
    return GaussianMixture(model.weights_, model.means_, model.covariances_)


source_gmm = fit_mixture(figure67_data["source"], SOURCE_COMPONENTS)
target_gmm = fit_mixture(figure67_data["target"], TARGET_COMPONENTS)

# Each representation is normalized by the maximum shared by its own pair of
# intra-domain cost matrices.
component_source = gaussian_cost_matrix(source_gmm)
component_target = gaussian_cost_matrix(target_gmm)
component_scale = float(max(component_source.max(), component_target.max()))
# metric="sqeuclidean" is POT's default; naming it keeps the notebook
# explicit that these are the intra-space squared-distance matrices.
point_source = ot.dist(
    figure67_data["source"], figure67_data["source"], metric="sqeuclidean"
)
point_target = ot.dist(
    figure67_data["target"], figure67_data["target"], metric="sqeuclidean"
)
point_scale = float(max(point_source.max(), point_target.max()))

figure67_problem = {
    **figure67_data,
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
    "fit_random_state": DEFAULT_SEED,
}
print(
    "coupling dimensions:",
    (len(figure67_problem["source"]), len(figure67_problem["target"])),
    "->",
    (source_gmm.K, target_gmm.K),
)
'''),
        ("code", r'''
# ---------------------------------------------------------------
# Penalty. Section 7.2 sets lambda = 0.01 for both partial methods, and
# both solvers receive that number unchanged, on cost matrices that have
# already been divided by their own pair maximum.
#
# Partial GW is non-convex, so the Frank-Wolfe start matters. The solver
# default is the independent coupling, which on the 6 x 7 component
# problem spreads every source component over the noise component too and
# stalls at the empty coupling, even though the matching reached from the
# balanced start scores a strictly better objective. Both partial solves
# therefore start at the balanced coupling for the same representation,
# which `solve_point_cloud_matching` receives below.
# ---------------------------------------------------------------
PAPER_LAMBDA = 0.01

point_source_n = figure67_problem["point_source_cost_normalized"]
point_target_n = figure67_problem["point_target_cost_normalized"]
component_source_n = figure67_problem["component_source_cost_normalized"]
component_target_n = figure67_problem["component_target_cost_normalized"]

p = ot.unif(len(figure67_problem["source"]))
q = ot.unif(len(figure67_problem["target"]))
started = perf_counter()
coupling_gw = ot.gromov.gromov_wasserstein(
    point_source_n, point_target_n, p, q, loss_fun="square_loss"
)
runtime_gw = perf_counter() - started

started = perf_counter()
coupling_mgw = MGW2_GM_coup(
    source_gmm, target_gmm, C1=component_source_n, C2=component_target_n
)
runtime_mgw_coupling = perf_counter() - started

lambda_point = PAPER_LAMBDA
lambda_component = PAPER_LAMBDA

figure67_solved = solve_point_cloud_matching(
    figure67_problem,
    coupling_gw=coupling_gw,
    coupling_mgw=coupling_mgw,
    lambda_point=lambda_point,
    lambda_component=lambda_component,
    # These keys feed the solve identifiers; keep them stable so that a
    # refactor does not silently change the recorded provenance.
    method_parameters={
        "paper_lambda": PAPER_LAMBDA,
        "point_solver_lambda": lambda_point,
        "component_solver_lambda": lambda_component,
        "partial_init": "balanced coupling of the same representation",
    },
    runtime_gw=runtime_gw,
    runtime_mgw_coupling=runtime_mgw_coupling,
)
# Reported for the record: the balanced couplings must match the noise, and
# the distortion they carry is what the partial couplings shed.
figure67_solved["point_mean_distortion"] = gw_mean_distortion(
    point_source_n, point_target_n, coupling_gw
)
figure67_solved["component_mean_distortion"] = gw_mean_distortion(
    component_source_n, component_target_n, coupling_mgw
)
print(
    f"solver lambda: point={lambda_point:.6f}, "
    f"component={lambda_component:.6f}"
)

for method, key in [
    ("GW2", "coupling_gw"),
    ("PGW2", "coupling_pgw"),
    ("MGW2", "coupling_mgw"),
    ("pMGW2", "coupling_pmgw"),
]:
    print(
        f"{method}: mass={figure67_solved[key].sum():.7f}, "
        f"solve={figure67_solved['solve_ids'][method][:12]}"
    )
'''),
        ("code", r'''
p = ot.unif(len(figure67_problem["source"]))
q = ot.unif(len(figure67_problem["target"]))
source_weights = figure67_problem["source_gmm"].weights
target_weights = figure67_problem["target_gmm"].weights
diagnostics = {
    "GW2": coupling_marginal_diagnostics(
        figure67_solved["coupling_gw"], p, q
    ),
    "PGW2": coupling_marginal_diagnostics(
        figure67_solved["coupling_pgw"], p, q, partial=True
    ),
    "MGW2": coupling_marginal_diagnostics(
        figure67_solved["coupling_mgw"], source_weights, target_weights
    ),
    "pMGW2": coupling_marginal_diagnostics(
        figure67_solved["coupling_pmgw"],
        source_weights,
        target_weights,
        partial=True,
    ),
}

common_inputs = {
    "source_points": figure67_problem["source"],
    "target_ring_points": figure67_problem["target_ring"],
    "target_noise_points": figure67_problem["noise"],
    "source_gmm": {
        "weights": source_weights,
        "means": figure67_problem["source_gmm"].comp_mean,
        "covariances": figure67_problem["source_gmm"].comp_cov,
    },
    "target_gmm": {
        "weights": target_weights,
        "means": figure67_problem["target_gmm"].comp_mean,
        "covariances": figure67_problem["target_gmm"].comp_cov,
    },
    "point_cost_scale": figure67_problem["point_cost_scale"],
    "component_cost_scale": figure67_problem["component_cost_scale"],
}

# Every panel carries what its own solve was given, so that no reader has to
# pair a penalty with the right entry of common_inputs to interpret it:
# paper_lambda is what Section 7.2 states, solver_lambda is the number the
# solver received, cost_normalization_scale is the constant its cost matrices
# were divided by, and partial_init is the starting point of the solve. The
# balanced methods take no penalty and no starting point, and record null.
POINT_SCALE = figure67_problem["point_cost_scale"]
COMPONENT_SCALE = figure67_problem["component_cost_scale"]
PARTIAL_INIT = "balanced coupling of the same representation"

method_panels = []
for method, key, solver_lambda, cost_scale in [
    ("GW2", "coupling_gw", None, POINT_SCALE),
    ("PGW2", "coupling_pgw", figure67_solved["lambda_point"], POINT_SCALE),
    ("MGW2", "coupling_mgw", None, COMPONENT_SCALE),
    ("pMGW2", "coupling_pmgw", figure67_solved["lambda_component"],
     COMPONENT_SCALE),
]:
    partial = method in {"PGW2", "pMGW2"}
    method_panels.append(
        {
            "method": method,
            "paper_lambda": PAPER_LAMBDA if partial else None,
            "solver_lambda": solver_lambda,
            "cost_normalization_scale": cost_scale,
            "partial_init": PARTIAL_INIT if partial else None,
            "solver": (
                "PGW_Metric partial_gromov_ver1"
                if partial
                else "POT gromov_wasserstein"
            ),
            "solve_id": figure67_solved["solve_ids"][method],
            "coupling": figure67_solved[key],
            "runtime_seconds": figure67_solved["runtime_seconds"].get(
                method, figure67_solved["runtime_seconds"].get(f"{method}_map")
            ),
            **diagnostics[method],
        }
    )
'''),
        ("markdown", r'''
## Figure 6 — barycentric projection followed by nearest-target assignment
'''),
        ("code", r'''
figure6 = make_figure6(figure67_data, figure67_solved)
figure6_path = OUTPUT_DIR / "figure6.pdf"
figure6.savefig(figure6_path, format="pdf", bbox_inches="tight")
write_figure_sidecar(
    METADATA_DIR / "figure6.json",
    figure=6,
    outputs=[figure6_path],
    panels=method_panels,
    inputs=common_inputs,
    seeds={
        "source_sampling": DEFAULT_SEED,
        "target_sampling": DEFAULT_SEED + 1,
        "noise_sampling": DEFAULT_SEED + 2,
        "fit": DEFAULT_SEED,
    },
    notes="All four methods use barycentric projections before nearest assignment.",
)
plt.show()
'''),
        ("markdown", r'''
## Figure 7 — raw maps from the same balanced and partial mixture GW solves
'''),
        ("code", r'''
figure7 = make_figure7(figure67_data, figure67_solved)
figure7_path = OUTPUT_DIR / "figure7.pdf"
figure7.savefig(figure7_path, format="pdf", bbox_inches="tight")
mixture_panels = [
    panel for panel in method_panels if panel["method"] in {"MGW2", "pMGW2"}
]
write_figure_sidecar(
    METADATA_DIR / "figure7.json",
    figure=7,
    outputs=[figure7_path],
    panels=mixture_panels,
    inputs=common_inputs,
    seeds={
        "source_sampling": DEFAULT_SEED,
        "target_sampling": DEFAULT_SEED + 1,
        "noise_sampling": DEFAULT_SEED + 2,
        "fit": DEFAULT_SEED,
    },
    notes=(
        "solve_id values are identical to the balanced and partial mixture "
        "GW panels in Figure 6."
    ),
)
plt.show()
'''),
    ],
}


def build(cells):
    made = []
    for kind, source in cells:
        text = source.strip("\n") + "\n"
        made.append(
            nbf.v4.new_code_cell(text)
            if kind == "code"
            else nbf.v4.new_markdown_cell(text)
        )
    return nbf.v4.new_notebook(
        cells=made,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.13.15"},
        },
    )


def main():
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    for name, cells in CELLS.items():
        nbf.write(build(cells), NOTEBOOK_DIR / name)
        print(f"wrote {NOTEBOOK_DIR / name}")


if __name__ == "__main__":
    main()
