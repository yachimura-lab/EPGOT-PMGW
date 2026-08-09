"""Rebuild the four canonical paper notebooks from concise source cells."""

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "docs" / "notebook"


def code(source):
    return nbf.v4.new_code_cell(dedent(source).strip() + "\n")


def markdown(source):
    return nbf.v4.new_markdown_cell(dedent(source).strip() + "\n")


def notebook(cells):
    return nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.13.15"},
        },
    )


SETUP = """
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from pgot import set_reproducible

set_reproducible()
OUTPUT_DIR = Path("outputs")
METADATA_DIR = Path("metadata")
OUTPUT_DIR.mkdir(exist_ok=True)
METADATA_DIR.mkdir(exist_ok=True)
"""


def figure123_notebook():
    return notebook(
        [
            markdown(
                """
                # Paper Figures 1–3 — entropic partial OT

                Each paper figure has one canonical solve cell and one unique output.
                The paper penalty is passed unchanged to `entropic_partial_ot`, whose
                real-real cost is `M - 2*lambda`. The JSON sidecars record the extended
                marginal residuals and the exact stopping condition for every panel.
                """
            ),
            code(SETUP),
            code(
                """
                import matplotlib.pyplot as plt

                from pgot import (
                    FIGURE_PARAMETERS,
                    make_figure1,
                    make_figure2,
                    make_figure3,
                    paper_gaussian_mixtures,
                    solve_epot_panels,
                    write_figure_sidecar,
                )

                source, target = paper_gaussian_mixtures()
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
                }
                """
            ),
            markdown(
                """
                ## Figure 1

                Continuous coupling-weight colormap with a horizontal colorbar. Component
                labels, pair categories, and the pair legend are intentionally absent.
                """
            ),
            code(
                """
                figure1_records = solve_epot_panels([0.3], [0.01], source, target)
                figure1 = make_figure1(figure1_records)
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
                """
            ),
            markdown("## Figure 2\n\nThe exact paper sweep: one row and five lambda values at epsilon=0.01."),
            code(
                """
                params2 = FIGURE_PARAMETERS[2]
                figure2_records = solve_epot_panels(
                    params2["lambda"], [params2["epsilon"]], source, target
                )
                figure2 = make_figure2(figure2_records)
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
                """
            ),
            markdown(
                """
                ## Figure 3

                Seven lambda columns and three epsilon rows. The color scale is the maximum
                coupling weight across these 21 solves—not the unrelated cost maximum.
                A stalled Sinkhorn panel is refined on the same convex entropic objective,
                and the sidecar records both the fallback and final marginal residuals.
                """
            ),
            code(
                """
                params3 = FIGURE_PARAMETERS[3]
                figure3_records = solve_epot_panels(
                    params3["lambda"], params3["epsilon"], source, target
                )
                figure3 = make_figure3(figure3_records)
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
                """
            ),
        ]
    )


def figure4_notebook():
    return notebook(
        [
            markdown(
                """
                # Paper Figure 4 — partial displacement interpolation

                The component coupling is solved once per `(lambda, epsilon)` row and reused
                for all five interpolation times. Coupling weights are neither thresholded nor
                renormalized, so each density integrates to the partial mass `Z_lambda`.
                """
            ),
            code(SETUP),
            code(
                """
                import matplotlib.pyplot as plt
                import numpy as np
                from matplotlib.colors import LogNorm

                from pgot import (
                    FIGURE_PARAMETERS,
                    entropic_partial_coupling,
                    entropic_partial_displacement_interpolation,
                    paper_gaussian_mixtures,
                    solve_id,
                    submixture_pdf,
                    write_figure_sidecar,
                )

                source, target = paper_gaussian_mixtures()
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
                """
            ),
            code(
                """
                configurations = FIGURE_PARAMETERS[4]["lambda_epsilon"]
                times = FIGURE_PARAMETERS[4]["t"]
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
                """
            ),
            code(
                """
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
                """
            ),
            code(
                """
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
                                rf"$\\lambda={solve['paper_lambda']}$, "
                                rf"$\\varepsilon={solve['epsilon']}$"
                                "\\n"
                                rf"$Z_\\lambda={solve['matched_mass']:.4f}$"
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
                """
            ),
        ]
    )


def figure5_notebook():
    return notebook(
        [
            markdown(
                """
                # Paper Figure 5 — barycentric projection map

                The known Section 7.1 mixtures are used directly. Samples are only evaluation
                points. Each lambda is solved once, and the exact same coupling is reused for
                its map and all five interpolation panels in the two-row paper layout.
                """
            ),
            code(SETUP),
            code(
                """
                import matplotlib.pyplot as plt
                import numpy as np

                from pgot import (
                    DEFAULT_SEED,
                    FIGURE_PARAMETERS,
                    make_figure5,
                    paper_gaussian_mixtures,
                    sample_from_gmm,
                    solve_figure5,
                    write_figure_sidecar,
                )

                source, target = paper_gaussian_mixtures()
                source_points = sample_from_gmm(
                    source.weights, source.comp_mean, source.comp_cov, 2000,
                    rng=np.random.default_rng(DEFAULT_SEED + 10),
                )
                target_points = sample_from_gmm(
                    target.weights, target.comp_mean, target.comp_cov, 2000,
                    rng=np.random.default_rng(DEFAULT_SEED + 11),
                )
                """
            ),
            code(
                """
                params = FIGURE_PARAMETERS[5]
                figure5_records = solve_figure5(
                    source_points,
                    source,
                    target,
                    lambdas=params["lambda"],
                    epsilon=params["epsilon"],
                )
                for record in figure5_records:
                    print(
                        f"lambda={record['paper_lambda']}, epsilon={record['epsilon']}, "
                        f"matched mass={record['matched_mass']:.7f}, solve={record['solve_id'][:12]}"
                    )
                """
            ),
            code(
                """
                figure5 = make_figure5(
                    source_points, target_points, figure5_records, times=params["t"]
                )
                figure5_path = OUTPUT_DIR / "figure5.pdf"
                figure5.savefig(figure5_path, format="pdf", bbox_inches="tight")
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
                        "times": params["t"],
                    },
                    seeds={
                        "source_sampling": DEFAULT_SEED + 10,
                        "target_sampling": DEFAULT_SEED + 11,
                        "fit": None,
                    },
                    notes="Known GMM parameters; canonical dummy-point EPOT and matched-density map (6.2).",
                )
                plt.show()
                """
            ),
        ]
    )


def figure67_notebook():
    return notebook(
        [
            markdown(
                """
                # Paper Figures 6–7 — point and mixture GW

                Source and target rings are independent samples from radius-4 GMMs; ten points
                from a separate ball distribution are added to the target. The 6/7-component
                fit and its W2 matrices are shared. Every method is solved exactly once:
                Figure 7 displays the raw mixture barycentric maps, and Figure 6 applies nearest
                target assignment to those same maps.

                The dimensionality claim is limited to the coupling solve (300×310 versus 6×7).
                At N=300 the GMM fitting cost means this notebook does not claim end-to-end speedup.
                """
            ),
            code(SETUP),
            code(
                """
                import matplotlib.pyplot as plt
                import numpy as np
                import ot

                from pgot import (
                    DEFAULT_SEED,
                    coupling_marginal_diagnostics,
                    fit_figure67_problem,
                    generate_figure67_data,
                    make_figure6,
                    make_figure7,
                    solve_figure67,
                    write_figure_sidecar,
                )

                figure67_data = generate_figure67_data(seed=DEFAULT_SEED)
                print(
                    "source/target radii:",
                    np.hypot(*figure67_data["source"][:, :2].T).mean(),
                    np.hypot(*figure67_data["target_ring"][:, :2].T).mean(),
                )
                print("source z standard deviation:", figure67_data["source"][:, 2].std())
                """
            ),
            code(
                """
                figure67_problem = fit_figure67_problem(
                    figure67_data, random_state=DEFAULT_SEED
                )
                print(
                    "coupling dimensions:",
                    (len(figure67_problem["source"]), len(figure67_problem["target"])),
                    "->",
                    (figure67_problem["source_gmm"].K, figure67_problem["target_gmm"].K),
                )
                """
            ),
            code(
                """
                figure67_solved = solve_figure67(figure67_problem, paper_lambda=0.01)
                print(f"lambda_tilde={figure67_solved['lambda_tilde']:.6f}")
                print(
                    f"solver lambda: point={figure67_solved['lambda_point']:.6f}, "
                    f"component={figure67_solved['lambda_component']:.6f}"
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
                """
            ),
            code(
                """
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
                    "penalty_conversion": "Lambda = lambda_tilde * balanced mean distortion",
                }
                method_panels = []
                for method, key, solver_lambda in [
                    ("GW2", "coupling_gw", None),
                    ("PGW2", "coupling_pgw", figure67_solved["lambda_point"]),
                    ("MGW2", "coupling_mgw", None),
                    ("pMGW2", "coupling_pmgw", figure67_solved["lambda_component"]),
                ]:
                    method_panels.append(
                        {
                            "method": method,
                            "paper_lambda": 0.01 if "P" in method or method.startswith("p") else None,
                            "solver_lambda": solver_lambda,
                            "lambda_tilde": figure67_solved["lambda_tilde"],
                            "solver": (
                                "POT gromov_wasserstein"
                                if method in {"GW2", "MGW2"}
                                else "PGW_Metric partial_gromov_ver1"
                            ),
                            "solve_id": figure67_solved["solve_ids"][method],
                            "coupling": figure67_solved[key],
                            "runtime_seconds": figure67_solved["runtime_seconds"].get(
                                method, figure67_solved["runtime_seconds"].get(f"{method}_map")
                            ),
                            **diagnostics[method],
                        }
                    )
                """
            ),
            markdown("## Figure 6 — barycentric projection followed by nearest-target assignment"),
            code(
                """
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
                """
            ),
            markdown("## Figure 7 — raw maps from the same MGW2 and pMGW2 solves"),
            code(
                """
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
                    notes="solve_id values are identical to the MGW2/pMGW2 panels in Figure 6.",
                )
                plt.show()
                """
            ),
        ]
    )


def main():
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "figure1_2_3.ipynb": figure123_notebook(),
        "figure4.ipynb": figure4_notebook(),
        "figure5.ipynb": figure5_notebook(),
        "PMGW_corrected.ipynb": figure67_notebook(),
    }
    for name, value in outputs.items():
        nbf.write(value, NOTEBOOK_DIR / name)
        print(f"wrote {NOTEBOOK_DIR / name}")


if __name__ == "__main__":
    main()
